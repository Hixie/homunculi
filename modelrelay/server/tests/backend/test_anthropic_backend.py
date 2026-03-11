"""
Tests for AnthropicBackend class behaviour (not just SSE parsing).
HTTP calls are intercepted with unittest.mock.
"""
from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch, MagicMock

from ...backends.anthropic import AnthropicBackend
from ...backends.base import EventType, NormalizedUsage


# ── SSE stream helpers ────────────────────────────────────────────────────────

def _sse_bytes(*events: dict) -> bytes:
    lines = []
    for ev in events:
        lines.append(f"data: {json.dumps(ev)}\n".encode())
        lines.append(b"\n")
    return b"".join(lines)


def _make_response(body: bytes, headers: dict | None = None):
    h = {
        "anthropic-ratelimit-requests-remaining": "490",
        "anthropic-ratelimit-tokens-remaining":   "87500",
        "anthropic-ratelimit-requests-reset":     "2024-03-15T09:23:00Z",
    }
    if headers:
        h.update(headers)
    resp = MagicMock()
    resp.headers = h
    resp.__iter__ = lambda self: iter(body.splitlines(keepends=True))
    resp.__enter__ = lambda self: self
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _simple_turn_sse(text="hello", input_tokens=5, output_tokens=3) -> bytes:
    return _sse_bytes(
        {"type": "message_start",
         "message": {"usage": {"input_tokens": input_tokens}}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {},
         "usage": {"output_tokens": output_tokens}},
        {"type": "message_stop"},
    )


async def _collect_events_until_done(backend, sse_body: bytes) -> list:
    """
    Patch urlopen, send one turn, and collect all events until TURN_DONE or ERROR.
    Properly manages the async generator lifecycle.
    """
    with patch("urllib.request.urlopen", return_value=_make_response(sse_body)):
        await backend.send_turn("user message")
        events = []
        gen = backend.events()
        try:
            async for ev in gen:
                events.append(ev)
                if ev.type in (EventType.TURN_DONE, EventType.ERROR):
                    break
        finally:
            await gen.aclose()
    return events


async def _drain_connected(backend):
    """Consume the CONNECTED event from the queue directly."""
    ev = await asyncio.wait_for(backend._queue.get(), timeout=1.0)
    assert ev.type == EventType.CONNECTED, f"Expected CONNECTED, got {ev.type}"


# ── tests ─────────────────────────────────────────────────────────────────────

class TestAnthropicBackendHistory(unittest.IsolatedAsyncioTestCase):
    """Assistant turn must be stored in history so multi-turn sessions work."""

    async def test_assistant_content_appended_after_turn(self):
        """After a completed turn the assistant message is in history."""
        b = AnthropicBackend(api_key="test-key")
        await b.connect()
        await _drain_connected(b)

        await _collect_events_until_done(b, _simple_turn_sse("hi"))

        self.assertEqual(len(b._history), 2)
        self.assertEqual(b._history[0]["role"], "user")
        self.assertEqual(b._history[1]["role"], "assistant")

    async def test_second_turn_includes_prior_assistant_content(self):
        """On the second POST the history has user/assistant/user ordering."""
        b = AnthropicBackend(api_key="test-key")
        await b.connect()
        await _drain_connected(b)

        # First turn
        await _collect_events_until_done(b, _simple_turn_sse("first"))

        # Second turn — capture the request body sent to the API
        captured_bodies = []

        def fake_urlopen(req):
            captured_bodies.append(json.loads(req.data))
            return _make_response(_simple_turn_sse("second"))

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            b._history.append({"role": "user", "content": "second prompt"})
            # drive the request directly without send_turn to avoid extra history append
            task = asyncio.create_task(b._do_request())
            events = []
            gen = b.events()
            try:
                async for ev in gen:
                    events.append(ev)
                    if ev.type in (EventType.TURN_DONE, EventType.ERROR):
                        break
            finally:
                await gen.aclose()
            await task

        self.assertEqual(len(captured_bodies), 1)
        messages = captured_bodies[0]["messages"]
        roles = [m["role"] for m in messages]
        # Must be: user, assistant, user  (not user, user)
        self.assertEqual(roles, [("user"), "assistant", "user"])

    async def test_text_in_assistant_history(self):
        """The stored assistant content includes the text block."""
        b = AnthropicBackend(api_key="test-key")
        await b.connect()
        await _drain_connected(b)

        await _collect_events_until_done(b, _simple_turn_sse("response text"))

        assistant_msg = b._history[-1]
        self.assertEqual(assistant_msg["role"], "assistant")
        content = assistant_msg["content"]
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        self.assertIn("response text", texts)


class TestAnthropicBackendRateLimits(unittest.IsolatedAsyncioTestCase):

    async def test_rate_limit_headers_populate_usage(self):
        """anthropic-ratelimit-* headers are read into NormalizedUsage."""
        b = AnthropicBackend(api_key="test-key")
        await b.connect()
        await _drain_connected(b)

        headers = {
            "anthropic-ratelimit-requests-remaining": "123",
            "anthropic-ratelimit-tokens-remaining":   "45678",
            "anthropic-ratelimit-requests-reset":     "2024-01-01T00:00:00Z",
        }
        events = []
        with patch("urllib.request.urlopen",
                   return_value=_make_response(_simple_turn_sse(), headers=headers)):
            b._history.append({"role": "user", "content": "hi"})
            task = asyncio.create_task(b._do_request())
            gen = b.events()
            try:
                async for ev in gen:
                    events.append(ev)
                    if ev.type in (EventType.TURN_DONE, EventType.ERROR):
                        break
            finally:
                await gen.aclose()
            await task

        td = [e for e in events if e.type == EventType.TURN_DONE][0]
        usage: NormalizedUsage = td.data["usage"]
        self.assertEqual(usage.requests_remaining, 123)
        self.assertEqual(usage.tokens_remaining_per_min, 45678)
        self.assertEqual(usage.reset_at, "2024-01-01T00:00:00Z")


class TestAnthropicBackendHTTPErrors(unittest.IsolatedAsyncioTestCase):

    async def _get_first_non_connected_event(self, backend, exc):
        with patch("urllib.request.urlopen", side_effect=exc):
            await backend.send_turn("hi")
            gen = backend.events()
            try:
                async for ev in gen:
                    return ev
            finally:
                await gen.aclose()

    async def test_http_429_emits_rate_limited(self):
        """HTTP 429 → RATE_LIMITED event (not a crash)."""
        import urllib.error
        b = AnthropicBackend(api_key="test-key")
        await b.connect()
        await _drain_connected(b)

        headers = MagicMock()
        headers.get = lambda k, d=None: "30" if k == "Retry-After" else d
        exc = urllib.error.HTTPError(url="", code=429, msg="Too Many Requests",
                                     hdrs=headers, fp=None)
        ev = await self._get_first_non_connected_event(b, exc)
        self.assertEqual(ev.type, EventType.RATE_LIMITED)

    async def test_http_429_retry_after_propagated(self):
        """Retry-After header value is carried in the RATE_LIMITED event."""
        import urllib.error
        b = AnthropicBackend(api_key="test-key")
        await b.connect()
        await _drain_connected(b)

        headers = MagicMock()
        headers.get = lambda k, d=None: "42" if k == "Retry-After" else d
        exc = urllib.error.HTTPError(url="", code=429, msg="Rate limited",
                                     hdrs=headers, fp=None)
        ev = await self._get_first_non_connected_event(b, exc)
        self.assertEqual(ev.data.get("retry_after_s"), 42.0)

    async def test_http_500_retries_then_fatal_error(self):
        """HTTP 5xx retries up to 3 times then emits fatal ERROR."""
        import urllib.error
        b = AnthropicBackend(api_key="test-key", retry_delay_s=0.01)
        await b.connect()
        await _drain_connected(b)

        exc = urllib.error.HTTPError(url="", code=500, msg="Server Error",
                                     hdrs=MagicMock(), fp=None)
        call_count = 0

        def raise_500(req):
            nonlocal call_count
            call_count += 1
            raise exc

        events = []
        with patch("urllib.request.urlopen", side_effect=raise_500):
            await b.send_turn("hi")
            gen = b.events()
            try:
                async for ev in gen:
                    events.append(ev)
                    if ev.type == EventType.ERROR:
                        break
                    if len(events) > 10:
                        break
            finally:
                await gen.aclose()

        error_events = [e for e in events if e.type == EventType.ERROR]
        self.assertGreater(len(error_events), 0)
        self.assertTrue(error_events[-1].data.get("fatal"))
        self.assertGreater(call_count, 1)


class TestAnthropicBackendTranslateTools(unittest.TestCase):

    def test_translate_produces_input_schema(self):
        from ...backends.base import CanonicalToolSchema
        schemas = [CanonicalToolSchema(
            name="stat", description="Get metadata",
            parameters={"type": "object", "properties": {"resource": {"type": "string"}},
                        "required": ["resource"]})]
        tools = AnthropicBackend.translate_tools(schemas)
        self.assertEqual(tools[0]["name"], "stat")
        self.assertIn("input_schema", tools[0])
        self.assertNotIn("type", tools[0])


class TestAnthropicBackendLogging(unittest.IsolatedAsyncioTestCase):

    async def test_wire_rx_logged_for_every_sse_frame(self):
        """Every parsed SSE data frame must be passed to logger.log_wire_rx,
        including frames that produce no BackendEvent (e.g. content_block_start)."""
        from unittest.mock import MagicMock
        logger = MagicMock()

        b = AnthropicBackend(api_key="test-key", logger=logger)
        await b.connect()
        await _drain_connected(b)

        await _collect_events_until_done(b, _simple_turn_sse("hello"))

        # _simple_turn_sse produces 6 SSE frames; all must be logged
        calls = logger.log_wire_rx.call_args_list
        self.assertGreaterEqual(len(calls), 6,
            f"expected >= 6 wire_rx calls, got {len(calls)}")
        backends_used = {c.args[0] for c in calls}
        self.assertEqual(backends_used, {"anthropic"})
        frame_types = [c.args[1]["type"] for c in calls]
        self.assertIn("message_start", frame_types)
        self.assertIn("message_stop", frame_types)
        self.assertIn("content_block_delta", frame_types)

    async def test_wire_rx_disposition_correct(self):
        """Processed frames carry disposition='processed'; informational-only
        frames (content_block_start/stop, message_delta) that also update state
        are processed; unrecognised frames carry 'ignored'."""
        from unittest.mock import MagicMock
        logger = MagicMock()

        b = AnthropicBackend(api_key="test-key", logger=logger)
        await b.connect()
        await _drain_connected(b)

        await _collect_events_until_done(b, _simple_turn_sse("hi"))

        by_type = {c.args[1]["type"]: c.args[2]
                   for c in logger.log_wire_rx.call_args_list}
        for expected_processed in ("message_start", "content_block_start",
                                   "content_block_delta", "content_block_stop",
                                   "message_delta", "message_stop"):
            if expected_processed in by_type:
                self.assertEqual(by_type[expected_processed], "processed",
                    f"{expected_processed} should be processed, got {by_type[expected_processed]!r}")

    async def test_wire_rx_called_with_full_frame(self):
        """The full parsed frame dict is passed, not just the type."""
        from unittest.mock import MagicMock
        logger = MagicMock()

        b = AnthropicBackend(api_key="test-key", logger=logger)
        await b.connect()
        await _drain_connected(b)

        await _collect_events_until_done(b, _simple_turn_sse("hi"))

        # Find the content_block_delta call and check it has nested data
        delta_calls = [c for c in logger.log_wire_rx.call_args_list
                       if c.args[1].get("type") == "content_block_delta"]
        self.assertTrue(delta_calls, "no content_block_delta log call found")
        frame = delta_calls[0].args[1]
        self.assertIn("delta", frame)
        self.assertEqual(frame["delta"]["type"], "text_delta")

    async def test_no_logger_does_not_crash(self):
        """Backends with no logger attached must work identically."""
        b = AnthropicBackend(api_key="test-key")  # no logger=
        await b.connect()
        await _drain_connected(b)
        events = await _collect_events_until_done(b, _simple_turn_sse("ok"))
        self.assertTrue(any(e.type == EventType.TURN_DONE for e in events))


class TestAnthropicBackendWireRxError(unittest.IsolatedAsyncioTestCase):

    async def test_unparseable_sse_frame_logged_as_wire_rx_error(self):
        """A data: line that is not valid JSON must be logged via
        log_wire_rx_error, not silently dropped."""
        from unittest.mock import MagicMock, patch
        logger = MagicMock()
        b = AnthropicBackend(api_key="test-key", logger=logger)
        await b.connect()
        await _drain_connected(b)

        # Build an SSE stream that has one bad line then a valid message_stop
        bad_sse = (
            b"data: not valid json{{\n\n"
            b'data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}\n\n'
            b'data: {"type":"message_stop"}\n\n'
        )
        await _collect_events_until_done(b, bad_sse)

        # log_wire_rx_error must have been called for the bad line
        self.assertTrue(logger.log_wire_rx_error.called,
            "unparseable SSE frame must be logged via log_wire_rx_error")
        args = logger.log_wire_rx_error.call_args[0]
        self.assertEqual(args[0], "anthropic")
        self.assertIn("not valid json", args[1])
        self.assertTrue(args[2])

    async def test_valid_frames_still_logged_after_bad_one(self):
        """A parse error must not stop subsequent valid frames from being logged."""
        from unittest.mock import MagicMock
        logger = MagicMock()
        b = AnthropicBackend(api_key="test-key", logger=logger)
        await b.connect()
        await _drain_connected(b)

        bad_then_good = (
            b"data: {bad\n\n"
            b'data: {"type":"message_start","message":{"usage":{"input_tokens":2}}}\n\n'
            b'data: {"type":"message_stop"}\n\n'
        )
        await _collect_events_until_done(b, bad_then_good)

        good_calls = [c for c in logger.log_wire_rx.call_args_list
                      if c.args[1].get("type") == "message_start"]
        self.assertTrue(good_calls,
            "valid frames after a bad one must still be logged via log_wire_rx")


