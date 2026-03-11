"""
Tests for OpenAIRealtimeBackend frame parsing using a fake WebSocket.

We don't open a real network connection. Instead we push frames directly
into the backend's _recv_loop via a fake websocket object.
"""
from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from ...backends.openai_realtime import OpenAIRealtimeBackend
from ...backends.base import EventType


class FakeWS:
    """Minimal fake WebSocket that yields pre-loaded frames then blocks."""
    def __init__(self, frames: list[dict]):
        self._queue: asyncio.Queue = asyncio.Queue()
        for f in frames:
            self._queue.put_nowait(json.dumps(f))
        # sentinel: block forever after frames exhausted
        self._done = asyncio.Event()

    async def recv(self) -> str:
        if not self._queue.empty():
            return self._queue.get_nowait()
        await self._done.wait()          # blocks until test cancels
        raise asyncio.CancelledError()

    async def send(self, data: str): pass
    async def close(self): self._done.set()


async def run_backend_with_frames(frames: list[dict], n_events: int,
                                   timeout: float = 2.0) -> list:
    """Spin up a backend, inject frames through a fake WS, collect up to n_events.

    Stops as soon as n_events have been collected OR the fake WS is exhausted
    (whichever comes first), so tests that expect fewer events than the queue
    contains don't hang.
    """
    backend = OpenAIRealtimeBackend(api_key="test", model="gpt-4o-realtime")
    fake_ws = FakeWS(frames)
    backend._ws = fake_ws
    backend._recv_task = asyncio.create_task(backend._recv_loop())

    events = []
    try:
        async with asyncio.timeout(timeout):
            async for evt in backend.events():
                events.append(evt)
                if len(events) >= n_events:
                    break
    except (asyncio.TimeoutError, TimeoutError):
        pass
    finally:
        await backend.close()

    return events


class TestOpenAIRealtimeBackend(unittest.IsolatedAsyncioTestCase):

    async def test_translate_tools_type_field(self):
        from ...tools import build_registry
        schemas = build_registry().get_schemas()
        tools = OpenAIRealtimeBackend.translate_tools(schemas)
        for t in tools:
            self.assertEqual(t["type"], "function")
            self.assertIn("name", t)
            self.assertIn("description", t)
            self.assertIn("parameters", t)

    async def test_text_delta_event(self):
        frames = [{"type": "response.output_text.delta", "delta": "Hello, world"}]
        events = await run_backend_with_frames(frames, n_events=1)
        self.assertEqual(events[0].type, EventType.TEXT_DELTA)
        self.assertEqual(events[0].data["text"], "Hello, world")

    async def test_text_delta_carries_text_key(self):
        """data["text"] must be present — required by Orchestrator for forwarding."""
        frames = [{"type": "response.output_text.delta", "delta": "hi"}]
        events = await run_backend_with_frames(frames, n_events=1)
        self.assertIn("text", events[0].data)

    async def test_function_call_arguments_accumulate_then_done(self):
        frames = [
            {"type": "response.function_call_arguments.delta",
             "call_id": "call_1", "name": "stat", "delta": "{\"resour"},
            {"type": "response.function_call_arguments.delta",
             "call_id": "call_1", "delta": "ce\":\"f.py\"}"},
            {"type": "response.function_call_arguments.done",
             "call_id": "call_1"},
        ]
        events = await run_backend_with_frames(frames, n_events=1)
        self.assertEqual(events[0].type, EventType.TOOL_CALL)
        self.assertEqual(events[0].data["name"], "stat")
        self.assertEqual(events[0].data["args"]["resource"], "f.py")
        self.assertEqual(events[0].data["call_id"], "call_1")

    async def test_tool_name_comes_from_output_item_added(self):
        """The function name is delivered on response.output_item.added, not on the
        argument delta frames.  The backend must seed the call from that event so
        the TOOL_CALL carries the correct name even when delta frames omit it."""
        frames = [
            # Real Realtime API sequence: name arrives here, before any delta
            {"type": "response.output_item.added",
             "item": {"type": "function_call", "call_id": "call_1", "name": "read"}},
            # Subsequent delta frames do NOT repeat the name
            {"type": "response.function_call_arguments.delta",
             "call_id": "call_1", "delta": "{\"resource\":\"text_clock.py\"}"},
            {"type": "response.function_call_arguments.done",
             "call_id": "call_1"},
        ]
        events = await run_backend_with_frames(frames, n_events=1)
        self.assertEqual(events[0].type, EventType.TOOL_CALL)
        self.assertEqual(events[0].data["name"], "read",
            "tool name must be taken from response.output_item.added, not the delta")
        self.assertEqual(events[0].data["args"]["resource"], "text_clock.py")
        self.assertEqual(events[0].data["call_id"], "call_1")

    async def test_malformed_tool_args_dont_crash(self):
        frames = [
            {"type": "response.function_call_arguments.delta",
             "call_id": "c1", "name": "read", "delta": "not-valid-json"},
            {"type": "response.function_call_arguments.done", "call_id": "c1"},
        ]
        events = await run_backend_with_frames(frames, n_events=1)
        self.assertEqual(events[0].type, EventType.TOOL_CALL)
        self.assertIn("_error", events[0].data["args"])

    async def test_response_done_yields_turn_done(self):
        frames = [{
            "type": "response.done",
            "response": {"usage": {"input_tokens": 42, "output_tokens": 17}},
        }]
        events = await run_backend_with_frames(frames, n_events=1)
        self.assertEqual(events[0].type, EventType.TURN_DONE)
        usage = events[0].data["usage"]
        self.assertEqual(usage.input_tokens, 42)
        self.assertEqual(usage.output_tokens, 17)

    async def test_rate_limits_updated(self):
        frames = [{
            "type": "rate_limits.updated",
            "rate_limits": [{"name": "requests", "remaining": 0, "reset_seconds": 12.5}],
        }]
        events = await run_backend_with_frames(frames, n_events=1)
        self.assertEqual(events[0].type, EventType.RATE_LIMITED)
        self.assertEqual(events[0].data["retry_after_s"], 12.5)

    async def test_rate_limits_non_zero_remaining_not_emitted(self):
        """rate_limits.updated with remaining > 0 is informational; must NOT emit
        RATE_LIMITED.  The OpenAI Realtime API sends rate_limits.updated *after*
        response.done; emitting RATE_LIMITED for it would overwrite the 'idle'
        state that TURN_DONE just established."""
        frames = [
            {"type": "response.output_text.delta", "delta": "hello"},
            {
                "type": "response.done",
                "response": {"usage": {"input_tokens": 10, "output_tokens": 5}},
            },
            # Sent by the server immediately after response.done
            {
                "type": "rate_limits.updated",
                "rate_limits": [{"name": "requests", "remaining": 5, "reset_seconds": None}],
            },
        ]
        # Collect one more event than expected (TEXT_DELTA + TURN_DONE = 2) so
        # that a spurious RATE_LIMITED would be caught rather than left unseen.
        events = await run_backend_with_frames(frames, n_events=3, timeout=0.3)
        types = [e.type for e in events]
        self.assertIn(EventType.TEXT_DELTA, types)
        self.assertIn(EventType.TURN_DONE, types)
        self.assertNotIn(EventType.RATE_LIMITED, types,
            "rate_limits.updated with remaining > 0 must not emit RATE_LIMITED "
            "-- it overwrites the idle state after a completed turn")

    async def test_error_frame_non_fatal(self):
        frames = [{"type": "error", "error": {"code": "some_error", "message": "oops"}}]
        events = await run_backend_with_frames(frames, n_events=1)
        self.assertEqual(events[0].type, EventType.ERROR)
        self.assertFalse(events[0].data["fatal"])

    async def test_error_frame_fatal_on_invalid_api_key(self):
        frames = [{"type": "error", "error": {"code": "invalid_api_key", "message": "bad key"}}]
        events = await run_backend_with_frames(frames, n_events=1)
        self.assertEqual(events[0].type, EventType.ERROR)
        self.assertTrue(events[0].data["fatal"])

    async def test_unknown_frame_types_ignored(self):
        """session.created, response.created etc. produce no BackendEvents."""
        frames = [
            {"type": "session.created"},
            {"type": "session.updated"},
            {"type": "conversation.item.created"},
            {"type": "response.created"},
            {"type": "response.output_text.delta", "delta": "real event"},
        ]
        events = await run_backend_with_frames(frames, n_events=1)
        # Only one event: the TEXT_DELTA
        self.assertEqual(events[0].type, EventType.TEXT_DELTA)

    async def test_multiple_tool_calls_independent(self):
        """Two concurrent calls accumulate independently."""
        frames = [
            {"type": "response.function_call_arguments.delta",
             "call_id": "c1", "name": "stat", "delta": "{\"resource\":\"a\"}"},
            {"type": "response.function_call_arguments.delta",
             "call_id": "c2", "name": "stat", "delta": "{\"resource\":\"b\"}"},
            {"type": "response.function_call_arguments.done", "call_id": "c1"},
            {"type": "response.function_call_arguments.done", "call_id": "c2"},
        ]
        events = await run_backend_with_frames(frames, n_events=2)
        tool_events = [e for e in events if e.type == EventType.TOOL_CALL]
        self.assertEqual(len(tool_events), 2)
        names_and_resources = {e.data["call_id"]: e.data["args"]["resource"]
                               for e in tool_events}
        self.assertEqual(names_and_resources["c1"], "a")
        self.assertEqual(names_and_resources["c2"], "b")

    async def test_inject_note_sends_system_message(self):
        """inject_note sends a conversation.item.create of type message/system."""
        sent_frames = []

        backend = OpenAIRealtimeBackend(api_key="test")

        class CapturingWS:
            async def send(self_, data):
                sent_frames.append(json.loads(data))
            async def recv(self_):
                await asyncio.sleep(10)
            async def close(self_): pass

        backend._ws = CapturingWS()
        await backend.inject_note("do not retry this change")

        self.assertEqual(len(sent_frames), 1)
        frame = sent_frames[0]
        self.assertEqual(frame["type"], "conversation.item.create")
        self.assertEqual(frame["item"]["role"], "system")
        text = frame["item"]["content"][0]["text"]
        self.assertIn("do not retry", text)

    async def test_send_turn_sends_two_frames(self):
        """send_turn sends conversation.item.create then response.create."""
        sent_frames = []

        backend = OpenAIRealtimeBackend(api_key="test")

        class CapturingWS:
            async def send(self_, data):
                sent_frames.append(json.loads(data))
            async def recv(self_):
                await asyncio.sleep(10)
            async def close(self_): pass

        backend._ws = CapturingWS()
        await backend.send_turn("Fix the bug")

        self.assertEqual(len(sent_frames), 2)
        self.assertEqual(sent_frames[0]["type"], "conversation.item.create")
        self.assertEqual(sent_frames[0]["item"]["content"][0]["text"], "Fix the bug")
        self.assertEqual(sent_frames[1]["type"], "response.create")

    async def test_submit_tool_result_sends_two_frames(self):
        """submit_tool_result sends function_call_output then response.create."""
        sent_frames = []

        backend = OpenAIRealtimeBackend(api_key="test")

        class CapturingWS:
            async def send(self_, data):
                sent_frames.append(json.loads(data))
            async def recv(self_):
                await asyncio.sleep(10)
            async def close(self_): pass

        backend._ws = CapturingWS()
        await backend.submit_tool_result("call_1", "f.py: 100 lines")

        self.assertEqual(len(sent_frames), 2)
        item = sent_frames[0]["item"]
        self.assertEqual(item["type"], "function_call_output")
        self.assertEqual(item["call_id"], "call_1")
        self.assertEqual(item["output"], "f.py: 100 lines")
        self.assertEqual(sent_frames[1]["type"], "response.create")

    async def test_connect_uses_authorization_header(self):
        """connect() must call websockets.connect with an Authorization header
        and must NOT include the removed OpenAI-Beta beta header."""
        from unittest.mock import patch, AsyncMock, MagicMock
        import server.backends.openai_realtime as oai_module

        call_kwargs: list[dict] = []
        sent_frames: list[dict] = []

        async def fake_connect(url, **kwargs):
            call_kwargs.append(dict(kwargs))
            fake_ws = AsyncMock()
            async def capture_send(data):
                sent_frames.append(json.loads(data))
            fake_ws.send = capture_send
            fake_ws.recv = AsyncMock(side_effect=asyncio.CancelledError)
            fake_ws.close = AsyncMock()
            return fake_ws

        mock_ws_module = MagicMock()
        mock_ws_module.connect = fake_connect

        backend = OpenAIRealtimeBackend(
            api_key="sk-test", model="gpt-4o-realtime-preview",
            max_reconnect_attempts=3)

        with patch.object(oai_module, "websockets", mock_ws_module):
            await backend.connect()

        self.assertEqual(len(call_kwargs), 1)
        headers = call_kwargs[0]["additional_headers"]
        self.assertIn("Authorization", headers)
        self.assertIn("sk-test", headers["Authorization"])
        self.assertNotIn("OpenAI-Beta", headers,
            "OpenAI-Beta header must be absent in the GA interface")

        # session.update must include type="realtime"
        session_updates = [f for f in sent_frames if f.get("type") == "session.update"]
        self.assertTrue(session_updates, "no session.update sent during connect()")
        self.assertEqual(session_updates[0]["session"].get("type"), "realtime",
            "session.update session dict must include type='realtime' for GA")

        await backend.close()

    async def test_wire_rx_disposition_processed_for_handled_frames(self):
        """Frames that trigger a BackendEvent must be logged with disposition='processed'."""
        from unittest.mock import MagicMock
        from ...backends.openai_realtime import OpenAIRealtimeBackend

        logger = MagicMock()
        backend = OpenAIRealtimeBackend(api_key="test", model="gpt-4o-realtime",
                                         logger=logger)
        handled_frames = [
            {"type": "response.output_text.delta", "delta": "hi"},
            {"type": "response.output_item.added",
             "item": {"type": "function_call", "call_id": "c1", "name": "read"}},
            {"type": "response.function_call_arguments.delta",
             "call_id": "c1", "delta": "{}"},
            {"type": "response.function_call_arguments.done",
             "call_id": "c1", "name": "read", "arguments": "{}"},
            {"type": "response.done",
             "response": {"usage": {"input_tokens": 5, "output_tokens": 2}}},
        ]
        fake_ws = FakeWS(handled_frames)
        backend._ws = fake_ws
        backend._recv_task = asyncio.create_task(backend._recv_loop())

        await asyncio.sleep(0.15)
        await backend.close()

        by_type = {c.args[1]["type"]: c.args[2]
                   for c in logger.log_wire_rx.call_args_list}
        for etype in ("response.output_text.delta", "response.output_item.added",
                      "response.function_call_arguments.delta",
                      "response.function_call_arguments.done",
                      "response.done"):
            self.assertEqual(by_type.get(etype), "processed",
                f"{etype} should have disposition 'processed', got {by_type.get(etype)!r}")

    async def test_wire_rx_disposition_ignored_for_informational_frames(self):
        """Frames that produce no BackendEvent must be logged with disposition='ignored'."""
        from unittest.mock import MagicMock
        from ...backends.openai_realtime import OpenAIRealtimeBackend

        logger = MagicMock()
        backend = OpenAIRealtimeBackend(api_key="test", model="gpt-4o-realtime",
                                         logger=logger)
        informational_frames = [
            {"type": "session.created"},
            {"type": "session.updated"},
            {"type": "conversation.item.created"},
            {"type": "response.created"},
            {"type": "response.output_item.done"},
            {"type": "response.content_part.added"},
            {"type": "totally.unknown.frame"},
        ]
        fake_ws = FakeWS(informational_frames)
        backend._ws = fake_ws
        backend._recv_task = asyncio.create_task(backend._recv_loop())

        await asyncio.sleep(0.15)
        await backend.close()

        by_type = {c.args[1]["type"]: c.args[2]
                   for c in logger.log_wire_rx.call_args_list}
        for etype in ("session.created", "session.updated",
                      "conversation.item.created", "response.created",
                      "totally.unknown.frame"):
            self.assertEqual(by_type.get(etype), "ignored",
                f"{etype} should have disposition 'ignored', got {by_type.get(etype)!r}")

    async def test_unparseable_frame_logged_as_wire_rx_error(self):
        """A WebSocket message that is not valid JSON must be logged via
        log_wire_rx_error, not silently discarded."""
        from unittest.mock import MagicMock
        from ...backends.openai_realtime import OpenAIRealtimeBackend

        logger = MagicMock()
        backend = OpenAIRealtimeBackend(api_key="test", model="gpt-4o-realtime",
                                         logger=logger)
        # FakeWS normally JSON-encodes its dicts; we bypass that by
        # putting the raw invalid string directly into the queue.
        fake_ws = FakeWS([])
        fake_ws._queue.put_nowait("not valid json{{{")
        backend._ws = fake_ws
        backend._recv_task = asyncio.create_task(backend._recv_loop())

        await asyncio.sleep(0.1)
        await backend.close()

        logger.log_wire_rx_error.assert_called_once()
        args = logger.log_wire_rx_error.call_args[0]
        self.assertEqual(args[0], "openai-realtime")
        self.assertIn("not valid json", args[1])   # raw text preserved
        self.assertTrue(args[2])                    # error message non-empty


