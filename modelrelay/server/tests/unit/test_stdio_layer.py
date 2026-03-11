"""Tests for StdioLayer — focuses on UNKNOWN_TYPE error emission."""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from ...stdio.protocol import make_envelope, encode


class FakeBus:
    def __init__(self):
        self.published = []
    async def publish(self, topic, envelope):
        self.published.append((topic, envelope))


class TestStdioLayerUnknownType(unittest.IsolatedAsyncioTestCase):

    async def _make_layer_and_feed(self, envelopes):
        """Create a StdioLayer backed by an in-memory reader, feed envelopes, collect sends."""
        from ...stdio.layer import StdioLayer

        pending = {}
        bus = FakeBus()
        layer = StdioLayer(pending, bus)

        # Wire up an in-memory reader
        reader = asyncio.StreamReader()
        layer._reader = reader

        sent = []
        # Patch send() to capture outbound envelopes
        async def fake_send(env):
            sent.append(env)
            await bus.publish("_outbound", env)
        layer.send = fake_send

        # Feed lines
        for env in envelopes:
            reader.feed_data(encode(env))
        reader.feed_eof()

        await layer.run_inbound()
        return sent, bus

    async def test_unknown_type_emits_error(self):
        env = make_envelope("totally.unknown.type", {"x": 1})
        sent, _ = await self._make_layer_and_feed([env])
        error_msgs = [s for s in sent if s.get("type") == "error"]
        self.assertEqual(len(error_msgs), 1)
        self.assertEqual(error_msgs[0]["payload"]["code"], "UNKNOWN_TYPE")
        self.assertFalse(error_msgs[0]["payload"]["fatal"])

    async def test_known_types_do_not_emit_error(self):
        known = [
            make_envelope("tool.content_response",
                {"resource":"f","total_lines":1,"regions":[],"truncated":False,"error":None}),
            make_envelope("tool.stat_response",
                {"resource":"f","exists":True,"total_lines":1,"last_modified":None,"error":None}),
            make_envelope("cmd.prompt",    {"text": "hi"}),
            make_envelope("cmd.invalidate",{"resource": "f"}),
            make_envelope("cmd.quit",      {}),
        ]
        sent, _ = await self._make_layer_and_feed(known)
        error_msgs = [s for s in sent if s.get("type") == "error"]
        self.assertEqual(error_msgs, [])

    async def test_parse_error_still_emitted(self):
        """Malformed JSON still produces PARSE_ERROR (existing behaviour unchanged)."""
        from ...stdio.layer import StdioLayer
        pending = {}
        bus = FakeBus()
        layer = StdioLayer(pending, bus)
        reader = asyncio.StreamReader()
        layer._reader = reader
        sent = []
        async def fake_send(env):
            sent.append(env)
        layer.send = fake_send
        reader.feed_data(b"not valid json\n")
        reader.feed_eof()
        await layer.run_inbound()
        self.assertTrue(any(s.get("payload", {}).get("code") == "PARSE_ERROR" for s in sent))

    async def test_insert_and_replace_responses_are_known(self):
        """tool.insert_response and tool.replace_response must not produce
        UNKNOWN_TYPE errors — they were missing before the fix that caused
        the orchestrator to hang and usage to always read zero."""
        from ...stdio.protocol import make_envelope
        known = [
            make_envelope("tool.insert_response",  {"error": None}),
            make_envelope("tool.replace_response", {"error": None}),
        ]
        sent, _ = await self._make_layer_and_feed(known)
        error_msgs = [s for s in sent if s.get("type") == "error"]
        self.assertEqual(error_msgs, [],
            "insert/replace responses must be accepted, not rejected as UNKNOWN_TYPE")

    async def test_insert_response_resolves_pending_future(self):
        """tool.insert_response must resolve the pending Future so ctx.request()
        unblocks — otherwise the orchestrator hangs and usage shows zero."""
        from ...stdio.layer import StdioLayer
        from ...stdio.protocol import make_envelope
        pending: dict = {}
        bus = FakeBus()
        layer = StdioLayer(pending, bus)
        reader = asyncio.StreamReader()
        layer._reader = reader
        sent = []
        async def fake_send(env): sent.append(env)
        layer.send = fake_send

        env = make_envelope("tool.insert_response", {"error": None})
        id_ = env["id"]
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        pending[id_] = fut

        reader.feed_data((json.dumps(env) + "\n").encode())
        reader.feed_eof()
        await layer.run_inbound()

        self.assertTrue(fut.done(), "pending future must be resolved by insert_response")
        self.assertEqual(fut.result()["id"], id_)

    async def test_replace_response_resolves_pending_future(self):
        """tool.replace_response must resolve the pending Future."""
        from ...stdio.layer import StdioLayer
        from ...stdio.protocol import make_envelope
        pending: dict = {}
        bus = FakeBus()
        layer = StdioLayer(pending, bus)
        reader = asyncio.StreamReader()
        layer._reader = reader
        sent = []
        async def fake_send(env): sent.append(env)
        layer.send = fake_send

        env = make_envelope("tool.replace_response", {"error": None})
        id_ = env["id"]
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        pending[id_] = fut

        reader.feed_data((json.dumps(env) + "\n").encode())
        reader.feed_eof()
        await layer.run_inbound()

        self.assertTrue(fut.done(), "pending future must be resolved by replace_response")
        self.assertEqual(fut.result()["id"], id_)

    async def test_parse_error_logged(self):
        """Malformed JSON on stdin must be logged as protocol/inbound_error/PARSE_ERROR."""
        from ...stdio.layer import StdioLayer
        from unittest.mock import MagicMock
        logger = MagicMock()
        pending = {}
        bus = FakeBus()
        layer = StdioLayer(pending, bus, logger=logger)
        reader = asyncio.StreamReader()
        layer._reader = reader
        sent = []
        async def fake_send(env): sent.append(env)
        layer.send = fake_send

        reader.feed_data(b"not valid json{{\n")
        reader.feed_eof()
        await layer.run_inbound()

        logger.log_protocol_inbound_error.assert_called_once()
        args = logger.log_protocol_inbound_error.call_args[0]
        # log_protocol_inbound_error(raw, reason, detail)
        self.assertIn("not valid json", args[0])   # raw
        self.assertEqual(args[1], "PARSE_ERROR")   # reason
        # disposition="erroneous" is hardcoded inside the method — verified in test_logger

    async def test_unknown_type_logged(self):
        """An unknown message type must be logged as protocol/inbound_error/UNKNOWN_TYPE."""
        from ...stdio.layer import StdioLayer
        from ...stdio.protocol import make_envelope
        from unittest.mock import MagicMock
        logger = MagicMock()
        pending = {}
        bus = FakeBus()
        layer = StdioLayer(pending, bus, logger=logger)
        reader = asyncio.StreamReader()
        layer._reader = reader
        sent = []
        async def fake_send(env): sent.append(env)
        layer.send = fake_send

        env = make_envelope("totally.unknown", {"x": 1})
        reader.feed_data((json.dumps(env) + "\n").encode())
        reader.feed_eof()
        await layer.run_inbound()

        logger.log_protocol_inbound_error.assert_called_once()
        args = logger.log_protocol_inbound_error.call_args[0]
        # log_protocol_inbound_error(raw, reason, detail)
        self.assertEqual(args[1], "UNKNOWN_TYPE")      # reason
        self.assertIn("totally.unknown", args[2])      # detail
        # disposition="erroneous" is hardcoded inside the method — verified in test_logger

    async def test_no_logger_still_works(self):
        """StdioLayer must work identically when constructed without a logger."""
        from ...stdio.layer import StdioLayer
        from ...stdio.protocol import make_envelope
        pending = {}
        bus = FakeBus()
        layer = StdioLayer(pending, bus)   # no logger=
        reader = asyncio.StreamReader()
        layer._reader = reader
        sent = []
        async def fake_send(env): sent.append(env)
        layer.send = fake_send

        reader.feed_data(b"not valid json\n")
        reader.feed_eof()
        await layer.run_inbound()
        self.assertTrue(any(s.get("payload", {}).get("code") == "PARSE_ERROR"
                            for s in sent))

