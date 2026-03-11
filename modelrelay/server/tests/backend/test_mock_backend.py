import asyncio
import unittest
from ...backends.mock import MockBackend
from ...backends.base import (EventType, NormalizedUsage,
                                      ScriptedTurn, ScriptedToolCall,
                                      BackendScriptExhausted)


class TestMockBackend(unittest.IsolatedAsyncioTestCase):
    async def test_connected_event(self):
        b = MockBackend()
        await b.connect()
        events = []
        async for e in b.events():
            events.append(e)
            break
        self.assertEqual(events[0].type, EventType.CONNECTED)

    async def test_text_deltas_carry_text(self):
        turn = ScriptedTurn(text_chunks=["hello", " world"])
        b = MockBackend([turn])
        await b.connect()
        await b.send_turn("hi")
        await asyncio.sleep(0.1)
        events = []
        async for e in b.events():
            events.append(e)
            if e.type == EventType.TURN_DONE:
                break
        text_events = [e for e in events if e.type == EventType.TEXT_DELTA]
        self.assertEqual(len(text_events), 2)
        self.assertEqual(text_events[0].data["text"], "hello")
        self.assertEqual(text_events[1].data["text"], " world")

    async def test_turn_done(self):
        b = MockBackend([ScriptedTurn(text_chunks=["hi"])])
        await b.connect()
        await b.send_turn("go")
        await asyncio.sleep(0.1)
        events = []
        async for e in b.events():
            events.append(e)
            if e.type == EventType.TURN_DONE:
                break
        self.assertEqual(events[-1].type, EventType.TURN_DONE)

    async def test_tool_call_roundtrip(self):
        tc = ScriptedToolCall(name="stat", args={"resource": "f.py"})
        turn = ScriptedTurn(tool_calls=[tc])
        b = MockBackend([turn])
        await b.connect()
        await b.send_turn("go")
        await asyncio.sleep(0.05)
        events = []
        async for e in b.events():
            events.append(e)
            if e.type == EventType.TOOL_CALL:
                await b.submit_tool_result(e.data["call_id"], "ok")
            if e.type == EventType.TURN_DONE:
                break
        tool_events = [e for e in events if e.type == EventType.TOOL_CALL]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0].data["name"], "stat")

    async def test_multiple_tool_calls(self):
        """Two tool calls per turn are both yielded and both awaited."""
        tc1 = ScriptedToolCall(name="stat",   args={"resource": "a.py"})
        tc2 = ScriptedToolCall(name="search", args={"resource": "b.py", "pattern": "x"})
        turn = ScriptedTurn(tool_calls=[tc1, tc2])
        b = MockBackend([turn])
        await b.connect()
        await b.send_turn("go")
        await asyncio.sleep(0.05)
        tool_names = []
        async for e in b.events():
            if e.type == EventType.TOOL_CALL:
                tool_names.append(e.data["name"])
                await b.submit_tool_result(e.data["call_id"], "result")
            if e.type == EventType.TURN_DONE:
                break
        self.assertEqual(tool_names, ["stat", "search"])

    async def test_rate_limited_before_turn(self):
        turn = ScriptedTurn(rate_limited=5.0, text_chunks=["ok"])
        b = MockBackend([turn])
        await b.connect()
        await b.send_turn("go")
        await asyncio.sleep(0.1)
        events = []
        async for e in b.events():
            events.append(e)
            if e.type == EventType.TURN_DONE:
                break
        types = [e.type for e in events]
        rl_idx = types.index(EventType.RATE_LIMITED)
        td_idx = types.index(EventType.TURN_DONE)
        self.assertLess(rl_idx, td_idx)

    async def test_exhausted_emits_fatal_error(self):
        """Script exhausted → fatal ERROR event (not a raised exception)."""
        b = MockBackend([])   # empty script
        await b.connect()
        async for e in b.events():
            break  # consume CONNECTED
        await b.send_turn("prompt that exhausts the script")
        await asyncio.sleep(0.1)
        events = []
        async for e in b.events():
            events.append(e)
            break
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, EventType.ERROR)
        self.assertTrue(events[0].data.get("fatal"))

    async def test_inject_note_recorded(self):
        b = MockBackend()
        await b.inject_note("hello note")
        self.assertEqual(b.received_notes, ["hello note"])

    async def test_usage_in_turn_done(self):
        usage = NormalizedUsage(input_tokens=10, output_tokens=5)
        turn = ScriptedTurn(usage=usage)
        b = MockBackend([turn])
        await b.connect()
        await b.send_turn("go")
        await asyncio.sleep(0.1)
        events = []
        async for e in b.events():
            events.append(e)
            if e.type == EventType.TURN_DONE:
                break
        td = [e for e in events if e.type == EventType.TURN_DONE][0]
        self.assertIsInstance(td.data["usage"], NormalizedUsage)
