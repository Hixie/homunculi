"""Integration tests for the replace tool.

Key invariant: the tool must block until a tool.replace_response arrives from
the host before returning a result to the model.
"""
import asyncio
import unittest
from ..conftest import make_session
from ...backends.base import ScriptedTurn, ScriptedToolCall


def _replace_tc(**kwargs) -> ScriptedToolCall:
    defaults = {"resource": "f.py", "start_line": 3, "end_line": 5,
                "new_content": "new line\n", "total_lines": 10}
    return ScriptedToolCall(name="replace", args={**defaults, **kwargs})


class TestReplaceFlow(unittest.IsolatedAsyncioTestCase):

    async def _setup(self, tc):
        session, task = await make_session([ScriptedTurn(tool_calls=[tc])])
        await session.recv_until("status.activity")   # CONNECTED idle
        await session.send_host("cmd.prompt", {"text": "go"})
        req = await session.recv_until("tool.replace_request", timeout=3.0)
        return session, task, req

    # ── request forwarded correctly ───────────────────────────────────────────

    async def test_replace_request_forwarded(self):
        session, task, req = await self._setup(_replace_tc())
        p = req["payload"]
        self.assertEqual(p["start_line"],  3)
        self.assertEqual(p["end_line"],    5)
        self.assertEqual(p["total_lines"], 10)
        self.assertIn("new_lines", p)
        task.cancel()
        try: await task
        except: pass

    # ── tool blocks until response arrives ────────────────────────────────────

    async def test_tool_blocks_until_replace_response(self):
        """The tool must NOT return a result to the model before the host
        sends tool.replace_response."""
        session, task, req = await self._setup(_replace_tc())

        # Drain anything currently queued
        queued = []
        while True:
            try:
                env = await asyncio.wait_for(session.recv(timeout=0.05),
                                             timeout=0.1)
                queued.append(env)
            except (asyncio.TimeoutError, Exception):
                break

        continuing = [m for m in queued
                      if m.get("type") == "status.activity"
                      and "returned" in m.get("payload", {}).get("description", "")]
        self.assertEqual(len(continuing), 0,
            "tool must not continue before replace_response is received")

        # Now send the response
        await session.send_host("tool.replace_response",
                                {"error": None},
                                id_=req["id"])
        cont = await session.recv_until("status.activity", timeout=3.0)
        self.assertIn("returned", cont["payload"]["description"])

        task.cancel()
        try: await task
        except: pass

    async def test_host_error_propagated_to_model(self):
        """An error payload in the replace_response must be surfaced."""
        session, task, req = await self._setup(_replace_tc())
        await session.send_host("tool.replace_response",
                                {"error": "line range mismatch"},
                                id_=req["id"])
        await session.recv_until("status.activity", timeout=3.0)
        task.cancel()
        try: await task
        except: pass

    async def test_success_response_completes_turn(self):
        session, task, req = await self._setup(_replace_tc())
        await session.send_host("tool.replace_response", {"error": None},
                                id_=req["id"])
        await session.recv_until("status.activity", timeout=3.0)
        task.cancel()
        try: await task
        except: pass
