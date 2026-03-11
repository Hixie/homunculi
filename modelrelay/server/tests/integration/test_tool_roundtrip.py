import asyncio
import unittest
from ..conftest import make_session
from ...backends.base import ScriptedTurn, ScriptedToolCall


class TestToolRoundtrip(unittest.IsolatedAsyncioTestCase):
    async def test_content_request_forwarded(self):
        tc = ScriptedToolCall(name="read", args={"resource":"f.py","ranges":[{"start_line":1,"num_lines":"all"}]})
        session, task = await make_session([ScriptedTurn(tool_calls=[tc])])
        await session.recv_until("status.activity")  # connected
        await session.send_host("cmd.prompt", {"text": "go"})
        # The orchestrator will call the read tool, which sends content_request
        env = await session.recv_until("tool.content_request", timeout=3.0)
        self.assertEqual(env["type"], "tool.content_request")
        self.assertIn("ranges", env["payload"])
        # Now respond so the tool completes
        await session.send_host("tool.content_response", {
            "resource":"f.py","total_lines":5,
            "regions":[{"start_line":1,"end_line":5,"lines":["a\n"]*5}],
            "truncated":False,"error":None
        }, id_=env["id"])
        await session.recv_until("status.activity", timeout=3.0)
        task.cancel()
        try: await task
        except: pass

    async def test_content_response_resolves_tool(self):
        tc = ScriptedToolCall(name="stat", args={"resource":"f.py"})
        session, task = await make_session([ScriptedTurn(tool_calls=[tc])])
        await session.recv_until("status.activity")
        await session.send_host("cmd.prompt", {"text": "go"})
        env = await session.recv_until("tool.stat_request", timeout=3.0)
        await session.send_host("tool.stat_response", {
            "resource":"f.py","exists":True,"total_lines":10,"last_modified":"now","error":None
        }, id_=env["id"])
        done = await session.recv_until("status.activity", timeout=3.0)
        task.cancel()
        try: await task
        except: pass

    async def test_replace_request_forwarded(self):
        tc = ScriptedToolCall(name="replace", args={
            "resource":"f.py","start_line":1,"end_line":2,
            "new_content":"new\n","total_lines":5})
        session, task = await make_session([ScriptedTurn(tool_calls=[tc])])
        await session.recv_until("status.activity")
        await session.send_host("cmd.prompt", {"text": "go"})
        env = await session.recv_until("tool.replace_request", timeout=3.0)
        self.assertEqual(env["type"], "tool.replace_request")
        p = env["payload"]
        self.assertIn("new_lines", p)
        self.assertIn("total_lines", p)
        task.cancel()
        try: await task
        except: pass

    async def test_replace_awaits_response(self):
        """replace must block until tool.replace_response is received."""
        tc = ScriptedToolCall(name="replace", args={
            "resource":"f.py","start_line":1,"end_line":2,
            "new_content":"x\n","total_lines":5})
        session, task = await make_session([ScriptedTurn(tool_calls=[tc])])
        await session.recv_until("status.activity")
        await session.send_host("cmd.prompt", {"text": "go"})
        req = await session.recv_until("tool.replace_request", timeout=3.0)
        # Send the host response; the turn should then complete
        await session.send_host("tool.replace_response", {"error": None},
                                id_=req["id"])
        done = await session.recv_until("status.activity", timeout=5.0)
        self.assertIn("returned", done["payload"]["description"])
        task.cancel()
        try: await task
        except: pass
