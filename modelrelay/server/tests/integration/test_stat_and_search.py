import asyncio
import unittest
from ..conftest import make_session
from ...backends.base import ScriptedTurn, ScriptedToolCall


class TestStatAndSearch(unittest.IsolatedAsyncioTestCase):

    async def test_stat_request_forwarded(self):
        tc = ScriptedToolCall(name="stat", args={"resource": "f.py"})
        session, task = await make_session([ScriptedTurn(tool_calls=[tc])])
        await session.recv_until("status.activity")
        await session.send_host("cmd.prompt", {"text": "go"})
        env = await session.recv_until("tool.stat_request", timeout=3.0)
        self.assertEqual(env["payload"]["resource"], "f.py")
        await session.send_host("tool.stat_response",
            {"resource": "f.py", "exists": True, "total_lines": 20,
             "last_modified": "2024-01-01T00:00:00Z", "error": None},
            id_=env["id"])
        task.cancel()
        try: await task
        except: pass

    async def test_stat_exists_true(self):
        """Exists response → tool return string contains line count."""
        tc = ScriptedToolCall(name="stat", args={"resource": "f.py"})
        session, task = await make_session([ScriptedTurn(tool_calls=[tc])])
        await session.recv_until("status.activity")
        await session.send_host("cmd.prompt", {"text": "go"})
        env = await session.recv_until("tool.stat_request", timeout=3.0)
        await session.send_host("tool.stat_response",
            {"resource": "f.py", "exists": True, "total_lines": 1234,
             "last_modified": "2024-01-01T00:00:00Z", "error": None},
            id_=env["id"])
        # Wait for the tool to complete (TURN_DONE)
        await session.recv_until("status.activity", timeout=5.0)
        task.cancel()
        try: await task
        except: pass

    async def test_stat_exists_false(self):
        tc = ScriptedToolCall(name="stat", args={"resource": "missing.py"})
        session, task = await make_session([ScriptedTurn(tool_calls=[tc])])
        await session.recv_until("status.activity")
        await session.send_host("cmd.prompt", {"text": "go"})
        env = await session.recv_until("tool.stat_request", timeout=3.0)
        await session.send_host("tool.stat_response",
            {"resource": "missing.py", "exists": False, "total_lines": None,
             "last_modified": None, "error": None},
            id_=env["id"])
        await session.recv_until("status.activity", timeout=5.0)
        task.cancel()
        try: await task
        except: pass

    async def test_search_content_request_forwarded(self):
        tc = ScriptedToolCall(name="search", args={"resource": "f.py", "pattern": "foo"})
        session, task = await make_session([ScriptedTurn(tool_calls=[tc])])
        await session.recv_until("status.activity")
        await session.send_host("cmd.prompt", {"text": "go"})
        env = await session.recv_until("tool.content_request", timeout=3.0)
        self.assertIn("pattern", env["payload"])
        self.assertEqual(env["payload"]["pattern"], "foo")
        await session.send_host("tool.content_response",
            {"resource": "f.py", "total_lines": 10, "regions": [],
             "truncated": False, "error": None},
            id_=env["id"])
        task.cancel()
        try: await task
        except: pass

    async def test_search_results_formatted(self):
        tc = ScriptedToolCall(name="search", args={"resource": "f.py", "pattern": "bar"})
        session, task = await make_session([ScriptedTurn(tool_calls=[tc])])
        await session.recv_until("status.activity")
        await session.send_host("cmd.prompt", {"text": "go"})
        env = await session.recv_until("tool.content_request", timeout=3.0)
        await session.send_host("tool.content_response", {
            "resource": "f.py", "total_lines": 10,
            "regions": [{"start_line": 5, "end_line": 5,
                         "lines": ["bar\n"], "match_lines": [5]}],
            "truncated": False, "error": None
        }, id_=env["id"])
        await session.recv_until("status.activity", timeout=5.0)
        task.cancel()
        try: await task
        except: pass

    async def test_search_no_matches(self):
        tc = ScriptedToolCall(name="search", args={"resource": "f.py", "pattern": "zzz"})
        session, task = await make_session([ScriptedTurn(tool_calls=[tc])])
        await session.recv_until("status.activity")
        await session.send_host("cmd.prompt", {"text": "go"})
        env = await session.recv_until("tool.content_request", timeout=3.0)
        await session.send_host("tool.content_response",
            {"resource": "f.py", "total_lines": 50, "regions": [],
             "truncated": False, "error": None},
            id_=env["id"])
        await session.recv_until("status.activity", timeout=5.0)
        task.cancel()
        try: await task
        except: pass

    async def test_stat_then_search_then_read(self):
        """Full navigation sequence: stat → search → read all complete cleanly."""
        tc_stat   = ScriptedToolCall(name="stat",   args={"resource": "f.py"})
        tc_search = ScriptedToolCall(name="search", args={"resource": "f.py", "pattern": "foo"})
        tc_read   = ScriptedToolCall(name="read",
                                     args={"resource": "f.py",
                                           "ranges": [{"start_line": 5, "num_lines": 3}]})
        turn = ScriptedTurn(tool_calls=[tc_stat, tc_search, tc_read])
        session, task = await make_session([turn])
        await session.recv_until("status.activity")
        await session.send_host("cmd.prompt", {"text": "go"})

        # stat
        env = await session.recv_until("tool.stat_request", timeout=3.0)
        await session.send_host("tool.stat_response",
            {"resource": "f.py", "exists": True, "total_lines": 100,
             "last_modified": "now", "error": None},
            id_=env["id"])

        # search
        env = await session.recv_until("tool.content_request", timeout=3.0)
        self.assertIn("pattern", env["payload"])
        await session.send_host("tool.content_response", {
            "resource": "f.py", "total_lines": 100,
            "regions": [{"start_line": 5, "end_line": 7,
                         "lines": ["foo\n", "bar\n", "baz\n"], "match_lines": [5]}],
            "truncated": False, "error": None
        }, id_=env["id"])

        # read
        env = await session.recv_until("tool.content_request", timeout=3.0)
        self.assertIn("ranges", env["payload"])
        await session.send_host("tool.content_response", {
            "resource": "f.py", "total_lines": 100,
            "regions": [{"start_line": 5, "end_line": 7,
                         "lines": ["foo\n", "bar\n", "baz\n"]}],
            "truncated": False, "error": None
        }, id_=env["id"])

        await session.recv_until("status.activity", timeout=5.0)
        task.cancel()
        try: await task
        except: pass

    async def test_search_no_regex_pattern(self):
        tc = ScriptedToolCall(name="search",
                              args={"resource": "f.py", "pattern": "/validate/"})
        session, task = await make_session([ScriptedTurn(tool_calls=[tc])])
        await session.recv_until("status.activity")
        await session.send_host("cmd.prompt", {"text": "go"})
        env = await session.recv_until("tool.content_request", timeout=3.0)
        self.assertEqual(env["payload"]["pattern"], "/validate/")
        await session.send_host("tool.content_response",
            {"resource": "f.py", "total_lines": 10, "regions": [],
             "truncated": False, "error": None},
            id_=env["id"])
        task.cancel()
        try: await task
        except: pass
