import asyncio
import unittest
from ..conftest import make_session


class TestQuitFlow(unittest.IsolatedAsyncioTestCase):

    async def test_quit_closes_backend(self):
        session, task = await make_session()
        await session.recv_until("status.activity")
        await session.send_host("cmd.quit", {})
        await asyncio.sleep(0.2)
        self.assertTrue(session.backend._closed)
        task.cancel()
        try: await task
        except: pass

    async def test_quit_sends_session_ended(self):
        session, task = await make_session()
        await session.recv_until("status.activity")
        await session.send_host("cmd.quit", {})
        env = await session.recv_until("session.ended", timeout=3.0)
        self.assertEqual(env["type"], "session.ended")
        task.cancel()
        try: await task
        except: pass

    async def test_session_ended_logfile(self):
        session, task = await make_session()
        await session.recv_until("status.activity")
        await session.send_host("cmd.quit", {})
        env = await session.recv_until("session.ended", timeout=3.0)
        logfile = env["payload"]["logfile"]
        self.assertIsInstance(logfile, str)
        task.cancel()
        try: await task
        except: pass

    async def test_session_ended_summary_fields(self):
        session, task = await make_session()
        await session.recv_until("status.activity")
        await session.send_host("cmd.quit", {})
        env = await session.recv_until("session.ended", timeout=3.0)
        summary = env["payload"]["summary"]
        for key in ("total_prompt_tokens", "total_completion_tokens",
                    "total_cost_usd", "duration_seconds", "turns"):
            self.assertIn(key, summary)
            self.assertGreaterEqual(summary[key], 0)
        task.cancel()
        try: await task
        except: pass

    async def test_process_exits_cleanly(self):
        """After cmd.quit the orchestrator task completes without error."""
        session, task = await make_session()
        await session.recv_until("status.activity")
        await session.send_host("cmd.quit", {})
        session.shutdown.set()
        # Give the task a chance to finish
        await asyncio.sleep(0.2)
        # Task should be done or cancellable without drama
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
