import asyncio
import unittest
from ..conftest import make_session
from ...backends.base import ScriptedTurn, NormalizedUsage


class TestFullSession(unittest.IsolatedAsyncioTestCase):

    async def _fresh(self, script=None):
        session, task = await make_session(script or [])
        await session.recv_until("status.activity")  # consume CONNECTED idle
        return session, task

    async def test_startup_idle_activity(self):
        session, task = await make_session()
        env = await session.recv_until("status.activity")
        self.assertEqual(env["payload"]["state"], "idle")
        task.cancel()
        try: await task
        except: pass

    async def test_prompt_triggers_generating(self):
        session, task = await self._fresh([ScriptedTurn(text_chunks=["hi"])])
        await session.send_host("cmd.prompt", {"text": "hello"})
        env = await session.recv_until("status.activity", timeout=3.0)
        self.assertIn(env["payload"]["state"], ("generating", "idle"))
        task.cancel()
        try: await task
        except: pass

    async def test_text_delta_forwarded(self):
        session, task = await self._fresh([ScriptedTurn(text_chunks=["streamed text"])])
        await session.send_host("cmd.prompt", {"text": "go"})
        env = await session.recv_until("model.text_delta", timeout=3.0)
        self.assertEqual(env["type"], "model.text_delta")
        task.cancel()
        try: await task
        except: pass

    async def test_model_text_delta_content(self):
        session, task = await self._fresh([ScriptedTurn(text_chunks=["hello world"])])
        await session.send_host("cmd.prompt", {"text": "go"})
        env = await session.recv_until("model.text_delta", timeout=3.0)
        self.assertEqual(env["payload"]["text"], "hello world")
        task.cancel()
        try: await task
        except: pass

    async def test_turn_completes(self):
        session, task = await self._fresh([ScriptedTurn(text_chunks=["done"])])
        await session.send_host("cmd.prompt", {"text": "go"})
        # Keep draining until we see idle with "Turn complete"
        for _ in range(20):
            env = await session.recv_until("status.activity", timeout=3.0)
            if env["payload"]["state"] == "idle" and "Turn" in env["payload"]["description"]:
                break
        else:
            self.fail("Never reached idle 'Turn complete'")
        task.cancel()
        try: await task
        except: pass

    async def test_usage_emitted(self):
        session, task = await self._fresh([ScriptedTurn(text_chunks=["hi"])])
        await session.send_host("cmd.prompt", {"text": "go"})
        env = await session.recv_until("status.usage", timeout=5.0)
        self.assertIn("total_tokens", env["payload"])
        task.cancel()
        try: await task
        except: pass

    async def test_multiple_turns(self):
        """Two sequential prompts complete independently."""
        session, task = await make_session([
            ScriptedTurn(text_chunks=["first response"],
                         usage=NormalizedUsage(input_tokens=10, output_tokens=5)),
            ScriptedTurn(text_chunks=["second response"],
                         usage=NormalizedUsage(input_tokens=20, output_tokens=8)),
        ])
        await session.recv_until("status.activity")  # CONNECTED

        # First turn
        await session.send_host("cmd.prompt", {"text": "first"})
        delta1 = await session.recv_until("model.text_delta", timeout=3.0)
        self.assertEqual(delta1["payload"]["text"], "first response")
        await session.recv_until("status.usage", timeout=3.0)

        # Second turn
        await session.send_host("cmd.prompt", {"text": "second"})
        delta2 = await session.recv_until("model.text_delta", timeout=3.0)
        self.assertEqual(delta2["payload"]["text"], "second response")
        await session.recv_until("status.usage", timeout=3.0)

        task.cancel()
        try: await task
        except: pass
