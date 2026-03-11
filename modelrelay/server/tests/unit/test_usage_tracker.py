import asyncio
import unittest
from ...model.usage_tracker import UsageTracker
from ...backends.base import NormalizedUsage


class TestUsageTracker(unittest.TestCase):

    def test_accumulates_tokens(self):
        t = UsageTracker("test")
        t.update(NormalizedUsage(input_tokens=100, output_tokens=50))
        t.update(NormalizedUsage(input_tokens=200, output_tokens=100))
        d = t.as_dict()
        self.assertEqual(d["prompt_tokens"], 300)
        self.assertEqual(d["completion_tokens"], 150)
        self.assertEqual(d["total_tokens"], 450)

    def test_cost_anthropic_opus(self):
        t = UsageTracker("claude-opus-4-5")
        t.update(NormalizedUsage(input_tokens=1_000_000, output_tokens=1_000_000))
        d = t.as_dict()
        self.assertGreater(d["cost_usd"], 0)

    def test_cost_openai_gpt4o(self):
        t = UsageTracker("gpt-4o")
        t.update(NormalizedUsage(input_tokens=1_000_000, output_tokens=1_000_000))
        d = t.as_dict()
        self.assertGreater(d["cost_usd"], 0)

    def test_unknown_model_default_pricing(self):
        t = UsageTracker("some-unknown-model")
        t.update(NormalizedUsage(input_tokens=100, output_tokens=100))
        d = t.as_dict()
        self.assertGreaterEqual(d["cost_usd"], 0)

    def test_session_summary_fields(self):
        t = UsageTracker()
        t.update(NormalizedUsage(input_tokens=10, output_tokens=5))
        s = t.session_summary()
        for key in ("total_prompt_tokens", "total_completion_tokens",
                    "total_cost_usd", "duration_seconds", "turns"):
            self.assertIn(key, s)

    def test_emits_on_update(self):
        t = UsageTracker()
        t.update(NormalizedUsage(input_tokens=10, output_tokens=5))
        d = t.as_dict()
        self.assertEqual(d["prompt_tokens"], 10)


class TestUsageTrackerPolling(unittest.IsolatedAsyncioTestCase):

    async def test_emits_on_poll_interval(self):
        """Usage is emitted on a polling timer even without a TURN_DONE event."""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from ...tests.conftest import make_session
        from ...backends.base import ScriptedTurn

        # Empty script: the session starts but the model never says anything.
        # The polling timer should fire and push a status.usage message.
        session, task = await make_session([], usage_interval_s=0.1)
        await session.recv_until("status.activity")   # CONNECTED idle

        # Without any prompt, usage polling should still fire within ~0.3 s
        env = await session.recv_until("status.usage", timeout=1.0)
        self.assertEqual(env["type"], "status.usage")
        self.assertIn("total_tokens", env["payload"])

        task.cancel()
        try: await task
        except: pass
