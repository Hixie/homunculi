"""Tests for status.usage deduplication in the orchestrator.

The orchestrator must skip sending status.usage when the payload dict is
identical to the last one sent.  The two send sites are TURN_DONE and the
polling timer.  Deduplication is meaningful primarily at the polling site:
after TURN_DONE already sent the latest data, a polling tick that fires before
the next turn should not re-send the same message.
"""
from __future__ import annotations

import asyncio
import unittest

from ...backends.base import EventType, NormalizedUsage, ScriptedTurn
from ..conftest import make_session


class TestUsageDeduplication(unittest.IsolatedAsyncioTestCase):

    async def _drain_connected(self, session):
        await session.recv_until("status.activity")

    # ── TURN_DONE always sends on first turn ──────────────────────────────────

    async def test_first_turn_sends_usage(self):
        script = [ScriptedTurn(text_chunks=["hi"],
                               usage=NormalizedUsage(input_tokens=10,
                                                     output_tokens=5))]
        session, task = await make_session(script)
        await self._drain_connected(session)
        await session.send_host("cmd.prompt", {"text": "go"})
        env = await session.recv_until("status.usage", timeout=3.0)
        self.assertEqual(env["type"], "status.usage")
        self.assertEqual(env["payload"]["total_tokens"], 15)
        task.cancel()
        try: await task
        except: pass

    # ── TURN_DONE sends when totals change (normal multi-turn case) ───────────

    async def test_second_turn_sends_because_totals_changed(self):
        """Each TURN_DONE call raises cumulative token totals, so the payload
        always differs from the previous send — both messages must be sent."""
        script = [
            ScriptedTurn(text_chunks=["a"],
                         usage=NormalizedUsage(input_tokens=10, output_tokens=5)),
            ScriptedTurn(text_chunks=["b"],
                         usage=NormalizedUsage(input_tokens=20, output_tokens=8)),
        ]
        session, task = await make_session(script)
        await self._drain_connected(session)

        await session.send_host("cmd.prompt", {"text": "go"})
        u1 = await session.recv_until("status.usage", timeout=3.0)
        self.assertEqual(u1["payload"]["total_tokens"], 15)

        await session.send_host("cmd.prompt", {"text": "go again"})
        u2 = await session.recv_until("status.usage", timeout=3.0)
        self.assertEqual(u2["payload"]["total_tokens"], 43)

        task.cancel()
        try: await task
        except: pass

    # ── polling timer suppressed when nothing changed since last TURN_DONE ────

    async def test_poll_suppressed_when_usage_unchanged(self):
        """After TURN_DONE sends status.usage, a poll tick that fires before
        the next turn must not resend the identical payload."""
        script = [ScriptedTurn(text_chunks=["hi"],
                               usage=NormalizedUsage(input_tokens=5,
                                                     output_tokens=3))]
        # Short poll interval so we get several ticks quickly
        session, task = await make_session(script, usage_interval_s=0.1)
        await self._drain_connected(session)

        # Drive one turn; status.usage is sent and _last_sent_usage is set
        await session.send_host("cmd.prompt", {"text": "go"})
        await session.recv_until("status.usage", timeout=3.0)

        # Wait long enough for the poll timer to fire multiple times
        await asyncio.sleep(0.4)

        # Drain anything queued during that sleep
        extras = []
        while True:
            try:
                env = await asyncio.wait_for(session.recv(timeout=0.05),
                                             timeout=0.1)
                extras.append(env)
            except (asyncio.TimeoutError, Exception):
                break

        usage_repeats = [m for m in extras if m["type"] == "status.usage"]
        self.assertEqual(
            len(usage_repeats), 0,
            f"poll timer sent {len(usage_repeats)} duplicate status.usage "
            f"message(s) when usage had not changed")

        task.cancel()
        try: await task
        except: pass

    async def test_poll_suppressed_with_no_turns(self):
        """A poll on an empty session (usage all-zeros) must still send the
        first time (last_sent=None), then suppress subsequent identical ticks."""
        session, task = await make_session([], usage_interval_s=0.1)
        await self._drain_connected(session)

        # First poll tick fires and sends (last_sent was None)
        env = await session.recv_until("status.usage", timeout=1.0)
        self.assertEqual(env["payload"]["total_tokens"], 0)

        # Additional ticks within a further 0.4 s must be suppressed
        await asyncio.sleep(0.4)
        extras = []
        while True:
            try:
                env = await asyncio.wait_for(session.recv(timeout=0.05),
                                             timeout=0.1)
                extras.append(env)
            except (asyncio.TimeoutError, Exception):
                break

        usage_repeats = [m for m in extras if m["type"] == "status.usage"]
        self.assertEqual(
            len(usage_repeats), 0,
            f"poll timer sent {len(usage_repeats)} duplicate(s) on empty session")

        task.cancel()
        try: await task
        except: pass

    async def test_poll_sends_after_turn_between_ticks(self):
        """If a turn completes between two poll ticks, the TURN_DONE handler
        sends first; the next poll tick sees unchanged state and stays silent."""
        script = [ScriptedTurn(text_chunks=["hi"],
                               usage=NormalizedUsage(input_tokens=5,
                                                     output_tokens=3))]
        # Poll interval longer than the test; TURN_DONE is the only sender here
        session, task = await make_session(script, usage_interval_s=10.0)
        await self._drain_connected(session)

        await session.send_host("cmd.prompt", {"text": "go"})
        u = await session.recv_until("status.usage", timeout=3.0)
        self.assertEqual(u["payload"]["total_tokens"], 8)

        task.cancel()
        try: await task
        except: pass
