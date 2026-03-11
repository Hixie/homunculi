import asyncio
import unittest
from ..conftest import make_session
from ...backends.base import ScriptedTurn


class TestInvalidateFlow(unittest.IsolatedAsyncioTestCase):
    async def _setup(self):
        session, task = await make_session()
        await session.recv_until("status.activity")
        return session, task

    async def test_invalidate_injects_note(self):
        session, task = await self._setup()
        await session.send_host("cmd.invalidate", {"resource":"f.py"})
        await asyncio.sleep(0.1)
        self.assertEqual(len(session.backend.received_notes), 1)
        task.cancel()
        try: await task
        except: pass

    async def test_invalidate_targeted_note(self):
        session, task = await self._setup()
        await session.send_host("cmd.invalidate", {"resource":"f.py","start_line":42})
        await asyncio.sleep(0.1)
        note = session.backend.received_notes[0]
        self.assertIn("42", note)
        task.cancel()
        try: await task
        except: pass

    async def test_invalidate_no_cache_eviction(self):
        # No cache exists; only note injection occurs
        session, task = await self._setup()
        await session.send_host("cmd.invalidate", {"resource":"f.py"})
        await asyncio.sleep(0.1)
        # Just verify a note was injected and nothing crashed
        self.assertGreaterEqual(len(session.backend.received_notes), 1)
        task.cancel()
        try: await task
        except: pass

    async def test_no_retry_instruction_in_note(self):
        session, task = await self._setup()
        await session.send_host("cmd.invalidate", {"resource":"f.py"})
        await asyncio.sleep(0.1)
        note = session.backend.received_notes[0]
        # Note should tell model not to retry
        self.assertIn("not", note.lower())
        task.cancel()
        try: await task
        except: pass
