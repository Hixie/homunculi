"""
test_tui_shutdown.py
====================
Tests for the async shutdown mechanics of modelrelay_tui — specifically the
threading boundary between the curses main thread and the asyncio worker thread.

These tests do NOT require curses, a real terminal, or the modelrelay subprocess.
They exercise the same patterns used by _run_curses/_run_async directly.

Run with:
    python -m pytest test_tui_shutdown.py -v
or:
    python -m unittest test_tui_shutdown -v
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
import unittest


# ─── Helpers that mirror the _run_curses/_run_async pattern ──────────────────

def run_session(coro_factory, *, cancel_after: float | None = None,
                stop_loop_instead: bool = False) -> dict:
    """
    Run the given async factory (returns a coroutine) in a worker thread using
    a fresh event loop, mirroring the exact structure of _run_curses/_run_async.

    cancel_after: seconds after which the main thread cancels the task
                  (simulates Ctrl+C path).
    stop_loop_instead: if True, call loop.stop() instead of task.cancel()
                       — this is the old, broken behaviour.

    Returns a dict with keys:
        completed      bool    whether run_until_complete returned normally
        cancelled      bool    whether CancelledError was the exit reason
        error          Exception | None
        task_ref       list    the task, for post-mortem inspection
        loop_closed    bool    whether the loop was closed by _run_async
    """
    result = {
        "completed": False,
        "cancelled": False,
        "error": None,
        "task_ref": [],
        "loop_closed": False,
    }
    main_task_holder: list[asyncio.Task] = []
    exc_holder: list[BaseException] = []

    loop = asyncio.new_event_loop()

    def _run_async():
        asyncio.set_event_loop(loop)
        try:
            async def _run_with_task():
                task = asyncio.current_task()
                if task is not None:
                    main_task_holder.append(task)
                    result["task_ref"].append(task)
                await coro_factory()

            loop.run_until_complete(_run_with_task())
            result["completed"] = True
        except asyncio.CancelledError:
            result["cancelled"] = True
        except Exception as e:
            exc_holder.append(e)
            result["error"] = e
        finally:
            loop.close()
            result["loop_closed"] = True

    t = threading.Thread(target=_run_async, daemon=True)
    t.start()

    if cancel_after is not None:
        time.sleep(cancel_after)
        if stop_loop_instead:
            # ── Old broken behaviour ──────────────────────────────────────────
            if not loop.is_closed():
                loop.call_soon_threadsafe(loop.stop)
        else:
            # ── Correct behaviour ─────────────────────────────────────────────
            if main_task_holder and not loop.is_closed():
                loop.call_soon_threadsafe(main_task_holder[0].cancel)

    t.join(timeout=5)
    if t.is_alive():
        raise TimeoutError("Worker thread did not finish within 5 s")

    if exc_holder:
        result["error"] = exc_holder[0]
    return result


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestNormalShutdown(unittest.TestCase):
    """Session that ends naturally (equivalent to /quit posting 'quit')."""

    def _make_short_coro(self):
        """A coroutine that just completes immediately."""
        async def _coro():
            await asyncio.sleep(0)
        return _coro()

    def test_completes_without_error(self):
        result = run_session(lambda: self._make_short_coro())
        self.assertIsNone(result["error"],
            f"Expected no error on clean exit, got: {result['error']}")

    def test_run_until_complete_returns_normally(self):
        result = run_session(lambda: self._make_short_coro())
        self.assertTrue(result["completed"],
            "run_until_complete should return True on clean exit")

    def test_loop_is_closed_after_exit(self):
        result = run_session(lambda: self._make_short_coro())
        self.assertTrue(result["loop_closed"],
            "Event loop must be closed after the session ends")


class TestTaskCancellation(unittest.TestCase):
    """Ctrl+C path: main thread cancels the task via task.cancel()."""

    def _make_blocking_coro(self):
        """A coroutine that blocks until cancelled."""
        async def _coro():
            await asyncio.sleep(60)   # would block forever without cancellation
        return _coro()

    def test_task_cancel_does_not_raise_runtime_error(self):
        """
        task.cancel() must exit via CancelledError, never via RuntimeError.
        This is the regression test for the original bug where loop.stop()
        raised RuntimeError: 'Event loop stopped before Future completed'.
        """
        result = run_session(
            lambda: self._make_blocking_coro(),
            cancel_after=0.05,
            stop_loop_instead=False,
        )
        self.assertNotIsInstance(result["error"], RuntimeError,
            "task.cancel() must not produce RuntimeError")
        self.assertIsNone(result["error"],
            f"Expected no error after task.cancel(), got: {result['error']}")

    def test_task_cancel_is_recognised_as_cancelled(self):
        result = run_session(
            lambda: self._make_blocking_coro(),
            cancel_after=0.05,
            stop_loop_instead=False,
        )
        self.assertTrue(result["cancelled"],
            "CancelledError should be the exit reason after task.cancel()")

    def test_loop_is_closed_after_cancellation(self):
        result = run_session(
            lambda: self._make_blocking_coro(),
            cancel_after=0.05,
            stop_loop_instead=False,
        )
        self.assertTrue(result["loop_closed"],
            "Event loop must be closed even after cancellation")

    def test_worker_thread_joins_cleanly(self):
        """The worker thread must not linger after task cancellation."""
        # run_session raises TimeoutError if t.join(5) times out
        try:
            run_session(
                lambda: self._make_blocking_coro(),
                cancel_after=0.05,
                stop_loop_instead=False,
            )
        except TimeoutError:
            self.fail("Worker thread did not join after task cancellation")


class TestLoopStopRegression(unittest.TestCase):
    """
    Document the BROKEN behaviour that was fixed, so future regressions
    are immediately obvious.
    """

    def _make_blocking_coro(self):
        async def _coro():
            await asyncio.sleep(60)
        return _coro()

    def test_loop_stop_raises_runtime_error(self):
        """
        Calling loop.stop() while run_until_complete is active raises
        RuntimeError — this is exactly the bug that was fixed.
        This test documents that behaviour and ASSERTS it raises, so that
        if asyncio ever changes semantics the other tests become meaningful.
        """
        result = run_session(
            lambda: self._make_blocking_coro(),
            cancel_after=0.05,
            stop_loop_instead=True,   # the old broken approach
        )
        self.assertIsInstance(result["error"], RuntimeError,
            "loop.stop() inside run_until_complete MUST raise RuntimeError — "
            "if this test fails asyncio has changed behaviour")


class TestCancelledErrorIsNotPropagated(unittest.TestCase):
    """CancelledError from Ctrl+C must be swallowed, not re-raised to caller."""

    def _make_blocking_coro(self):
        async def _coro():
            await asyncio.sleep(60)
        return _coro()

    def test_no_exception_escapes_to_main_thread(self):
        """
        After task.cancel() the exc_holder must be empty — CancelledError
        is expected and should not be surfaced as an application error.
        """
        result = run_session(
            lambda: self._make_blocking_coro(),
            cancel_after=0.05,
            stop_loop_instead=False,
        )
        self.assertIsNone(result["error"],
            "CancelledError must not escape to the main thread as an error")


class TestTaskHolderPopulation(unittest.TestCase):
    """The main_task_holder list must be populated before cancel_after fires."""

    def test_task_ref_is_captured(self):
        """
        The task holder must contain the running task so cancel() can reach it.
        If it's empty when Ctrl+C arrives, the cancellation silently does nothing.
        """
        # Use a short sleep so the task is definitely running when we check
        async def _coro():
            await asyncio.sleep(0.5)

        result = run_session(lambda: _coro(), cancel_after=0.05)
        self.assertEqual(len(result["task_ref"]), 1,
            "main_task_holder must contain exactly one task")

    def test_cancelled_task_is_a_task_object(self):
        async def _coro():
            await asyncio.sleep(60)

        result = run_session(lambda: _coro(), cancel_after=0.05)
        task = result["task_ref"][0] if result["task_ref"] else None
        self.assertIsNotNone(task)
        self.assertIsInstance(task, asyncio.Task)


if __name__ == "__main__":
    unittest.main(verbosity=2)
