"""
test_input_queue.py
===================
Tests for the two-queue input routing in ModelRelaySession.

The invariant: keyboard input is routed to exactly one queue at a time.

  • _prompt_queue  — receives user prompt text; consumed by _prompt_loop
  • _confirm_queue — receives y/n answers;    consumed by _ask_confirm

The TUI routes to _confirm_queue when _input_confirming is True, and to
_prompt_queue otherwise.  Because _ask_confirm sets confirming=True before
waiting and clears it when done, the two consumers can never receive each
other's input.

Run with:
    python -m unittest test_input_queue -v
"""
from __future__ import annotations

import asyncio
import sys
import unittest
import unittest.mock as mock

# ── curses stub (no real terminal needed) ─────────────────────────────────────

curses_mock = mock.MagicMock()
for attr, val in [
    ("COLOR_BLACK", 0), ("COLOR_WHITE", 7), ("COLOR_GREEN", 2),
    ("COLOR_RED", 1),   ("COLOR_CYAN",  6), ("COLOR_YELLOW", 3),
    ("COLOR_MAGENTA", 5), ("COLOR_BLUE", 4), ("COLORS", 256),
    ("A_BOLD", 1), ("A_DIM", 2),
    ("KEY_ENTER", 343), ("KEY_BACKSPACE", 263), ("KEY_DC", 330),
    ("KEY_LEFT", 260), ("KEY_RIGHT", 261), ("KEY_HOME", 262),
    ("KEY_END", 360), ("KEY_RESIZE", 410),
]:
    setattr(curses_mock, attr, val)
sys.modules["curses"] = curses_mock

sys.path.insert(0, "/mnt/user-data/outputs")
import modelrelay_tui as tui_mod
from modelrelay_tui import ModelRelaySession, Config, Sandbox


# ── Minimal stubs ─────────────────────────────────────────────────────────────

class NullTUI:
    """Collects posts but does nothing visual."""
    def __init__(self):
        self.posts: list[dict] = []

    def post(self, kind: str, **data):
        self.posts.append({"kind": kind, **data})

    def set_async(self, loop, prompt_q, confirm_q):
        pass

    def posts_of(self, kind: str) -> list[dict]:
        return [p for p in self.posts if p["kind"] == kind]


def make_session(tmp_path=None) -> ModelRelaySession:
    import tempfile, pathlib
    if tmp_path is None:
        tmp_path = pathlib.Path(tempfile.mkdtemp())
    cfg = Config(working_dir=tmp_path, backend="anthropic", model="claude-test")
    sandbox = Sandbox(tmp_path)
    dummy_exe = tmp_path / "modelrelay"
    dummy_exe.touch()
    return ModelRelaySession(cfg, sandbox, dummy_exe, NullTUI())


# ── TestAskConfirm ────────────────────────────────────────────────────────────

class TestAskConfirm(unittest.IsolatedAsyncioTestCase):
    """Unit tests for _ask_confirm: reads from _confirm_queue."""

    async def test_y_returns_true(self):
        s = make_session()
        await s._confirm_queue.put("y")
        self.assertTrue(await s._ask_confirm("Apply?"))

    async def test_yes_returns_true(self):
        s = make_session()
        await s._confirm_queue.put("yes")
        self.assertTrue(await s._ask_confirm("Apply?"))

    async def test_n_returns_false(self):
        s = make_session()
        await s._confirm_queue.put("n")
        self.assertFalse(await s._ask_confirm("Apply?"))

    async def test_no_returns_false(self):
        s = make_session()
        await s._confirm_queue.put("no")
        self.assertFalse(await s._ask_confirm("Apply?"))

    async def test_invalid_then_y(self):
        s = make_session()
        for item in ("maybe", "sure", "y"):
            await s._confirm_queue.put(item)
        self.assertTrue(await s._ask_confirm("Apply?"))
        reprompts = [p for p in s.tui.posts if p.get("text") == "Please type y or n"]
        self.assertEqual(len(reprompts), 2)

    async def test_none_treated_as_yes(self):
        s = make_session()
        await s._confirm_queue.put(None)
        self.assertTrue(await s._ask_confirm("Apply?"))

    async def test_case_insensitive(self):
        for answer, expected in [("Y", True), ("N", False), ("YES", True), ("NO", False)]:
            with self.subTest(answer=answer):
                s = make_session()
                await s._confirm_queue.put(answer)
                self.assertEqual(expected, await s._ask_confirm("Apply?"))

    async def test_reads_confirm_queue_not_prompt_queue(self):
        """A value on _prompt_queue must NOT be consumed by _ask_confirm."""
        s = make_session()
        await s._prompt_queue.put("hello")    # goes to wrong queue
        await s._confirm_queue.put("y")       # correct queue
        result = await asyncio.wait_for(s._ask_confirm("Apply?"), timeout=1.0)
        self.assertTrue(result)
        # prompt_queue item must be untouched
        self.assertFalse(s._prompt_queue.empty(),
            "_prompt_queue item must not be consumed by _ask_confirm")

    async def test_posts_confirming_flag(self):
        s = make_session()
        await s._confirm_queue.put("y")
        await s._ask_confirm("Apply?")
        confirming_posts = [p for p in s.tui.posts
                           if p["kind"] == "input_active" and p.get("confirming")]
        self.assertTrue(confirming_posts,
            "_ask_confirm must post input_active(confirming=True)")

    async def test_deactivates_input_after_answer(self):
        s = make_session()
        await s._confirm_queue.put("y")
        await s._ask_confirm("Apply?")
        input_posts = [p for p in s.tui.posts if p["kind"] == "input_active"]
        self.assertFalse(input_posts[-1].get("active"),
            "Last input_active post must be active=False")


# ── TestQueueSeparation ───────────────────────────────────────────────────────

class TestQueueSeparation(unittest.IsolatedAsyncioTestCase):
    """
    Core invariant: _prompt_queue and _confirm_queue are fully independent.
    Items on one queue are never consumed by the reader of the other.
    """

    async def test_confirm_answer_does_not_reach_prompt_loop(self):
        """
        'y' on the confirm queue must not be consumed by _prompt_loop.
        """
        s = make_session()
        received_by_prompt: list[str] = []

        async def fake_prompt_consumer():
            item = await s._prompt_queue.get()
            received_by_prompt.append(item)

        # Start a consumer on the prompt queue
        pt = asyncio.create_task(fake_prompt_consumer())
        await asyncio.sleep(0.01)

        # Put 'y' on confirm queue — must not be stolen by prompt consumer
        await s._confirm_queue.put("y")
        await s._confirm_queue.put("also_confirm")

        # Now put a real prompt message
        await s._prompt_queue.put("hello")
        await asyncio.wait_for(pt, timeout=1.0)

        self.assertEqual(received_by_prompt, ["hello"],
            "_prompt_loop must only receive from _prompt_queue")
        self.assertFalse(s._confirm_queue.empty(),
            "confirm items must not be consumed by _prompt_loop")

    async def test_prompt_text_does_not_reach_ask_confirm(self):
        """
        A user message on _prompt_queue must not satisfy _ask_confirm.
        """
        s = make_session()

        # Start _ask_confirm — it reads from _confirm_queue
        await s._prompt_queue.put("this is a prompt, not a confirm answer")
        await s._confirm_queue.put("y")

        result = await asyncio.wait_for(s._ask_confirm("Apply?"), timeout=1.0)
        self.assertTrue(result)
        # Prompt queue item must be untouched
        self.assertFalse(s._prompt_queue.empty())

    async def test_concurrent_prompt_and_confirm(self):
        """
        _prompt_loop and _ask_confirm can run concurrently on their
        respective queues with no interference.
        """
        s = make_session()
        confirm_result: list[bool] = []
        prompt_result:  list[str]  = []

        async def do_confirm():
            async with s._confirm_lock:
                result = await s._ask_confirm("Apply?")
                confirm_result.append(result)

        async def do_prompt():
            item = await s._prompt_queue.get()
            prompt_result.append(item)

        ct = asyncio.create_task(do_confirm())
        pt = asyncio.create_task(do_prompt())
        await asyncio.sleep(0.01)

        # Feed both queues simultaneously
        await s._confirm_queue.put("n")
        await s._prompt_queue.put("hello model")

        await asyncio.gather(ct, pt)

        self.assertEqual(confirm_result, [False])
        self.assertEqual(prompt_result,  ["hello model"])

    async def test_no_items_lost_across_multiple_confirms(self):
        """
        Three sequential confirms followed by a user message — nothing lost
        and nothing misrouted.
        """
        s = make_session()
        for answer in ("y", "n", "y"):
            await s._confirm_queue.put(answer)
            async with s._confirm_lock:
                await s._ask_confirm(f"Confirm?")

        await s._prompt_queue.put("final message")
        item = await asyncio.wait_for(s._prompt_queue.get(), timeout=1.0)
        self.assertEqual(item, "final message")


# ── TestConfirmAndApply ───────────────────────────────────────────────────────

class TestConfirmAndApply(unittest.IsolatedAsyncioTestCase):
    """Tests for the _confirm_and_apply helper."""

    async def test_apply_called_on_yes(self):
        s = make_session()
        called = []
        await s._confirm_queue.put("y")
        async with s._confirm_lock:
            await s._confirm_and_apply(
                diff=[], question="Apply?",
                apply_fn=lambda: called.append(True) or None,
                resource="f.txt", reject_reason="rejected")
        self.assertEqual(called, [True])

    async def test_apply_not_called_on_no(self):
        s = make_session()
        called = []
        await s._confirm_queue.put("n")
        async with s._confirm_lock:
            await s._confirm_and_apply(
                diff=[], question="Apply?",
                apply_fn=lambda: called.append(True) or None,
                resource="f.txt", reject_reason="rejected")
        self.assertEqual(called, [])

    async def test_rejection_sends_response_then_invalidate(self):
        """On rejection, the tool response must arrive first (to unblock the
        pending future), then cmd.invalidate (to inject the don't-retry note)."""
        import modelrelay_tui as tui_mod
        s = make_session()
        await s._confirm_queue.put("n")
        respond_calls = []
        async with s._confirm_lock:
            await s._confirm_and_apply(
                diff=[], question="Apply?", apply_fn=lambda: None,
                resource="f.txt", reject_reason="User rejected", reject_line=5,
                respond_fn=lambda err: tui_mod.make_envelope("tool.replace_response", {"error": err}))
        # Two items: the tool response and cmd.invalidate
        self.assertFalse(s._send_queue.empty())
        first  = s._send_queue.get_nowait()
        second = s._send_queue.get_nowait()
        self.assertEqual(first.get("type"),  "tool.replace_response")
        self.assertEqual(second.get("type"), "cmd.invalidate")
        self.assertEqual(first["payload"]["error"], "User rejected")

    async def test_success_sends_response_no_invalidate(self):
        """On success, only the tool response is sent (error=None); no invalidate."""
        import modelrelay_tui as tui_mod
        s = make_session()
        await s._confirm_queue.put("y")
        async with s._confirm_lock:
            await s._confirm_and_apply(
                diff=[], question="Apply?", apply_fn=lambda: None,
                resource="f.txt", reject_reason="rejected",
                respond_fn=lambda err: tui_mod.make_envelope("tool.replace_response", {"error": err}))
        self.assertFalse(s._send_queue.empty())
        env = s._send_queue.get_nowait()
        self.assertEqual(env.get("type"), "tool.replace_response")
        self.assertIsNone(env["payload"]["error"])
        self.assertTrue(s._send_queue.empty(), "No cmd.invalidate on success")

    async def test_write_error_sends_response_with_error(self):
        """On write failure, the tool response carries the error string."""
        import modelrelay_tui as tui_mod
        s = make_session()
        await s._confirm_queue.put("y")
        async with s._confirm_lock:
            await s._confirm_and_apply(
                diff=[], question="Apply?", apply_fn=lambda: "disk full",
                resource="f.txt", reject_reason="rejected",
                respond_fn=lambda err: tui_mod.make_envelope("tool.replace_response", {"error": err}))
        self.assertFalse(s._send_queue.empty())
        env = s._send_queue.get_nowait()
        self.assertEqual(env.get("type"), "tool.replace_response")
        self.assertEqual(env["payload"]["error"], "disk full")

    async def test_no_respond_fn_no_send_on_success(self):
        """When respond_fn is omitted (old callers), success sends nothing."""
        s = make_session()
        await s._confirm_queue.put("y")
        async with s._confirm_lock:
            await s._confirm_and_apply(
                diff=[], question="Apply?", apply_fn=lambda: None,
                resource="f.txt", reject_reason="rejected")
        self.assertTrue(s._send_queue.empty())


    async def test_success_posts_checkmark(self):
        s = make_session()
        await s._confirm_queue.put("y")
        async with s._confirm_lock:
            await s._confirm_and_apply(
                diff=[], question="Apply?", apply_fn=lambda: None,
                resource="f.txt", reject_reason="rejected")
        self.assertTrue(any("✓ applied" in p.get("text","")
                            for p in s.tui.posts if p["kind"]=="model_line"))

    async def test_rejection_posts_cross(self):
        s = make_session()
        await s._confirm_queue.put("n")
        async with s._confirm_lock:
            await s._confirm_and_apply(
                diff=[], question="Apply?", apply_fn=lambda: None,
                resource="f.txt", reject_reason="rejected")
        self.assertTrue(any("✗ rejected" in p.get("text","")
                            for p in s.tui.posts if p["kind"]=="model_line"))


# ── TestRaceConditionRegression ───────────────────────────────────────────────

class TestRaceConditionRegression(unittest.IsolatedAsyncioTestCase):
    """
    Demonstrates that the old single-queue design was broken, and that
    the two-queue design fixes it.
    """

    async def test_single_queue_race_is_possible(self):
        """
        With one shared queue, a consumer registered first steals answers
        intended for the other consumer.  Documents the broken state.
        """
        shared_q: asyncio.Queue = asyncio.Queue()
        stolen_by_prompt: list = []
        got_by_confirm:   list = []

        # Register prompt consumer first (simulates _prompt_loop waiting)
        prompt_get = asyncio.create_task(shared_q.get())
        await asyncio.sleep(0)

        # Confirm consumer also waits on the SAME queue
        confirm_get = asyncio.create_task(shared_q.get())
        await asyncio.sleep(0)

        # User types "y" — should go to confirm, but prompt consumer wins
        await shared_q.put("y")
        await asyncio.sleep(0.05)

        if prompt_get.done():
            stolen_by_prompt.append(prompt_get.result())
        if confirm_get.done():
            got_by_confirm.append(confirm_get.result())

        for t in (prompt_get, confirm_get):
            if not t.done():
                t.cancel()
                try: await t
                except asyncio.CancelledError: pass

        self.assertTrue(len(stolen_by_prompt) > 0,
            "Prompt consumer stole 'y' from confirm consumer on shared queue.")
        self.assertEqual(got_by_confirm, [],
            "_ask_confirm was starved — confirms the original bug.")

    async def test_two_queue_design_prevents_race(self):
        """
        With two separate queues, 'y' always reaches _ask_confirm even if
        _prompt_loop is simultaneously waiting for input.
        """
        s = make_session()
        prompt_received:  list[str]  = []
        confirm_received: list[bool] = []

        prompt_waiting = asyncio.Event()

        async def prompt_consumer():
            prompt_waiting.set()
            item = await s._prompt_queue.get()
            prompt_received.append(item)

        async def confirm_consumer():
            async with s._confirm_lock:
                result = await s._ask_confirm("Apply?")
                confirm_received.append(result)

        pt = asyncio.create_task(prompt_consumer())
        await prompt_waiting.wait()   # ensure prompt is waiting
        ct = asyncio.create_task(confirm_consumer())
        await asyncio.sleep(0.02)

        # Put "y" on the confirm queue — must NOT go to prompt
        await s._confirm_queue.put("y")
        # Put a real message on the prompt queue
        await s._prompt_queue.put("hello")

        await asyncio.gather(pt, ct)

        self.assertEqual(confirm_received, [True],
            "'y' must reach _ask_confirm via _confirm_queue")
        self.assertEqual(prompt_received, ["hello"],
            "'hello' must reach _prompt_loop via _prompt_queue")

    async def test_old_lock_in_prompt_loop_deadlock(self):
        """
        Documents the second bug: holding _confirm_lock in _prompt_loop
        while blocked on get() prevents tool handlers from ever acquiring
        the lock.  Asserts the broken behaviour for reference.
        """
        lock = asyncio.Lock()
        tool_got_lock = asyncio.Event()

        async def broken_prompt_loop():
            async with lock:           # holds lock while waiting
                await asyncio.sleep(5) # simulates blocked get()

        async def tool_handler():
            try:
                async with asyncio.timeout(0.1):
                    async with lock:
                        tool_got_lock.set()
            except TimeoutError:
                pass  # expected: couldn't acquire lock

        pt = asyncio.create_task(broken_prompt_loop())
        await asyncio.sleep(0.01)
        tt = asyncio.create_task(tool_handler())
        await asyncio.gather(tt, return_exceptions=True)

        self.assertFalse(tool_got_lock.is_set(),
            "Tool handler was blocked — confirms the lock-in-prompt-loop deadlock.")

        pt.cancel()
        try: await pt
        except asyncio.CancelledError: pass


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestActivityDuringConfirm(unittest.IsolatedAsyncioTestCase):
    """
    _on_activity must not collapse the confirm box while _ask_confirm is
    waiting for the user.  insert/replace are fire-and-forget, so modelrelay
    sends status.activity(generating) almost immediately after the request —
    before the user has answered.
    """

    async def test_activity_generating_does_not_hide_confirm_box(self):
        """
        Sequence:
          1. _ask_confirm sets _confirming=1, posts input_active(True, confirming)
          2. status.activity(generating) fires — must NOT post input_active(False)
          3. User answers "y" → confirm resolves, _confirming back to 0
        """
        s = make_session()
        input_deactivated_during_confirm = []

        original_post = s.tui.post
        confirming_flag = [False]

        def tracking_post(kind, **kw):
            if kind == "input_active" and kw.get("confirming"):
                confirming_flag[0] = True
            if kind == "input_active" and not kw.get("active") and confirming_flag[0]:
                # Record whether _confirming was non-zero at this point
                input_deactivated_during_confirm.append(s._confirming > 0)
            original_post(kind, **kw)

        s.tui.post = tracking_post

        async def run_confirm():
            async with s._confirm_lock:
                return await s._ask_confirm("Apply?")

        confirm_task = asyncio.create_task(run_confirm())
        await asyncio.sleep(0.01)  # let _ask_confirm set _confirming and post

        # Simulate modelrelay sending activity(generating) while confirm is active
        await s._on_activity({"state": "generating", "description": "continuing"})

        # Confirm box should still be up — feed the answer
        await s._confirm_queue.put("y")
        result = await asyncio.wait_for(confirm_task, timeout=1.0)
        self.assertTrue(result)

        # The generating activity must not have deactivated input mid-confirm
        # (any deactivation recorded while _confirming>0 would be a bug)
        self.assertFalse(
            any(input_deactivated_during_confirm),
            "input_active(False) was posted while _confirming > 0 — "
            "this would collapse the confirm box before the user answers"
        )

    async def test_activity_idle_does_not_activate_prompt_during_confirm(self):
        """
        If an idle activity fires while confirming, it must not enable the
        prompt input box (which would route the user's 'y' to _prompt_queue).
        """
        s = make_session()
        prompt_activations_during_confirm = []

        original_post = s.tui.post
        def tracking_post(kind, **kw):
            if kind == "input_active" and kw.get("active") and not kw.get("confirming"):
                prompt_activations_during_confirm.append(s._confirming)
            original_post(kind, **kw)
        s.tui.post = tracking_post

        async def run_confirm():
            async with s._confirm_lock:
                return await s._ask_confirm("Apply?")

        confirm_task = asyncio.create_task(run_confirm())
        await asyncio.sleep(0.01)

        # Spurious idle while confirming
        await s._on_activity({"state": "idle", "description": "Turn complete"})

        await s._confirm_queue.put("n")
        await asyncio.wait_for(confirm_task, timeout=1.0)

        # Any prompt-box activation while _confirming > 0 would be a bug
        active_during_confirm = [c for c in prompt_activations_during_confirm if c > 0]
        self.assertEqual(active_during_confirm, [],
            "input_active(True, confirming=False) must not fire while _confirming > 0")

    async def test_confirming_counter_at_zero_after_answer(self):
        """After _ask_confirm resolves, _confirming must be back to zero."""
        s = make_session()
        await s._confirm_queue.put("y")
        async with s._confirm_lock:
            await s._ask_confirm("Apply?")
        self.assertEqual(s._confirming, 0)

    async def test_confirming_counter_at_zero_after_rejection(self):
        s = make_session()
        await s._confirm_queue.put("n")
        async with s._confirm_lock:
            await s._ask_confirm("Apply?")
        self.assertEqual(s._confirming, 0)

    async def test_activity_resumes_normally_after_confirm(self):
        """
        After the confirm dialog closes, _on_activity(idle) must re-enable
        the prompt input normally.
        """
        s = make_session()
        await s._confirm_queue.put("y")
        async with s._confirm_lock:
            await s._ask_confirm("Apply?")

        self.assertEqual(s._confirming, 0)
        # Now idle arrives — should activate prompt input
        input_actived = []
        original_post = s.tui.post
        s.tui.post = lambda kind, **kw: input_actived.append((kind, kw)) or original_post(kind, **kw)

        await s._on_activity({"state": "idle", "description": "Turn complete"})
        prompt_activations = [(k, kw) for k, kw in input_actived
                              if k == "input_active" and kw.get("active")]
        self.assertTrue(prompt_activations,
            "input_active(True) must fire for idle state after confirm is done")


class TestIdleArrivesBeforeConfirmAnswer(unittest.IsolatedAsyncioTestCase):
    """
    For fire-and-forget requests (replace, insert), modelrelay sends
    activity(generating) and then activity(idle) before the user answers.
    Those are suppressed while _confirming > 0.  When the confirm closes,
    the prompt must be re-activated based on _activity_state.
    """

    async def test_prompt_activated_after_confirm_when_idle_arrived_first(self):
        """
        Sequence:
          1. activity(waiting_for_tool)
          2. tool.replace_request  →  _ask_confirm starts, _confirming=1
          3. activity(generating)  →  suppressed
          4. activity(idle)        →  suppressed (input_active NOT posted)
          5. user answers "y"      →  confirm closes, _confirming=0
        Expected: input_active(True) fires so _prompt_loop can accept input.
        """
        s = make_session()
        input_active_posts = []

        original_post = s.tui.post
        def tracking_post(kind, **kw):
            if kind == "input_active":
                input_active_posts.append(dict(kw))
            original_post(kind, **kw)
        s.tui.post = tracking_post

        async def run_confirm():
            async with s._confirm_lock:
                return await s._ask_confirm("Apply?")

        confirm_task = asyncio.create_task(run_confirm())
        await asyncio.sleep(0.01)  # _ask_confirm running, _confirming=1

        # activity(generating) and activity(idle) arrive before user answers
        await s._on_activity({"state": "generating", "description": "continuing"})
        await s._on_activity({"state": "idle", "description": "Turn complete"})

        # idle was suppressed — now user answers
        await s._confirm_queue.put("y")
        await asyncio.wait_for(confirm_task, timeout=1.0)

        # The last input_active post must be active=True (prompt re-enabled)
        self.assertTrue(input_active_posts,
            "At least one input_active post must have been made")
        last = input_active_posts[-1]
        self.assertTrue(last.get("active"),
            f"Last input_active post must be active=True, got {last}. "
            "When idle arrived while confirming, prompt must be re-activated "
            "after the confirm dialog closes.")

    async def test_prompt_not_activated_if_not_idle_when_confirm_closes(self):
        """
        If the last activity was not idle (e.g. still generating), the prompt
        must NOT be activated when the confirm closes.
        """
        s = make_session()
        input_active_posts = []
        original_post = s.tui.post
        def tracking_post(kind, **kw):
            if kind == "input_active":
                input_active_posts.append(dict(kw))
            original_post(kind, **kw)
        s.tui.post = tracking_post

        async def run_confirm():
            async with s._confirm_lock:
                return await s._ask_confirm("Apply?")

        confirm_task = asyncio.create_task(run_confirm())
        await asyncio.sleep(0.01)

        # Only generating arrived, not idle
        await s._on_activity({"state": "generating", "description": "continuing"})

        await s._confirm_queue.put("n")
        await asyncio.wait_for(confirm_task, timeout=1.0)

        # Last post must be active=False (still generating)
        self.assertTrue(input_active_posts)
        last = input_active_posts[-1]
        self.assertFalse(last.get("active"),
            "Prompt must not activate if state is still 'generating' when confirm closes")

    async def test_activity_state_tracked_correctly(self):
        """_activity_state reflects the most recent activity received."""
        s = make_session()
        self.assertEqual(s._activity_state, "")
        await s._on_activity({"state": "idle", "description": ""})
        self.assertEqual(s._activity_state, "idle")
        await s._on_activity({"state": "generating", "description": ""})
        self.assertEqual(s._activity_state, "generating")
        await s._on_activity({"state": "waiting_for_tool", "description": ""})
        self.assertEqual(s._activity_state, "waiting_for_tool")
