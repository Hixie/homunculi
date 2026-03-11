"""
test_scroll.py
==============
Tests for the model-window scrolling mechanism.

Behaviour under test
--------------------
  * _scroll_by(+N)  — scroll back N lines from the live bottom.
  * _scroll_by(-N)  — scroll forward N lines toward the live bottom.
  * _scroll_to_bottom()  — snap back to live auto-scroll mode.
  * _page_up() / _page_down()  — one page of movement.
  * Anchor semantics: entering scroll mode snapshots _scroll_total so that
    new output arriving while the user is reading does not shift the view.
  * Scroll indicator: the bottom border shows "↑ N" when scrolled, plus a
    "M new ↓" badge when new content has arrived while scrolled.
  * Live view: when scroll_offset == 0 the newest lines are always visible.
  * Frozen view: when scroll_offset > 0 the view shows the anchored slice.
  * Resize resets scroll offset to 0.
  * Sending a message (Enter) snaps to the live bottom.
  * KEY_PPAGE / KEY_NPAGE are routed to page_up / page_down in the input
    loop, regardless of whether the input widget is active.
  * Mouse wheel events (BUTTON4/5) are routed to _scroll_by(±3).

Run with:
    python -m unittest test_scroll -v
"""
from __future__ import annotations

import sys
import unittest
import unittest.mock as mock


# ── Curses stub ──────────────────────────────────────────────────────────────

_curses_mock = mock.MagicMock()
for _attr, _val in [
    ("COLOR_BLACK", 0), ("COLOR_WHITE", 7), ("COLOR_GREEN", 2),
    ("COLOR_RED",   1), ("COLOR_CYAN",  6), ("COLOR_YELLOW", 3),
    ("COLOR_MAGENTA", 5), ("COLOR_BLUE", 4), ("COLORS", 256),
    ("A_BOLD", 1), ("A_DIM", 2),
    ("KEY_ENTER",     343), ("KEY_BACKSPACE", 263), ("KEY_DC",    330),
    ("KEY_LEFT",      260), ("KEY_RIGHT",     261), ("KEY_HOME",  262),
    ("KEY_END",       360), ("KEY_RESIZE",    410),
    ("KEY_PPAGE",     339), ("KEY_NPAGE",     338),   # Page Up / Page Down
    ("KEY_MOUSE",     409),
    ("BUTTON4_PRESSED", 0x10000),   # scroll-wheel up
    ("BUTTON5_PRESSED", 0x20000),   # scroll-wheel down
    ("ALL_MOUSE_EVENTS",      0x1fffffff),
    ("REPORT_MOUSE_POSITION", 0x8000000),
    ("LINES", 24), ("COLS", 80),
]:
    setattr(_curses_mock, _attr, _val)

sys.modules["curses"] = _curses_mock

sys.path.insert(0, "/mnt/user-data/outputs")
import modelrelay_tui as tui_mod   # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tui(h: int = 40, w: int = 80) -> tui_mod.TUI:
    _curses_mock.LINES = h
    _curses_mock.COLS  = w
    scr = mock.MagicMock()
    scr.getmaxyx.return_value = (h, w)
    return tui_mod.TUI(scr)


def _load(t: tui_mod.TUI, n: int, text: str = "line"):
    """Append n model_lines to the TUI."""
    for i in range(n):
        t._model_lines.append((f"{text} {i}", "normal"))


def _border_text(t: tui_mod.TUI) -> str:
    """Return whatever was last written to the bottom border row of the model
    window.  Collects all addstr calls on the model win for the last draw, picks
    the row == model_h - 1 ones, and joins them left-to-right."""
    t._model_win.reset_mock()
    t._draw_model()
    border_row = t._model_h - 1
    parts = []
    for call in t._model_win.addstr.call_args_list:
        args = call.args
        if len(args) >= 3 and args[0] == border_row:
            parts.append((args[1], args[2]))  # (col, text)
    parts.sort()
    return "".join(text for _, text in parts)


def _visible_texts(t: tui_mod.TUI) -> list[str]:
    """Return the text strings drawn in the content rows (not border)."""
    t._model_win.reset_mock()
    t._draw_model()
    border_row = t._model_h - 1
    rows: dict[int, str] = {}
    for call in t._model_win.addstr.call_args_list:
        args = call.args
        if len(args) >= 3 and args[0] != border_row:
            rows[args[0]] = args[2]
    return [rows[r] for r in sorted(rows)]


# ── Initial state ─────────────────────────────────────────────────────────────

class TestInitialScrollState(unittest.TestCase):
    def test_scroll_offset_starts_at_zero(self):
        t = _make_tui()
        self.assertEqual(t._scroll_offset, 0)

    def test_scroll_total_starts_at_zero(self):
        t = _make_tui()
        self.assertEqual(t._scroll_total, 0)

    def test_no_indicator_when_not_scrolled(self):
        t = _make_tui()
        _load(t, 50)
        border = _border_text(t)
        self.assertNotIn("↑", border)
        self.assertNotIn("new", border)


# ── _scroll_by: basic offset arithmetic ──────────────────────────────────────

class TestScrollByArithmetic(unittest.TestCase):
    def test_scroll_up_sets_offset(self):
        t = _make_tui()
        _load(t, 50)
        t._scroll_by(5)
        self.assertEqual(t._scroll_offset, 5)

    def test_scroll_up_snapshots_total(self):
        t = _make_tui()
        _load(t, 50)
        t._scroll_by(5)
        self.assertEqual(t._scroll_total, 50)

    def test_scroll_up_multiple_times_accumulates(self):
        t = _make_tui()
        _load(t, 50)
        t._scroll_by(5)
        t._scroll_by(5)
        self.assertEqual(t._scroll_offset, 10)

    def test_scroll_down_decrements_offset(self):
        t = _make_tui()
        _load(t, 50)
        t._scroll_by(10)
        t._scroll_by(-4)
        self.assertEqual(t._scroll_offset, 6)

    def test_scroll_down_to_zero_clears_anchor(self):
        t = _make_tui()
        _load(t, 50)
        t._scroll_by(10)
        t._scroll_by(-10)
        self.assertEqual(t._scroll_offset, 0)
        self.assertEqual(t._scroll_total, 0)

    def test_scroll_up_clamped_at_max(self):
        """Cannot scroll past the first content line."""
        t = _make_tui(h=20)   # model_h = 20 - 3 - 2 = 15, content_h = 14
        _load(t, 10)           # fewer lines than window → max_offset = 0
        t._scroll_by(100)
        self.assertEqual(t._scroll_offset, 0)

    def test_scroll_down_clamped_at_zero(self):
        t = _make_tui()
        _load(t, 50)
        t._scroll_by(5)
        t._scroll_by(-999)
        self.assertEqual(t._scroll_offset, 0)

    def test_scroll_by_zero_when_not_scrolled_does_nothing(self):
        t = _make_tui()
        _load(t, 50)
        t._scroll_by(0)
        self.assertEqual(t._scroll_offset, 0)
        self.assertEqual(t._scroll_total, 0)


# ── Anchor: new content does not shift the frozen view ───────────────────────

class TestScrollAnchor(unittest.TestCase):
    def test_anchor_snapshotted_on_first_scroll_up(self):
        t = _make_tui()
        _load(t, 40)
        t._scroll_by(5)
        # Anchor is 40; add 10 more lines
        _load(t, 10)
        # scroll_total stays at 40 even though len(model_lines) is now 50
        self.assertEqual(t._scroll_total, 40)

    def test_new_content_does_not_change_offset(self):
        t = _make_tui()
        _load(t, 40)
        t._scroll_by(5)
        offset_before = t._scroll_offset
        _load(t, 10)
        self.assertEqual(t._scroll_offset, offset_before)

    def test_second_scroll_up_does_not_re_snapshot(self):
        """The anchor must NOT be updated on subsequent scroll-up events."""
        t = _make_tui()
        _load(t, 40)
        t._scroll_by(3)           # anchor = 40
        _load(t, 10)              # model_lines now = 50
        t._scroll_by(3)           # should NOT re-anchor at 50
        self.assertEqual(t._scroll_total, 40)

    def test_new_below_count_reflects_added_lines(self):
        """new_below in the frozen view = lines added since anchoring."""
        t = _make_tui(h=40)
        _load(t, 40)
        t._scroll_by(5)
        _load(t, 7)   # 7 new lines arrive
        border = _border_text(t)
        self.assertIn("7 new", border)

    def test_no_new_badge_when_no_new_content(self):
        t = _make_tui()
        _load(t, 50)
        t._scroll_by(5)
        border = _border_text(t)
        self.assertNotIn("new", border)


# ── Scroll indicator ──────────────────────────────────────────────────────────

class TestScrollIndicator(unittest.TestCase):
    def test_up_arrow_in_border_when_scrolled(self):
        t = _make_tui()
        _load(t, 50)
        t._scroll_by(5)
        border = _border_text(t)
        self.assertIn("↑", border)

    def test_offset_shown_in_indicator(self):
        t = _make_tui()
        _load(t, 50)
        t._scroll_by(7)
        border = _border_text(t)
        self.assertIn("7", border)

    def test_no_indicator_after_returning_to_bottom(self):
        t = _make_tui()
        _load(t, 50)
        t._scroll_by(5)
        t._scroll_to_bottom()
        border = _border_text(t)
        self.assertNotIn("↑", border)

    def test_down_arrow_in_indicator_when_new_content(self):
        t = _make_tui(h=40)
        _load(t, 40)
        t._scroll_by(5)
        _load(t, 3)
        border = _border_text(t)
        self.assertIn("↓", border)


# ── Live vs frozen view content ───────────────────────────────────────────────

class TestLiveVsFrozenView(unittest.TestCase):
    def test_live_view_shows_newest_lines(self):
        """At offset 0 the tail of content is always visible."""
        t = _make_tui(h=20)
        # model_h = 20-3-2 = 15, content_h = 14
        for i in range(20):
            t._model_lines.append((f"line {i}", "normal"))
        texts = _visible_texts(t)
        # Must show the latest lines (6–19)
        self.assertTrue(any("line 19" in s for s in texts))
        self.assertFalse(any("line 0" in s for s in texts))

    def test_scrolled_view_does_not_show_newest_lines(self):
        """When scrolled, lines added after anchoring are hidden below."""
        t = _make_tui(h=20)
        for i in range(20):
            t._model_lines.append((f"old {i}", "normal"))
        t._scroll_by(5)
        # Add new lines that should NOT appear in the frozen view
        for i in range(5):
            t._model_lines.append((f"new {i}", "normal"))
        texts = _visible_texts(t)
        self.assertFalse(any("new " in s for s in texts),
            f"New lines should be hidden while scrolled; got: {texts}")

    def test_live_view_updates_when_new_content_arrives(self):
        """At offset 0, subsequent content is auto-scrolled into view."""
        t = _make_tui(h=20)
        for i in range(10):
            t._model_lines.append((f"line {i}", "normal"))
        texts_before = _visible_texts(t)
        t._model_lines.append(("the newest line", "normal"))
        texts_after = _visible_texts(t)
        self.assertFalse(any("the newest line" in s for s in texts_before))
        self.assertTrue(any("the newest line" in s for s in texts_after))


# ── _scroll_to_bottom ─────────────────────────────────────────────────────────

class TestScrollToBottom(unittest.TestCase):
    def test_resets_offset_to_zero(self):
        t = _make_tui()
        _load(t, 50)
        t._scroll_by(10)
        t._scroll_to_bottom()
        self.assertEqual(t._scroll_offset, 0)

    def test_resets_total_to_zero(self):
        t = _make_tui()
        _load(t, 50)
        t._scroll_by(10)
        t._scroll_to_bottom()
        self.assertEqual(t._scroll_total, 0)

    def test_live_view_restored_after_snap(self):
        t = _make_tui(h=20)
        for i in range(20):
            t._model_lines.append((f"old {i}", "normal"))
        t._scroll_by(5)
        for i in range(5):
            t._model_lines.append((f"latest {i}", "normal"))
        t._scroll_to_bottom()
        texts = _visible_texts(t)
        self.assertTrue(any("latest " in s for s in texts))


# ── _page_up / _page_down ─────────────────────────────────────────────────────

class TestPageScrolling(unittest.TestCase):
    def test_page_up_scrolls_by_model_h_minus_2(self):
        t = _make_tui(h=20)
        _load(t, 80)
        page_size = max(1, t._model_h - 2)
        t._page_up()
        self.assertEqual(t._scroll_offset, page_size)

    def test_page_down_decrements_offset_by_page_size(self):
        t = _make_tui(h=20)
        _load(t, 80)
        page_size = max(1, t._model_h - 2)
        t._page_up()
        t._page_up()
        t._page_down()
        self.assertEqual(t._scroll_offset, page_size)

    def test_page_up_cannot_scroll_past_top(self):
        t = _make_tui(h=20)
        _load(t, 5)   # very few lines
        t._page_up()
        t._page_up()
        # Should be clamped to max possible, not negative
        self.assertGreaterEqual(t._scroll_offset, 0)

    def test_page_down_to_bottom_returns_to_live(self):
        t = _make_tui(h=20)
        _load(t, 80)
        t._page_up()
        t._page_down()
        self.assertEqual(t._scroll_offset, 0)


# ── Resize resets scroll state ────────────────────────────────────────────────

class TestResizeResetsScroll(unittest.TestCase):
    def _simulate_resize(self, t: tui_mod.TUI, new_h: int, new_w: int):
        """Simulate KEY_RESIZE by calling _layout(is_resize=True) with
        ioctl patched to return the new size."""
        import struct, fcntl, termios
        packed = struct.pack('HHHH', new_h, new_w, 0, 0)
        with mock.patch('fcntl.ioctl', return_value=packed):
            with mock.patch.object(tui_mod._curses, 'resizeterm'):
                t._layout(is_resize=True)

    def test_resize_resets_scroll_offset(self):
        t = _make_tui()
        _load(t, 50)
        t._scroll_by(10)
        self.assertEqual(t._scroll_offset, 10)
        self._simulate_resize(t, 30, 80)
        self.assertEqual(t._scroll_offset, 0)

    def test_resize_resets_scroll_total(self):
        t = _make_tui()
        _load(t, 50)
        t._scroll_by(10)
        self._simulate_resize(t, 30, 80)
        self.assertEqual(t._scroll_total, 0)


# ── Input loop routing ────────────────────────────────────────────────────────

class TestInputLoopScrollRouting(unittest.TestCase):
    """Verify that KEY_PPAGE, KEY_NPAGE, and KEY_MOUSE are dispatched to the
    correct scroll methods in run_input_loop, *before* the input_active gate.

    Strategy: the draw_q is kept empty until the patched scroll method fires
    (its side_effect posts the quit); this ensures we observe the key *before*
    the loop exits.
    """

    def _run_one_key(self, t: tui_mod.TUI, key: int,
                     input_active: bool = False,
                     mouse_bstate: int = 0):
        """Drive run_input_loop for exactly one key event then a quit.

        Returns (mock_page_up, mock_page_down, mock_scroll_by).
        """
        import queue as _q

        t._input_active = input_active

        key_iter = iter([key])

        def fake_getch():
            try:
                return next(key_iter)
            except StopIteration:
                return -1   # keeps the loop spinning until quit is posted

        t._input_win.getch.side_effect = fake_getch

        if key == _curses_mock.KEY_MOUSE:
            _curses_mock.getmouse.return_value = (0, 0, 0, 0, mouse_bstate)

        # The draw_q starts empty.  We inject the quit message only *after*
        # one of the scroll methods is called.
        quit_posted = [False]

        def post_quit(*_, **__):
            quit_posted[0] = True

        def patched_get():
            if quit_posted[0]:
                quit_posted[0] = False
                return {"kind": "quit"}
            raise _q.Empty

        with mock.patch.object(t._draw_q, 'get_nowait', side_effect=patched_get):
            with mock.patch.object(t, '_page_up',   side_effect=post_quit) as mock_pu, \
                 mock.patch.object(t, '_page_down', side_effect=post_quit) as mock_pd, \
                 mock.patch.object(t, '_scroll_by', side_effect=post_quit) as mock_sb, \
                 mock.patch.object(t, '_redraw_all'), \
                 mock.patch.object(t, '_place_cursor'), \
                 mock.patch.object(tui_mod._curses, 'doupdate'):
                try:
                    t.run_input_loop()
                except Exception:
                    pass
                return mock_pu, mock_pd, mock_sb

    def test_key_ppage_calls_page_up_when_input_active(self):
        t = _make_tui()
        _load(t, 50)
        pu, _, _ = self._run_one_key(t, _curses_mock.KEY_PPAGE, input_active=True)
        pu.assert_called_once()

    def test_key_ppage_calls_page_up_when_input_inactive(self):
        """Scroll must work even while the model is generating (input locked)."""
        t = _make_tui()
        _load(t, 50)
        pu, _, _ = self._run_one_key(t, _curses_mock.KEY_PPAGE, input_active=False)
        pu.assert_called_once()

    def test_key_npage_calls_page_down(self):
        t = _make_tui()
        _load(t, 50)
        _, pd, _ = self._run_one_key(t, _curses_mock.KEY_NPAGE)
        pd.assert_called_once()

    def test_mouse_wheel_up_calls_scroll_by_positive(self):
        t = _make_tui()
        _load(t, 50)
        _, _, sb = self._run_one_key(
            t, _curses_mock.KEY_MOUSE,
            mouse_bstate=_curses_mock.BUTTON4_PRESSED)
        sb.assert_called_once_with(3)

    def test_mouse_wheel_down_calls_scroll_by_negative(self):
        t = _make_tui()
        _load(t, 50)
        _, _, sb = self._run_one_key(
            t, _curses_mock.KEY_MOUSE,
            mouse_bstate=_curses_mock.BUTTON5_PRESSED)
        sb.assert_called_once_with(-3)

    def test_mouse_wheel_up_works_when_input_inactive(self):
        t = _make_tui()
        _load(t, 50)
        _, _, sb = self._run_one_key(
            t, _curses_mock.KEY_MOUSE,
            input_active=False,
            mouse_bstate=_curses_mock.BUTTON4_PRESSED)
        sb.assert_called_once_with(3)


# ── Enter snaps to bottom ─────────────────────────────────────────────────────

class TestEnterSnapsToBottom(unittest.TestCase):
    """Submitting a message via Enter should snap to the live view so the
    response is immediately visible."""

    def test_enter_resets_scroll_offset(self):
        t = _make_tui()
        _load(t, 50)
        t._scroll_by(10)
        self.assertEqual(t._scroll_offset, 10)

        import queue as _q
        import asyncio

        loop = asyncio.new_event_loop()
        pq   = asyncio.Queue()
        cq   = asyncio.Queue()
        t.set_async(loop, pq, cq)
        t._input_active     = True
        t._input_buf        = "hello"
        t._input_cursor     = 5
        t._input_confirming = False

        keys_sent = [False]

        def fake_getch():
            if not keys_sent[0]:
                keys_sent[0] = True
                return 10   # Enter
            return -1

        t._input_win.getch.side_effect = fake_getch

        # Return quit once Enter has been handled (offset reset to 0).
        quit_posted = [False]

        def patched_get():
            if keys_sent[0] and t._scroll_offset == 0:
                quit_posted[0] = True
            if quit_posted[0]:
                quit_posted[0] = False
                return {"kind": "quit"}
            raise _q.Empty

        with mock.patch.object(t._draw_q, 'get_nowait', side_effect=patched_get):
            with mock.patch.object(t, '_draw_model'), \
                 mock.patch.object(t, '_draw_input'), \
                 mock.patch.object(t, '_place_cursor'), \
                 mock.patch.object(tui_mod._curses, 'doupdate'), \
                 mock.patch.object(t, '_redraw_all'):
                try:
                    t.run_input_loop()
                except Exception:
                    pass

        loop.close()
        self.assertEqual(t._scroll_offset, 0,
            "Enter should have snapped scroll_offset back to 0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
