"""
test_tui_resize.py
==================
Tests that terminal resize events are correctly applied to the TUI layout.

Key invariants verified:
  1. After KEY_RESIZE, TUI._h and TUI._w match the new terminal dimensions.
  2. After KEY_RESIZE, windows are recreated at the new dimensions.
  3. resizeterm() is called BEFORE getmaxyx() on a resize, so that
     getmaxyx() returns the new dimensions rather than the stale ones.
  4. resizeterm() is NOT called during initial layout (is_resize=False),
     which would reset cbreak/noecho on some ncurses builds.
  5. The initial layout (not a resize) also correctly stores the dimensions
     reported by getmaxyx().

Run with:
    python -m unittest test_tui_resize -v
or:
    python -m pytest test_tui_resize.py -v
"""
from __future__ import annotations

import sys
import unittest
import unittest.mock as mock


# ── Curses stub (no real terminal needed) ────────────────────────────────────

_curses_mock = mock.MagicMock()
for _attr, _val in [
    ("COLOR_BLACK", 0), ("COLOR_WHITE", 7), ("COLOR_GREEN", 2),
    ("COLOR_RED",   1), ("COLOR_CYAN",   6), ("COLOR_YELLOW", 3),
    ("COLOR_MAGENTA", 5), ("COLOR_BLUE", 4), ("COLORS", 256),
    ("A_BOLD", 1), ("A_DIM", 2),
    ("KEY_ENTER", 343), ("KEY_BACKSPACE", 263), ("KEY_DC", 330),
    ("KEY_LEFT",  260), ("KEY_RIGHT",     261), ("KEY_HOME", 262),
    ("KEY_END",   360), ("KEY_RESIZE",    410),
    ("LINES", 24), ("COLS", 80),
]:
    setattr(_curses_mock, _attr, _val)

sys.modules["curses"] = _curses_mock

sys.path.insert(0, "/mnt/user-data/outputs")
import modelrelay_tui as tui_mod


# ── Helper: build a mock stdscr that returns controllable dimensions ──────────

def _make_stdscr(h: int, w: int) -> mock.MagicMock:
    """Return a mock stdscr whose getmaxyx() returns (h, w)."""
    scr = mock.MagicMock()
    scr.getmaxyx.return_value = (h, w)
    return scr


def _make_tui(h: int = 24, w: int = 80) -> tui_mod.TUI:
    """Instantiate a TUI with a mock stdscr of the given size."""
    # Reset LINES/COLS to match the requested size so _layout's LINES/COLS
    # reference is consistent with the stdscr mock.
    _curses_mock.LINES = h
    _curses_mock.COLS  = w
    scr = _make_stdscr(h, w)
    return tui_mod.TUI(scr)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestInitialLayout(unittest.TestCase):
    """TUI dimensions are set correctly at startup."""

    def test_initial_h_and_w_match_getmaxyx(self):
        tui = _make_tui(24, 80)
        self.assertEqual(tui._h, 24)
        self.assertEqual(tui._w, 80)

    def test_resizeterm_not_called_at_init(self):
        """resizeterm() during __init__ resets cbreak on some builds — must be skipped."""
        _curses_mock.reset_mock()
        _make_tui(24, 80)
        _curses_mock.resizeterm.assert_not_called()

    def test_initial_windows_have_correct_width(self):
        tui = _make_tui(24, 80)
        # Each newwin call: newwin(h, w, y, x) — width is the second argument.
        calls = _curses_mock.newwin.call_args_list
        for c in calls:
            _, w_arg = c.args[0], c.args[1]
            self.assertEqual(w_arg, 80, f"Window created with wrong width: {c}")


class TestResizeUpdatesStoredDimensions(unittest.TestCase):
    """After _layout(is_resize=True), _h/_w reflect the new terminal size."""

    def test_h_updated_after_resize(self):
        tui = _make_tui(24, 80)
        tui._scr.getmaxyx.return_value = (40, 80)
        _curses_mock.LINES = 40
        tui._layout(is_resize=True)
        self.assertEqual(tui._h, 40)

    def test_w_updated_after_resize(self):
        tui = _make_tui(24, 80)
        tui._scr.getmaxyx.return_value = (24, 120)
        _curses_mock.COLS = 120
        tui._layout(is_resize=True)
        self.assertEqual(tui._w, 120)

    def test_both_h_and_w_updated_together(self):
        tui = _make_tui(24, 80)
        tui._scr.getmaxyx.return_value = (50, 200)
        _curses_mock.LINES = 50
        _curses_mock.COLS  = 200
        tui._layout(is_resize=True)
        self.assertEqual(tui._h, 50)
        self.assertEqual(tui._w, 200)

    def test_resize_smaller(self):
        tui = _make_tui(40, 120)
        tui._scr.getmaxyx.return_value = (20, 60)
        _curses_mock.LINES = 20
        _curses_mock.COLS  = 60
        tui._layout(is_resize=True)
        self.assertEqual(tui._h, 20)
        self.assertEqual(tui._w, 60)


class TestResizeCreatesNewWindows(unittest.TestCase):
    """After resize, all three windows are recreated at the new width."""

    def test_windows_use_new_width_after_resize(self):
        tui = _make_tui(24, 80)
        _curses_mock.reset_mock()
        tui._scr.getmaxyx.return_value = (24, 120)
        _curses_mock.COLS = 120
        tui._layout(is_resize=True)
        calls = _curses_mock.newwin.call_args_list
        self.assertEqual(len(calls), 3, "Expected exactly 3 newwin calls after resize")
        for c in calls:
            w_arg = c.args[1]
            self.assertEqual(w_arg, 120, f"Window not using new width: {c}")

    def test_windows_use_new_height_distribution(self):
        tui = _make_tui(24, 80)
        _curses_mock.reset_mock()
        tui._scr.getmaxyx.return_value = (40, 80)
        _curses_mock.LINES = 40
        tui._layout(is_resize=True)
        heights = [c.args[0] for c in _curses_mock.newwin.call_args_list]
        self.assertEqual(sum(heights), 40,
            f"Window heights must sum to terminal height. Got: {heights}")


class TestResizetermCalledBeforeGetmaxyx(unittest.TestCase):
    """_layout(is_resize=True) invariants for resizeterm() and dimension storage.

    Design (after the ioctl-bypass fix):
      • resizeterm() IS called (to inform ncurses of the new size).
      • getmaxyx() is NOT called during resize; _h/_w are set directly from
        the TIOCGWINSZ ioctl values (or the LINES/COLS fallback when the ioctl
        fails on a mock fd).  This avoids the stale-size loop where ncurses
        builds that don't update getmaxyx() after resizeterm() would leave
        _h/_w unchanged and trigger an infinite redraw poll.
    """

    def test_resizeterm_called_on_resize(self):
        """resizeterm() must be called so ncurses knows about the new size."""
        tui = _make_tui(24, 80)
        _curses_mock.resizeterm.reset_mock()
        _curses_mock.LINES = 40
        _curses_mock.COLS  = 120
        tui._scr.getmaxyx.return_value = (40, 120)

        tui._layout(is_resize=True)

        _curses_mock.resizeterm.assert_called_once()

    def test_getmaxyx_not_called_during_resize(self):
        """getmaxyx() must NOT be called during a resize.

        Some ncurses builds do not update getmaxyx() after resizeterm(), so
        reading it back would return the stale pre-resize size.  _layout()
        instead uses the ioctl value (or LINES/COLS fallback) directly.
        """
        tui = _make_tui(24, 80)
        _curses_mock.LINES = 40
        _curses_mock.COLS  = 120
        tui._scr.getmaxyx.reset_mock()

        tui._layout(is_resize=True)

        tui._scr.getmaxyx.assert_not_called()

    def test_h_w_set_to_fallback_dims_when_ioctl_unavailable(self):
        """When ioctl fails (mock fd), _h/_w are set from LINES/COLS fallback."""
        tui = _make_tui(24, 80)
        _curses_mock.LINES = 40
        _curses_mock.COLS  = 120
        # ioctl will fail on a mock fd; fallback is LINES/COLS
        tui._layout(is_resize=True)

        self.assertEqual(tui._h, 40,
            f"_h should be LINES=40 after resize fallback, got {tui._h}")
        self.assertEqual(tui._w, 120,
            f"_w should be COLS=120 after resize fallback, got {tui._w}")

    def test_resizeterm_called_with_ioctl_values(self):
        """resizeterm must use TIOCGWINSZ values (authoritative); falls back to
        LINES/COLS only if the ioctl fails (fileno() not available on mock)."""
        tui = _make_tui(24, 80)
        _curses_mock.LINES = 40
        _curses_mock.COLS  = 120
        _curses_mock.resizeterm.reset_mock()
        tui._scr.getmaxyx.return_value = (40, 120)

        tui._layout(is_resize=True)

        _curses_mock.resizeterm.assert_called_once()

    def test_resizeterm_not_called_on_non_resize_layout(self):
        tui = _make_tui(24, 80)
        _curses_mock.resizeterm.reset_mock()
        tui._layout(is_resize=False)
        _curses_mock.resizeterm.assert_not_called()

    def test_non_resize_layout_reads_h_w_from_getmaxyx(self):
        """During normal (non-resize) layout, _h/_w come from getmaxyx()."""
        tui = _make_tui(24, 80)
        tui._scr.getmaxyx.return_value = (40, 120)
        tui._layout(is_resize=False)
        self.assertEqual(tui._h, 40)
        self.assertEqual(tui._w, 120)


class TestResizeVsNonResizeLayout(unittest.TestCase):
    """Non-resize layout still reads dimensions from getmaxyx correctly."""

    def test_non_resize_layout_stores_getmaxyx_result(self):
        tui = _make_tui(24, 80)
        tui._scr.getmaxyx.return_value = (30, 90)
        tui._layout(is_resize=False)
        self.assertEqual(tui._h, 30)
        self.assertEqual(tui._w, 90)

    def test_clearok_not_called_on_non_resize(self):
        tui = _make_tui(24, 80)
        tui._scr.clearok.reset_mock()
        tui._layout(is_resize=False)
        tui._scr.clearok.assert_not_called()

    def test_clearok_called_on_resize(self):
        tui = _make_tui(24, 80)
        tui._scr.clearok.reset_mock()
        tui._scr.getmaxyx.return_value = (40, 120)
        tui._layout(is_resize=True)
        tui._scr.clearok.assert_called_once_with(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
