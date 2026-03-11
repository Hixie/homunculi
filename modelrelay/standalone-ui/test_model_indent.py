"""
test_model_indent.py
====================
Tests that indentation in model text output is preserved when the TUI renders
it in the model window.

The bug (now fixed): `_draw_model` called `line.strip()` before wrapping,
which silently discarded all leading whitespace.  A model response containing:

    Here are the steps:
      1. First do this
      2. Then do that
         - sub-item here

would be rendered as:

    Here are the steps:
    1. First do this
    2. Then do that
    - sub-item here

This suite verifies the fix holds by populating `TUI._model_lines` / `TUI._text_buf`
directly (bypassing the async session) and inspecting the text strings that
`_draw_model` passes to curses `addstr`.

Run with:
    python -m unittest test_model_indent -v
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
    ("KEY_ENTER", 343), ("KEY_BACKSPACE", 263), ("KEY_DC", 330),
    ("KEY_LEFT",  260), ("KEY_RIGHT",     261), ("KEY_HOME", 262),
    ("KEY_END",   360), ("KEY_RESIZE",    410),
    ("LINES", 24), ("COLS", 80),
]:
    setattr(_curses_mock, _attr, _val)

sys.modules["curses"] = _curses_mock

sys.path.insert(0, "/mnt/user-data/outputs")
import modelrelay_tui as tui_mod                        # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tui(h: int = 40, w: int = 80) -> tui_mod.TUI:
    """Construct a TUI with a mock stdscr of the given dimensions."""
    _curses_mock.LINES = h
    _curses_mock.COLS  = w
    scr = mock.MagicMock()
    scr.getmaxyx.return_value = (h, w)
    return tui_mod.TUI(scr)


def _rendered_lines(t: tui_mod.TUI) -> list[str]:
    """Call _draw_model and return every text string passed to addstr,
    in row order, with empty strings and the bottom-border rule filtered out.

    _safe_addstr delegates to win.addstr(row, col, text, attr).
    We collect the `text` argument (index 2) from every such call on
    the model window mock.
    """
    t._model_win.reset_mock()
    t._draw_model()
    texts = []
    for call in t._model_win.addstr.call_args_list:
        args = call.args
        if len(args) >= 3:
            text = args[2]
            if isinstance(text, str) and text.strip() and not set(text.strip()) <= {"─"}:
                texts.append(text)
    return texts


def _feed_text(t: tui_mod.TUI, text: str):
    """Populate a TUI's model_lines as though the model streamed `text`."""
    t._model_lines.append((text, "text"))


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestIndentPreservedInFlushedLines(unittest.TestCase):
    """Indentation in committed model_lines entries is kept on screen.

    This is the primary path: text has been flushed from _text_buf into
    _model_lines (e.g. after the model turn completes).
    """

    def test_two_space_indent_preserved(self):
        t = _make_tui()
        _feed_text(t, "  indented line")
        lines = _rendered_lines(t)
        self.assertTrue(
            any(l.startswith("  ") for l in lines),
            f"Expected a line starting with two spaces; got: {lines!r}")

    def test_four_space_indent_preserved(self):
        t = _make_tui()
        _feed_text(t, "    deeply indented")
        lines = _rendered_lines(t)
        self.assertTrue(
            any(l.startswith("    ") for l in lines),
            f"Expected a line starting with four spaces; got: {lines!r}")

    def test_unindented_line_has_no_leading_space(self):
        t = _make_tui()
        _feed_text(t, "no indent here")
        lines = _rendered_lines(t)
        self.assertTrue(
            any(l == "no indent here" for l in lines),
            f"Expected unindented line to be unchanged; got: {lines!r}")

    def test_mixed_indents_all_preserved(self):
        """A response with multiple indent depths preserves each one."""
        t = _make_tui()
        _feed_text(t, "top level\n  two spaces\n    four spaces\n      six spaces")
        lines = _rendered_lines(t)
        self.assertTrue(any(l == "top level"         for l in lines), lines)
        self.assertTrue(any(l.startswith("  ")       for l in lines), lines)
        self.assertTrue(any(l.startswith("    ")     for l in lines), lines)
        self.assertTrue(any(l.startswith("      ")   for l in lines), lines)

    def test_bullet_list_indent_preserved(self):
        """Common markdown bullet list indentation is kept."""
        t = _make_tui()
        _feed_text(t, "Steps:\n  - first\n  - second\n    - sub-item")
        lines = _rendered_lines(t)
        self.assertTrue(any(l == "Steps:"              for l in lines), lines)
        self.assertTrue(any(l == "  - first"           for l in lines), lines)
        self.assertTrue(any(l == "  - second"          for l in lines), lines)
        self.assertTrue(any(l == "    - sub-item"      for l in lines), lines)

    def test_numbered_list_indent_preserved(self):
        t = _make_tui()
        _feed_text(t, "  1. first item\n  2. second item")
        lines = _rendered_lines(t)
        self.assertTrue(any(l == "  1. first item"  for l in lines), lines)
        self.assertTrue(any(l == "  2. second item" for l in lines), lines)

    def test_trailing_whitespace_stripped(self):
        """Trailing spaces are removed; leading spaces are kept."""
        t = _make_tui()
        _feed_text(t, "  leading kept   ")
        lines = _rendered_lines(t)
        self.assertTrue(
            any(l.startswith("  ") and not l.endswith(" ") for l in lines),
            f"Expected leading spaces kept, trailing stripped; got: {lines!r}")


class TestIndentPreservedInStreamingBuffer(unittest.TestCase):
    """Indentation is preserved while text is still in _text_buf (mid-stream).

    The TUI appends deltas to _text_buf and calls _draw_model on every
    delta — so _draw_model must also handle _text_buf correctly.
    """

    def test_indented_text_in_buf_preserved(self):
        t = _make_tui()
        t._text_buf = "  streaming indent"
        lines = _rendered_lines(t)
        self.assertTrue(
            any(l.startswith("  ") for l in lines),
            f"Expected indented line in _text_buf to be preserved; got: {lines!r}")

    def test_multiline_buf_indents_preserved(self):
        t = _make_tui()
        t._text_buf = "header\n  bullet one\n  bullet two"
        lines = _rendered_lines(t)
        self.assertTrue(any(l == "header"       for l in lines), lines)
        self.assertTrue(any(l == "  bullet one" for l in lines), lines)
        self.assertTrue(any(l == "  bullet two" for l in lines), lines)


class TestLongIndentedLineWrapsCorrectly(unittest.TestCase):
    """An indented line that exceeds the terminal width wraps with its indent
    carried onto every continuation line — not reflowed back to column 0.
    """

    def test_continuation_lines_carry_indent(self):
        t = _make_tui(h=40, w=40)   # narrow window to force wrapping
        # 2-space indent + 60 chars of text — will need multiple wrap lines
        long_text = "  " + "word " * 12   # ~62 chars total
        _feed_text(t, long_text.rstrip())
        lines = _rendered_lines(t)
        # Every physical line produced must start with the 2-space indent
        self.assertTrue(len(lines) > 1,
            "Expected the long line to wrap into multiple lines")
        for line in lines:
            self.assertTrue(
                line.startswith("  "),
                f"Continuation line lost its indent: {line!r}  (all lines: {lines!r})")

    def test_deeper_indent_also_wraps_with_indent(self):
        t = _make_tui(h=40, w=40)
        long_text = "    " + "word " * 12
        _feed_text(t, long_text.rstrip())
        lines = _rendered_lines(t)
        self.assertTrue(len(lines) > 1, "Expected wrapping")
        for line in lines:
            self.assertTrue(
                line.startswith("    "),
                f"Continuation line lost its indent: {line!r}")


class TestRegressionStripBug(unittest.TestCase):
    """Regression: the old code called line.strip() which removed all leading
    whitespace.  These cases would have failed before the fix.
    """

    def test_single_indented_line_was_not_stripped(self):
        """The exact scenario that triggered the bug report."""
        t = _make_tui()
        _feed_text(t, "  - here is a bullet")
        lines = _rendered_lines(t)
        self.assertNotIn(
            "- here is a bullet", lines,
            "Stripped (bug) version found — indent was incorrectly removed")
        self.assertIn(
            "  - here is a bullet", lines,
            f"Indented version not found; rendered: {lines!r}")

    def test_code_block_style_indent_was_not_stripped(self):
        t = _make_tui()
        _feed_text(t, "    x = 1\n    y = 2")
        lines = _rendered_lines(t)
        # Neither stripped form should appear
        self.assertNotIn("x = 1", lines,
            "Stripped (bug) version found for first line")
        self.assertNotIn("y = 2", lines,
            "Stripped (bug) version found for second line")
        self.assertIn("    x = 1", lines,
            f"Indented version not found; rendered: {lines!r}")
        self.assertIn("    y = 2", lines,
            f"Indented version not found; rendered: {lines!r}")


class TestRealSessionLogReplay(unittest.TestCase):
    """Regression tests derived from the actual session log 20260311T064207.jsonl.

    The model response was:
        Sure! Here's a simple example of 5 lines, each more indented than the previous:

        ```
        Line 1
            Line 2
                Line 3
                    Line 4
                        Line 5
        ```

        Each line adds an extra level of indentation. ...

    The screenshot (2026-03-10-234315_1460x1122_scrot.png) shows all five
    "Line N" entries rendered flush-left with no indentation — the strip bug.
    These tests confirm the fixed code renders them correctly.

    Key insight from the log: indentation arrives as a *separate* delta token
    before the word "Line".  For example Line 2's indent is delivered as the
    token `"   "` followed by `" Line"` followed by `" "` followed by `"2"`.
    The TUI concatenates them in _text_buf to produce `"    Line 2"` (4 spaces)
    which is the correct input to _draw_model.  The old strip() call then threw
    those spaces away.
    """

    # The exact text produced by concatenating all TEXT_DELTA events in the log.
    _LOG_TEXT = (
        "Sure! Here's a simple example of 5 lines, each more indented than the previous:\n\n"
        "```\n"
        "Line 1\n"
        "    Line 2\n"
        "        Line 3\n"
        "            Line 4\n"
        "                Line 5\n"
        "```\n\n"
        "Each line adds an extra level of indentation. "
        "You can adjust the number of spaces for each indentation level as needed."
    )

    # The delta tokens as they actually arrived from the wire (seq 13–290 in the log,
    # TEXT_DELTA events only).  Feeding them one at a time tests the streaming path.
    _LOG_DELTAS = [
        "Sure", "!", " Here's", " a", " simple", " example", " of", " ", "5",
        " lines", ",", " each", " more", " ind", "ented", " than", " the", " previous",
        ":\n\n",
        "``", "`\n",
        "Line", " ", "1", "\n",
        "   ", " Line", " ", "2", "\n",
        "       ", " Line", " ", "3", "\n",
        "           ", " Line", " ", "4", "\n",
        "               ", " Line", " ", "5", "\n",
        "``", "`\n\n",
        "Each", " line", " adds", " an", " extra", " level", " of", " indentation",
        ".", " You", " can", " adjust", " the", " number", " of", " spaces", " for",
        " each", " indentation", " level", " as", " needed", ".",
    ]

    def _feed_full(self) -> tui_mod.TUI:
        """Feed the complete response as a single committed entry."""
        t = _make_tui()
        _feed_text(t, self._LOG_TEXT)
        return t

    def _feed_streaming(self) -> tui_mod.TUI:
        """Feed the response as individual deltas, simulating live streaming.

        Each delta goes into _text_buf; we call _draw_model after each one
        just as the real TUI does on every text_delta event.
        """
        t = _make_tui()
        for delta in self._LOG_DELTAS:
            t._text_buf += delta
            t._draw_model()   # as TUI._apply does on every text_delta
        # Flush final buffer into model_lines (as flush_line would do)
        if t._text_buf:
            t._model_lines.append((t._text_buf, "text"))
            t._text_buf = ""
        return t

    # ── helpers ──────────────────────────────────────────────────────────────

    def _assert_line_present(self, lines, expected, msg=""):
        self.assertIn(expected, lines,
            f"{msg}\nExpected {expected!r} in rendered lines.\nGot: {lines!r}")

    def _assert_line_absent(self, lines, bad, msg=""):
        self.assertNotIn(bad, lines,
            f"{msg}\nDid NOT expect {bad!r} in rendered lines.\nGot: {lines!r}")

    # ── flushed (committed) path ──────────────────────────────────────────────

    def test_flushed_line1_no_indent(self):
        lines = _rendered_lines(self._feed_full())
        self._assert_line_present(lines, "Line 1",
            "Line 1 should have no indent")

    def test_flushed_line2_four_spaces(self):
        lines = _rendered_lines(self._feed_full())
        self._assert_line_present(lines, "    Line 2",
            "Line 2 should have 4 spaces indent")
        self._assert_line_absent(lines, "Line 2",
            "Stripped Line 2 (bug) must not appear")

    def test_flushed_line3_eight_spaces(self):
        lines = _rendered_lines(self._feed_full())
        self._assert_line_present(lines, "        Line 3",
            "Line 3 should have 8 spaces indent")
        self._assert_line_absent(lines, "Line 3",
            "Stripped Line 3 (bug) must not appear")

    def test_flushed_line4_twelve_spaces(self):
        lines = _rendered_lines(self._feed_full())
        self._assert_line_present(lines, "            Line 4",
            "Line 4 should have 12 spaces indent")

    def test_flushed_line5_sixteen_spaces(self):
        lines = _rendered_lines(self._feed_full())
        self._assert_line_present(lines, "                Line 5",
            "Line 5 should have 16 spaces indent")

    def test_flushed_indent_strictly_increases(self):
        """Leading-space count must be 0 < 4 < 8 < 12 < 16 across Lines 1–5."""
        lines = _rendered_lines(self._feed_full())
        indents = {}
        for line in lines:
            for n in range(1, 6):
                if line.rstrip().endswith(f"Line {n}"):
                    indents[n] = len(line) - len(line.lstrip())
        self.assertEqual(sorted(indents.keys()), [1, 2, 3, 4, 5],
            f"Not all 5 lines found; indents map: {indents}")
        for n in range(1, 5):
            self.assertLess(indents[n], indents[n + 1],
                f"Indent of Line {n} ({indents[n]}) not < Line {n+1} ({indents[n+1]})")

    # ── streaming (mid-turn _text_buf) path ──────────────────────────────────

    def test_streaming_line2_four_spaces(self):
        """Indentation must be preserved in the streaming path too."""
        t = self._feed_streaming()
        lines = _rendered_lines(t)
        self._assert_line_present(lines, "    Line 2",
            "Streaming: Line 2 should have 4 spaces indent")

    def test_streaming_line5_sixteen_spaces(self):
        t = self._feed_streaming()
        lines = _rendered_lines(t)
        self._assert_line_present(lines, "                Line 5",
            "Streaming: Line 5 should have 16 spaces indent")

    def test_streaming_indent_strictly_increases(self):
        """Streamed token-by-token: indent still strictly increases across lines."""
        t = self._feed_streaming()
        lines = _rendered_lines(t)
        indents = {}
        for line in lines:
            for n in range(1, 6):
                if line.rstrip().endswith(f"Line {n}"):
                    indents[n] = len(line) - len(line.lstrip())
        self.assertEqual(sorted(indents.keys()), [1, 2, 3, 4, 5],
            f"Not all 5 lines found in streamed output; indents: {indents}")
        for n in range(1, 5):
            self.assertLess(indents[n], indents[n + 1],
                f"Streamed indent of Line {n} ({indents[n]}) not < Line {n+1} ({indents[n+1]})")

    def test_tokens_concatenate_to_correct_indent(self):
        """Unit-level: verify the raw delta tokens for Line 2 produce '    Line 2'
        after concatenation (the precondition for the rendering fix to matter)."""
        # From the log: seq 113='   ', seq 117=' Line', seq 121=' ', seq 125='2'
        fragment = "   " + " Line" + " " + "2"
        self.assertEqual(fragment, "    Line 2",
            "Delta concatenation should yield 4-space-indented line")


if __name__ == "__main__":
    unittest.main(verbosity=2)
