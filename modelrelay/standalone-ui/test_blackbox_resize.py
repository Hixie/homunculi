"""
test_blackbox_resize.py
=======================
Black-box integration tests for modelrelay_tui.  The process is started as a
real subprocess attached to a real PTY; no part of the TUI, curses, or
modelrelay is mocked.  All interaction happens through the PTY master fd,
exactly as a terminal emulator would.

Two test classes:

  TestBlackBoxResize  – PTY resize reaches the TUI and the rendered layout
                        adapts to the new column count.

  TestBlackBoxTyping  – The user can type at the prompt: characters echo,
                        backspace erases, Enter submits the message, and
                        the input is locked while the model is responding.

Prerequisites
-------------
- The mock backend requires no API key.
- A modelrelay wrapper is auto-generated at /tmp/modelrelay_test_wrapper.py
  the first time the tests run (stubs the websockets package so modelrelay
  works without it installed).  Set MODELRELAY_PATH in the environment to use
  a different executable.

Run with:
    python -m unittest test_blackbox_resize -v
or:
    python -m pytest test_blackbox_resize.py -v
"""
from __future__ import annotations

import fcntl
import json
import os
import pty
import re
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
import unittest

# ── Path constants ────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))
_TUI_PATH    = os.path.join(_HERE, "modelrelay_tui.py")

# Default wrapper: stubs the websockets package so modelrelay runs without it.
_DEFAULT_WRAPPER = "/tmp/modelrelay_test_wrapper.py"
_MODELRELAY_PATH = os.environ.get("MODELRELAY_PATH", _DEFAULT_WRAPPER)

_WRAPPER_SRC = """\
#!/usr/bin/env python3
\"\"\"Stubs websockets and runs modelrelay.server.__main__.\"\"\"
import sys, types
ws = types.ModuleType("websockets")
ws.connect = None
sys.modules["websockets"] = ws
import pathlib
_candidates = [
    pathlib.Path(__file__).parent.parent / "modelrelay_src3",
    pathlib.Path("/home/claude/modelrelay_src3"),
]
for _c in _candidates:
    if (_c / "modelrelay" / "server" / "__main__.py").exists():
        sys.path.insert(0, str(_c))
        break
from modelrelay.server.__main__ import main
import asyncio
asyncio.run(main())
"""

# ── PTY helpers ───────────────────────────────────────────────────────────────

def _set_winsize(fd: int, rows: int, cols: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def _strip_escapes(data: bytes) -> bytes:
    """Strip ANSI/VT escape sequences, leaving only printable ASCII.

    This is what a terminal emulator would actually render as visible
    characters.  Used to assert that typed text appears on screen.
    """
    # CSI sequences:  ESC [ <params> <letter>
    data = re.sub(rb"\x1b\[[^a-zA-Z]*[a-zA-Z]", b"", data)
    # Character-set designations:  ESC ( <char>  or  ESC ) <char>
    data = re.sub(rb"\x1b[()].", b"", data)
    # Any remaining ESC + one char
    data = re.sub(rb"\x1b.", b"", data)
    # Keep only printable ASCII (0x20 – 0x7e)
    return bytes(b for b in data if 0x20 <= b <= 0x7e)


def _max_horiz_rule_run(data: bytes) -> int:
    """Return the length of the longest consecutive run of UTF-8 '─' (U+2500).

    The TUI fills the full terminal width with this character for horizontal
    rules, so the maximum run length equals the column count of the most
    recent layout.
    """
    SEP = b"\xe2\x94\x80"
    max_run = cur = 0
    i = 0
    while i < len(data):
        if data[i : i + 3] == SEP:
            cur += 1
            max_run = max(max_run, cur)
            i += 3
        else:
            cur = 0
            i += 1
    return max_run


# ── Fixture ───────────────────────────────────────────────────────────────────

class _TUIFixture:
    """Manages the lifecycle of a modelrelay_tui subprocess on a PTY.

    Usage::

        with _TUIFixture(rows=24, cols=80) as fix:
            fix.drive_to_curses(working_dir)
            fix.write(b"hello")
            fix.resize(30, 120)
    """

    #: Seconds to wait for the TUI to re-render after a resize before reading.
    RESIZE_SETTLE_S = 1.5

    def __init__(self, rows: int = 24, cols: int = 80):
        self.rows = rows
        self.cols = cols
        self._proc:      subprocess.Popen | None = None
        self._master_fd: int = -1
        self._slave_fd:  int = -1
        self._buf_lock  = threading.Lock()
        self._buf:       list[bytes] = []

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "_TUIFixture":
        self._start()
        return self

    def __exit__(self, *_) -> None:
        self._stop()

    # ── public API ────────────────────────────────────────────────────────────

    def write(self, data: bytes) -> None:
        """Send raw bytes to the TUI's stdin via the PTY master fd."""
        os.write(self._master_fd, data)

    def resize(self, rows: int, cols: int) -> None:
        """Resize the PTY slave and notify the child process.

        TIOCSWINSZ on the slave updates what TIOCGWINSZ returns inside the
        child; the TUI's 50ms poll detects this without needing SIGWINCH, but
        we send it anyway for environments where it is supported.
        """
        _set_winsize(self._slave_fd, rows, cols)
        try:
            os.kill(self._proc.pid, signal.SIGWINCH)
        except OSError:
            pass
        self.rows = rows
        self.cols = cols

    def captured(self) -> bytes:
        """Return all PTY output accumulated since the last clear_captured()."""
        with self._buf_lock:
            return b"".join(self._buf)

    def clear_captured(self) -> None:
        with self._buf_lock:
            self._buf.clear()

    def wait_for(self, pattern: bytes, timeout: float = 10.0) -> bool:
        """Block until *pattern* appears in captured output or *timeout* expires."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if pattern in self.captured():
                return True
            time.sleep(0.05)
        return False

    def drive_to_curses(self, working_dir: str) -> None:
        """Drive the config wizard to completion and wait for the idle prompt.

        Sends the working directory, accepts any saved config, waits for the
        curses alternate screen, then waits 1.5 s for the mock backend to emit
        status.activity(idle) so the input prompt becomes active.
        """
        assert self.wait_for(b"Working directory", timeout=8), \
            "Timed out waiting for working directory prompt"
        self.write(f"{working_dir}\n".encode())

        assert self.wait_for(b"saved settings", timeout=8), \
            "Timed out waiting for saved-settings prompt"
        self.write(b"Y\n")

        # ESC[?1049h = enter alternate screen — curses is now running
        assert self.wait_for(b"\x1b[?1049h", timeout=12), \
            "Timed out waiting for curses to start"

        # Allow time for status.activity(idle) → input_active=True
        time.sleep(1.5)

    # ── internal ──────────────────────────────────────────────────────────────

    def _start(self) -> None:
        master_fd, slave_fd = pty.openpty()
        _set_winsize(slave_fd, self.rows, self.cols)
        self._master_fd = master_fd
        self._slave_fd  = slave_fd

        self._proc = subprocess.Popen(
            [sys.executable, _TUI_PATH, _MODELRELAY_PATH],
            stdin=slave_fd, stdout=slave_fd, stderr=slave_fd,
            env={**os.environ, "TERM": "xterm-256color"},
        )
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self) -> None:
        while True:
            try:
                chunk = os.read(self._master_fd, 4096)
                with self._buf_lock:
                    self._buf.append(chunk)
            except OSError:
                break

    def _stop(self) -> None:
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        for fd in (self._slave_fd, self._master_fd):
            try:
                os.close(fd)
            except OSError:
                pass


# ── Setup helpers ─────────────────────────────────────────────────────────────

def _ensure_wrapper() -> None:
    """Write the websockets-stub wrapper script if it does not yet exist."""
    if os.path.exists(_MODELRELAY_PATH):
        return
    with open(_MODELRELAY_PATH, "w") as f:
        f.write(_WRAPPER_SRC)
    os.chmod(_MODELRELAY_PATH, 0o755)


def _make_working_dir() -> str:
    """Return a temp dir pre-seeded with a mock-backend .modelrelay config."""
    d = tempfile.mkdtemp()
    cfg = {
        "backend":            "mock",
        "model":              "mock-model",
        "log_dir":            d,
        "usage_interval_s":   3600.0,
        "reconnect_attempts": 3,
    }
    with open(os.path.join(d, ".modelrelay"), "w") as f:
        json.dump(cfg, f)
    return d


# ── TestBlackBoxResize ────────────────────────────────────────────────────────

class TestBlackBoxResize(unittest.TestCase):
    """PTY resize reaches the live TUI and the rendered layout adapts."""

    INITIAL_ROWS = 24
    INITIAL_COLS = 80
    RESIZED_ROWS = 30
    RESIZED_COLS = 120

    @classmethod
    def setUpClass(cls):
        _ensure_wrapper()
        cls.working_dir = _make_working_dir()

    def test_initial_render_uses_initial_width(self):
        """The TUI draws rules spanning the initial column count."""
        with _TUIFixture(self.INITIAL_ROWS, self.INITIAL_COLS) as fix:
            fix.drive_to_curses(self.working_dir)
            run = _max_horiz_rule_run(fix.captured())
            self.assertGreater(run, 0,
                "No horizontal rule found in initial render")
            self.assertAlmostEqual(run, self.INITIAL_COLS, delta=2,
                msg=f"Initial rule run {run} ≠ COLS={self.INITIAL_COLS}")

    def test_resize_triggers_redraw(self):
        """A PTY resize causes the TUI to produce new output."""
        with _TUIFixture(self.INITIAL_ROWS, self.INITIAL_COLS) as fix:
            fix.drive_to_curses(self.working_dir)
            fix.clear_captured()
            fix.resize(self.RESIZED_ROWS, self.RESIZED_COLS)
            time.sleep(_TUIFixture.RESIZE_SETTLE_S)
            self.assertGreater(len(fix.captured()), 0,
                "No output after resize — TUI did not re-render")

    def test_resize_changes_rule_width(self):
        """After resize the horizontal rules match the new column count, not the old."""
        with _TUIFixture(self.INITIAL_ROWS, self.INITIAL_COLS) as fix:
            fix.drive_to_curses(self.working_dir)
            run_before = _max_horiz_rule_run(fix.captured())

            fix.clear_captured()
            fix.resize(self.RESIZED_ROWS, self.RESIZED_COLS)
            time.sleep(_TUIFixture.RESIZE_SETTLE_S)
            run_after = _max_horiz_rule_run(fix.captured())

            self.assertGreater(run_after, 0,
                "No horizontal rule found after resize")
            self.assertNotEqual(run_before, run_after,
                f"Rule width unchanged after resize "
                f"(before={run_before}, after={run_after})")
            self.assertAlmostEqual(run_after, self.RESIZED_COLS, delta=2,
                msg=f"After resize, rule run {run_after} ≠ new COLS={self.RESIZED_COLS}")

    def test_resize_smaller_also_redraws(self):
        """Shrinking the terminal produces a re-render at the smaller width."""
        small_rows, small_cols = 20, 60
        with _TUIFixture(self.INITIAL_ROWS, self.INITIAL_COLS) as fix:
            fix.drive_to_curses(self.working_dir)
            fix.clear_captured()
            fix.resize(small_rows, small_cols)
            time.sleep(_TUIFixture.RESIZE_SETTLE_S)
            run = _max_horiz_rule_run(fix.captured())
            self.assertAlmostEqual(run, small_cols, delta=2,
                msg=f"After shrink, rule run {run} ≠ new COLS={small_cols}")

    def test_multiple_resizes_each_redraws(self):
        """Each successive resize produces a correctly-sized re-render."""
        sizes = [(30, 100), (24, 80), (40, 160)]
        with _TUIFixture(self.INITIAL_ROWS, self.INITIAL_COLS) as fix:
            fix.drive_to_curses(self.working_dir)
            for rows, cols in sizes:
                fix.clear_captured()
                fix.resize(rows, cols)
                time.sleep(_TUIFixture.RESIZE_SETTLE_S)
                run = _max_horiz_rule_run(fix.captured())
                self.assertAlmostEqual(run, cols, delta=2,
                    msg=f"Resize → {cols} cols: rule run was {run}")


# ── TestBlackBoxTyping ────────────────────────────────────────────────────────

class TestBlackBoxTyping(unittest.TestCase):
    """The user can type at the prompt: characters echo, editing works, and
    Enter submits the message to the model."""

    ROWS = 24
    COLS = 80

    @classmethod
    def setUpClass(cls):
        _ensure_wrapper()
        cls.working_dir = _make_working_dir()

    def test_characters_echo_as_typed(self):
        """Each printable character typed at the prompt appears on screen
        immediately — before Enter is pressed.

        We type characters one at a time, clearing the capture buffer before
        each so we see only the output produced by that single keystroke.
        """
        with _TUIFixture(self.ROWS, self.COLS) as fix:
            fix.drive_to_curses(self.working_dir)

            for ch in b"hello":
                fix.clear_captured()
                fix.write(bytes([ch]))
                time.sleep(0.15)
                rendered = _strip_escapes(fix.captured())
                self.assertIn(bytes([ch]), rendered,
                    f"Typed {chr(ch)!r} did not appear in rendered output "
                    f"(got {rendered!r})")

    def test_backspace_erases_last_character(self):
        """Backspace emits the BS-SP-BS erase sequence and places the cursor
        back so the next keystroke overwrites the deleted cell.

        BS-SP-BS (0x08 0x20 0x08) is the canonical terminal erase pattern:
        move cursor left, write a space over the character, move left again.
        """
        with _TUIFixture(self.ROWS, self.COLS) as fix:
            fix.drive_to_curses(self.working_dir)
            fix.write(b"abc")
            time.sleep(0.2)

            # Capture only what the backspace keystroke produces
            fix.clear_captured()
            fix.write(b"\x7f")      # DEL — the key curses maps to KEY_BACKSPACE
            time.sleep(0.15)
            raw = fix.captured()

            self.assertIn(b"\x08 \x08", raw,
                f"Backspace did not produce the BS-SP-BS erase sequence; "
                f"raw PTY output was {raw!r}")

            # After erasing 'c', the cursor is back on that cell; the next
            # character typed should land there and appear normally.
            fix.clear_captured()
            fix.write(b"x")
            time.sleep(0.15)
            self.assertIn(b"x", _strip_escapes(fix.captured()),
                "Character typed after backspace did not appear in output")

    def test_enter_submits_message_to_model(self):
        """Pressing Enter submits the buffered input to the model backend.

        With the default empty mock script the backend immediately returns a
        fatal 'script exhausted' error.  Either the brief 'waiting for model'
        message or the error itself confirms the TUI processed the submission.
        """
        with _TUIFixture(self.ROWS, self.COLS) as fix:
            fix.drive_to_curses(self.working_dir)
            fix.write(b"hello model")
            time.sleep(0.2)

            fix.clear_captured()
            fix.write(b"\n")

            submitted = (
                fix.wait_for(b"waiting for model", timeout=3) or
                fix.wait_for(b"exhausted",          timeout=3)
            )
            self.assertTrue(submitted,
                "Enter did not trigger a model turn — the message was not submitted")

    def test_input_locked_while_model_responding(self):
        """Input is deactivated while the model is responding.

        After Enter the TUI immediately sets input_active=False and renders
        the "waiting for model…" label in the input box before the backend
        has a chance to reply.  We assert that this label appears — positive
        evidence that the lock engaged — rather than making a racy negative
        assertion about characters typed after a near-instant mock response.
        """
        with _TUIFixture(self.ROWS, self.COLS) as fix:
            fix.drive_to_curses(self.working_dir)

            # Type text and submit
            fix.write(b"go\n")

            # "waiting for model…" is rendered by _draw_input() only when
            # input_active is False.  Its presence confirms the lock fired.
            self.assertTrue(
                fix.wait_for(b"waiting for model", timeout=3),
                "Input box never showed 'waiting for model…' after Enter — "
                "the input lock did not engage"
            )


    def test_typing_survives_resize(self):
        """Input at the prompt works normally both before and after a PTY resize.

        This exercises the interaction between the two features: the resize
        redraws all three curses windows (including _input_win), and the TUI
        must correctly re-establish the input state so that subsequent
        keystrokes continue to echo.

        The test structure is:
          1. Type a character before the resize and confirm it echoes.
          2. Resize the terminal to a different size.
          3. Wait for the layout to settle.
          4. Type a different character and confirm it echoes in the new layout.
          5. Assert the new-layout rule width matches the resized column count,
             confirming the resize actually took effect and was not suppressed.
        """
        resized_rows, resized_cols = 30, 120

        with _TUIFixture(self.ROWS, self.COLS) as fix:
            fix.drive_to_curses(self.working_dir)

            # ── Step 1: typing works before resize ───────────────────────────
            fix.clear_captured()
            fix.write(b"a")
            time.sleep(0.15)
            pre_render = _strip_escapes(fix.captured())
            self.assertIn(b"a", pre_render,
                "Typing did not work before the resize — pre-condition failed")

            # ── Step 2-3: resize and wait for the new layout ──────────────────
            fix.clear_captured()
            fix.resize(resized_rows, resized_cols)
            time.sleep(_TUIFixture.RESIZE_SETTLE_S)

            # Confirm the resize actually produced a new layout at the right width
            run_after = _max_horiz_rule_run(fix.captured())
            self.assertAlmostEqual(run_after, resized_cols, delta=2,
                msg=f"Resize to {resized_cols} cols did not produce matching "
                    f"rules (got run={run_after}) — resize may not have fired")

            # ── Step 4-5: typing still works after resize ─────────────────────
            fix.clear_captured()
            fix.write(b"b")
            time.sleep(0.15)
            post_render = _strip_escapes(fix.captured())
            self.assertIn(b"b", post_render,
                f"Typing did not work after the resize — input box was not "
                f"correctly restored by the layout update "
                f"(stripped output: {post_render!r})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
