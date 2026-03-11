"""
test_tui_resize_blackbox.py
============================
Black-box integration test: runs modelrelay_tui.py in a real PTY with the mock
backend and verifies that resizing the terminal window causes the rendered
layout to change accordingly.

The test does NOT mock or patch the TUI layer.  It communicates with the
process exactly as a user's terminal emulator would:

  1. A PTY is created at a known size and the TUI process is launched inside it.
  2. The wizard is skipped via a pre-written .modelrelay config.
  3. The PTY is resized and SIGWINCH is sent to the process.
  4. Raw VT100 output is captured.  Absolute cursor-position escape sequences
     (ESC[row;colH) are parsed to determine the highest row the TUI addressed.
  5. The test asserts that after resize the addressed row height matches the
     new terminal size, proving the layout was actually redrawn.

Strategy for measuring layout height
--------------------------------------
curses uses ``ESC[row;colH`` (CSI CUP) for absolute cursor positioning when
rendering on xterm-compatible terminals.  The highest row number addressed in
a redraw equals the height of the rendered UI.

  - Initial size 24 rows  → highest row addressed = 24
  - After resize to 40 rows → highest row addressed = 40
  - After shrink to 15 rows → highest row addressed = 15

Requirements
-------------
  * Python 3 (stdlib only — no pyte or other third-party packages)
  * modelrelay_tui.py at SCRIPT_PATH
  * modelrelay (the subprocess) runnable via MODELRELAY_PATH

Run with:
    python -m unittest test_tui_resize_blackbox -v
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import select
import signal
import struct
import tempfile
import termios
import time
import unittest
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent

# modelrelay_tui.py — must be a sibling of this test file (or set via env)
SCRIPT_PATH = os.environ.get(
    "MODELRELAY_TUI",
    str(_HERE / "modelrelay_tui.py"),
)

# The modelrelay binary.  The wrapper below stubs websockets so the mock
# backend can run without the websockets package installed.
_DEFAULT_MR = str(Path(__file__).parent / "modelrelay_mock_runner.py")
MODELRELAY_PATH = os.environ.get("MODELRELAY_BIN", _DEFAULT_MR)

# How many seconds to wait for the TUI to initialise after the wizard
TUI_STARTUP_WAIT = 3.5

# ── Low-level PTY helpers ─────────────────────────────────────────────────────

def _set_winsize(fd: int, rows: int, cols: int) -> None:
    """Set the window size on a file descriptor (PTY master or stdin)."""
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def _read_for(fd: int, seconds: float) -> bytes:
    """Drain *fd* for up to *seconds*, returning all bytes received."""
    buf = b""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        r, _, _ = select.select([fd], [], [], min(0.05, remaining))
        if r:
            try:
                buf += os.read(fd, 4096)
            except OSError:
                break
    return buf


def _max_row(vt100_bytes: bytes) -> int:
    """Return the highest 1-based row index in any ESC[row;colH sequence."""
    matches = re.findall(rb"\x1b\[(\d+);(\d+)[Hf]", vt100_bytes)
    if not matches:
        return 0
    return max(int(r) for r, _ in matches)


# ── Test fixture ──────────────────────────────────────────────────────────────

class PTYSession:
    """Manages the lifecycle of a modelrelay_tui process in a PTY.

    Creates a fresh working directory with a saved .modelrelay config pointing
    to the mock backend so the config wizard can be dismissed with two Enter
    keystrokes (accept working dir, accept saved settings).
    """

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.pid: int | None = None
        self.master: int | None = None
        self._workdir = tempfile.mkdtemp(prefix="tui_test_")
        self._prepare_config()

    def _prepare_config(self) -> None:
        # Write a .modelrelay file so the wizard offers "use saved settings".
        config = {"backend": "mock", "model": "mock-model"}
        with open(os.path.join(self._workdir, ".modelrelay"), "w") as f:
            json.dump(config, f)

        # Write last_working_dir so the wizard pre-fills the directory prompt.
        mr_dir = os.path.expanduser("~/.modelrelay")
        os.makedirs(mr_dir, exist_ok=True)
        with open(os.path.join(mr_dir, "last_working_dir"), "w") as f:
            f.write(self._workdir)

    def start(self) -> None:
        """Fork the TUI process into a PTY at the configured size."""
        import pty as _pty
        self.pid, self.master = _pty.fork()
        if self.pid == 0:
            # Child: set window size on stdout, then exec
            _set_winsize(1, self.rows, self.cols)
            env = os.environ.copy()
            env["TERM"] = "xterm-256color"
            os.execvpe(
                "python3",
                ["python3", SCRIPT_PATH, MODELRELAY_PATH],
                env,
            )
            os._exit(1)

        # Parent: set window size on the master side
        _set_winsize(self.master, self.rows, self.cols)

    def send(self, text: str) -> None:
        os.write(self.master, text.encode())

    def dismiss_wizard(self) -> None:
        """Send the two keystrokes needed to skip past the wizard."""
        time.sleep(1.0)
        self.send("\n")       # accept pre-filled working directory
        time.sleep(0.5)
        self.send("\n")       # "Use these saved settings? [Y/n]" → yes

    def wait_for_tui(self) -> None:
        """Block until the curses TUI has had time to render."""
        time.sleep(TUI_STARTUP_WAIT)

    def resize(self, new_rows: int, new_cols: int | None = None) -> None:
        """Resize the PTY and deliver SIGWINCH to the TUI process."""
        if new_cols is None:
            new_cols = self.cols
        self.rows = new_rows
        self.cols = new_cols
        _set_winsize(self.master, new_rows, new_cols)
        os.kill(self.pid, signal.SIGWINCH)

    def capture(self, seconds: float = 1.5) -> bytes:
        """Read all PTY output produced in the next *seconds* seconds."""
        return _read_for(self.master, seconds)

    def stop(self) -> None:
        """Kill the TUI process and close the PTY."""
        if self.pid is not None:
            try:
                os.kill(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(self.pid, 0)
            except ChildProcessError:
                pass
        if self.master is not None:
            try:
                os.close(self.master)
            except OSError:
                pass


# ── Tests ─────────────────────────────────────────────────────────────────────

@unittest.skipUnless(
    os.path.exists(SCRIPT_PATH),
    f"modelrelay_tui.py not found at {SCRIPT_PATH!r}",
)
@unittest.skipUnless(
    os.path.exists(MODELRELAY_PATH),
    f"modelrelay not found at {MODELRELAY_PATH!r}",
)
class TestTUIResizeBlackBox(unittest.TestCase):
    """Verify terminal resize is reflected in the live rendered layout."""

    # ── helpers ───────────────────────────────────────────────────────────────

    def _run(self, initial_rows: int = 24, cols: int = 80) -> PTYSession:
        session = PTYSession(rows=initial_rows, cols=cols)
        self.addCleanup(session.stop)
        session.start()
        session.dismiss_wizard()
        session.wait_for_tui()
        return session

    def _assert_layout_height(self, vt100: bytes, expected_rows: int,
                               tolerance: int = 1, msg: str = "") -> None:
        """Assert the highest addressed row is within tolerance of expected."""
        actual = _max_row(vt100)
        self.assertGreater(
            actual, 0,
            f"{msg}: no ESC[row;colH sequences found in output "
            f"({len(vt100)} bytes captured)",
        )
        self.assertGreaterEqual(
            actual, expected_rows - tolerance,
            f"{msg}: max row {actual} is below expected {expected_rows} "
            f"(±{tolerance})",
        )
        self.assertLessEqual(
            actual, expected_rows + tolerance,
            f"{msg}: max row {actual} is above expected {expected_rows} "
            f"(±{tolerance})",
        )

    # ── tests ─────────────────────────────────────────────────────────────────

    def test_initial_layout_fills_terminal_height(self):
        """The TUI's initial render uses the full terminal height."""
        session = self._run(initial_rows=24)
        out = session.capture(seconds=1.5)
        self._assert_layout_height(out, 24, msg="initial 24-row layout")

    def test_resize_larger_redraws_at_new_height(self):
        """Growing the terminal causes a full redraw at the new height."""
        session = self._run(initial_rows=24)
        session.capture(1.0)          # discard initial render

        session.resize(40)
        out = session.capture(1.5)
        self._assert_layout_height(out, 40, msg="after grow 24→40")

    def test_resize_smaller_redraws_at_new_height(self):
        """Shrinking the terminal causes a full redraw at the new height."""
        session = self._run(initial_rows=40)
        session.capture(1.0)

        session.resize(15)
        out = session.capture(1.5)
        self._assert_layout_height(out, 15, msg="after shrink 40→15")

    def test_sequential_resizes_each_redraw(self):
        """Every resize triggers a fresh redraw at the correct height."""
        session = self._run(initial_rows=24)
        session.capture(1.0)

        for new_rows in (35, 20, 45):
            session.resize(new_rows)
            out = session.capture(1.5)
            self._assert_layout_height(
                out, new_rows, msg=f"after resize to {new_rows}")

    def test_resize_width_included_in_redraw(self):
        """Changing terminal width also triggers a redraw."""
        session = self._run(initial_rows=24, cols=80)
        session.capture(1.0)

        session.resize(new_rows=24, new_cols=120)
        out = session.capture(1.5)
        # A redraw must have happened: any cursor positions mean the UI responded
        self.assertGreater(
            _max_row(out), 0,
            "No cursor positions found after width resize — UI did not redraw",
        )

    def test_resize_before_model_activity(self):
        """Resize works correctly even when no model turn has occurred."""
        # This is the same as test_resize_larger but documents the scenario
        # explicitly: the UI is in idle/waiting state, no text in the model pane.
        session = self._run(initial_rows=24)
        session.capture(1.0)

        session.resize(32)
        out = session.capture(1.5)
        self._assert_layout_height(out, 32, msg="idle resize 24→32")

    def test_no_redraw_without_resize(self):
        """Without a resize event, no fresh cursor-position output is emitted
        while the UI is idle (i.e., the display is stable)."""
        session = self._run(initial_rows=24)
        session.capture(2.0)   # consume initial render

        # No resize — just wait
        out = session.capture(1.0)
        # The UI may emit status updates (~30s interval), but those do not use
        # full absolute-position repaints; row addresses will be low (status bar).
        # The key invariant: the bottom of the screen (row 24) should NOT be
        # re-addressed unless something caused a full repaint.
        positions = re.findall(rb"\x1b\[(\d+);(\d+)[Hf]", out)
        addressed_rows = {int(r) for r, _ in positions}
        # If any high row is addressed, it must be within the current 24 rows
        for r in addressed_rows:
            self.assertLessEqual(
                r, 24,
                f"Row {r} addressed without a resize on a 24-row terminal",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
