#!/usr/bin/env python3
"""
ModelRelay Terminal
An interactive host for the modelrelay subprocess implementing the full LOJP protocol.
"""
from __future__ import annotations

import asyncio
import curses as _curses
import difflib
import json
import os
import re
import sys
import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ─── ANSI Colour Palette ────────────────────────────────────────────────────

ESC = "\033["

def _c(*codes): return f"\033[{';'.join(str(c) for c in codes)}m"

RESET       = _c(0)
BOLD        = _c(1)
DIM         = _c(2)
ITALIC      = _c(3)
UNDERLINE   = _c(4)

BLACK       = _c(30)
RED         = _c(31)
GREEN       = _c(32)
YELLOW      = _c(33)
BLUE        = _c(34)
MAGENTA     = _c(35)
CYAN        = _c(36)
WHITE       = _c(37)

BR_BLACK    = _c(90)
BR_RED      = _c(91)
BR_GREEN    = _c(92)
BR_YELLOW   = _c(93)
BR_BLUE     = _c(94)
BR_MAGENTA  = _c(95)
BR_CYAN     = _c(96)
BR_WHITE    = _c(97)

BG_BLACK    = _c(40)
BG_RED      = _c(41)
BG_GREEN    = _c(42)
BG_YELLOW   = _c(43)
BG_BLUE     = _c(44)
BG_MAGENTA  = _c(45)
BG_CYAN     = _c(46)
BG_WHITE    = _c(47)
BG_BR_BLACK = _c(100)

CLEAR_LINE  = "\033[2K\r"
CLEAR_EOL   = "\033[K"
SAVE_CURSOR = "\033[s"
REST_CURSOR = "\033[u"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
MOVE_UP     = "\033[1A"


def _rl(prompt: str) -> str:
    """Wrap every ANSI escape sequence in a prompt string with readline's
    invisible-character delimiters (\\001 … \\002).

    Python's input() uses GNU readline to render the prompt.  Readline counts
    *every* byte in the prompt string toward the display width, so bare ANSI
    codes make it miscalculate where the typed text begins.  That causes text
    to wrap at the wrong column and to overwrite the prompt on the same line.

    Delimiters \\001 (RL_PROMPT_START_IGNORE) and \\002 (RL_PROMPT_END_IGNORE)
    tell readline that the bytes between them contribute zero display width.
    """
    return re.sub(r'(\033\[[0-9;]*[a-zA-Z])', r'\001\1\002', prompt)


def term_width() -> int:
    try:
        return os.get_terminal_size().columns
    except Exception:
        return 80


def hr(char="─", color=BR_BLACK) -> str:
    return f"{color}{char * term_width()}{RESET}"


def box_top(title: str = "", width: int = 0, color=CYAN) -> str:
    w = width or term_width()
    if title:
        pad = w - len(title) - 4
        left = pad // 2
        right = pad - left
        return f"{color}╭{'─' * left} {BOLD}{title}{RESET}{color} {'─' * right}╮{RESET}"
    return f"{color}╭{'─' * (w - 2)}╮{RESET}"


def box_bot(width: int = 0, color=CYAN) -> str:
    w = width or term_width()
    return f"{color}╰{'─' * (w - 2)}╯{RESET}"


def box_row(content: str, width: int = 0, color=CYAN, pad=1) -> str:
    w = width or term_width()
    # Strip ANSI for length calc
    visible = re.sub(r'\033\[[0-9;]*m', '', content)
    inner = w - 2 - pad * 2
    padded = " " * pad + content + " " * max(0, inner - len(visible)) + " " * pad
    return f"{color}│{RESET}{padded}{color}│{RESET}"


# ─── LOJP Protocol helpers ──────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def make_envelope(type_: str, payload: dict, id_: str | None = None) -> dict:
    return {"lojp": "1.0", "id": id_ or str(uuid.uuid4()),
            "ts": _now_iso(), "type": type_, "payload": payload}


def encode(env: dict) -> bytes:
    return (json.dumps(env, ensure_ascii=False) + "\n").encode("utf-8")


def decode(line: str) -> dict:
    return json.loads(line.strip())


# Factory functions for host → modelrelay messages
def cmd_prompt(text: str) -> dict:
    return make_envelope("cmd.prompt", {"text": text})


def cmd_invalidate(resource: str, reason: str = "User rejected change",
                   start_line: int | None = None) -> dict:
    p: dict = {"resource": resource, "reason": reason}
    if start_line is not None:
        p["start_line"] = start_line
    return make_envelope("cmd.invalidate", p)


def cmd_quit() -> dict:
    return make_envelope("cmd.quit", {})


def tool_content_response(id_: str, resource: str, total_lines: int,
                           regions: list, truncated: bool = False,
                           error: str | None = None) -> dict:
    return make_envelope("tool.content_response", {
        "resource": resource, "total_lines": total_lines,
        "regions": regions, "truncated": truncated, "error": error
    }, id_=id_)


def tool_stat_response(id_: str, resource: str, exists: bool,
                       total_lines: int | None = None,
                       last_modified: str | None = None,
                       error: str | None = None) -> dict:
    return make_envelope("tool.stat_response", {
        "resource": resource, "exists": exists,
        "total_lines": total_lines, "last_modified": last_modified, "error": error
    }, id_=id_)


def tool_replace_response(id_: str, error: str | None = None) -> dict:
    return make_envelope("tool.replace_response", {"error": error}, id_=id_)


def tool_insert_response(id_: str, error: str | None = None) -> dict:
    return make_envelope("tool.insert_response", {"error": error}, id_=id_)


# ─── File sandboxing ─────────────────────────────────────────────────────────

class Sandbox:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def resolve(self, resource: str) -> Path | None:
        """Resolve resource to an absolute path within sandbox. Returns None if outside."""
        try:
            p = (self.root / resource).resolve()
            p.relative_to(self.root)  # raises ValueError if outside
            return p
        except (ValueError, Exception):
            return None

    def relative(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except Exception:
            return str(p)


# ─── File operations ─────────────────────────────────────────────────────────

def read_file_lines(path: Path) -> list[str]:
    """Read file returning list of lines (with newlines)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.readlines()


def stat_file(sandbox: Sandbox, resource: str) -> tuple[bool, int | None, str | None, str | None]:
    """Returns (exists, total_lines, last_modified_iso, error)."""
    p = sandbox.resolve(resource)
    if p is None:
        return False, None, None, f"Access denied: {resource!r} is outside the sandbox"
    try:
        if not p.exists():
            return False, None, None, None
        lines = read_file_lines(p)
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        return True, len(lines), mtime.strftime("%Y-%m-%dT%H:%M:%SZ"), None
    except Exception as e:
        return False, None, None, str(e)


def read_file_ranges(sandbox: Sandbox, resource: str,
                     ranges: list[dict]) -> tuple[int, list[dict], str | None]:
    """Returns (total_lines, regions, error)."""
    p = sandbox.resolve(resource)
    if p is None:
        return 0, [], f"Access denied: {resource!r} is outside the sandbox"
    try:
        all_lines = read_file_lines(p)
        total = len(all_lines)
        regions = []
        for rng in ranges:
            start = rng["start_line"]  # 1-indexed
            num = rng["num_lines"]
            if num == "all":
                end = total
            else:
                end = min(start + num - 1, total)
            start = max(1, start)
            end = max(start, end)
            region_lines = all_lines[start - 1:end]
            regions.append({
                "start_line": start,
                "end_line": end,
                "lines": region_lines,
            })
        return total, regions, None
    except Exception as e:
        return 0, [], str(e)


def search_file(sandbox: Sandbox, resource: str, pattern: str,
                max_results: int = 20, context_lines: int = 2) -> tuple[int, list[dict], bool, str | None]:
    """Returns (total_lines, regions, truncated, error)."""
    p = sandbox.resolve(resource)
    if p is None:
        return 0, [], False, f"Access denied: {resource!r} is outside the sandbox"
    try:
        all_lines = read_file_lines(p)
        total = len(all_lines)
        # Find matching line indices (0-indexed)
        match_indices = [i for i, line in enumerate(all_lines) if pattern in line]
        truncated = len(match_indices) > max_results
        match_indices = match_indices[:max_results]

        # Merge overlapping context windows
        regions = []
        if match_indices:
            windows = []
            for idx in match_indices:
                s = max(0, idx - context_lines)
                e = min(total - 1, idx + context_lines)
                windows.append((s, e, idx))

            # Merge overlapping
            merged = [list(windows[0])]
            for s, e, m in windows[1:]:
                if s <= merged[-1][1] + 1:
                    merged[-1][1] = max(merged[-1][1], e)
                    merged[-1][2] = m  # update match line (keep latest)
                else:
                    merged.append([s, e, m])

            # We need all match lines per region
            for (ws, we, _) in merged:
                start_1 = ws + 1
                end_1 = we + 1
                region_lines = all_lines[ws:we + 1]
                # Find which absolute lines are matches in this region
                match_lines_in_region = [i + 1 for i in range(ws, we + 1)
                                         if pattern in all_lines[i]]
                regions.append({
                    "start_line": start_1,
                    "end_line": end_1,
                    "lines": region_lines,
                    "match_lines": match_lines_in_region,
                })

        return total, regions, truncated, None
    except Exception as e:
        return 0, [], False, str(e)


def apply_replace(path: Path, start_line: int, end_line: int, new_lines: list[str]) -> str | None:
    """Apply replacement. Returns error string or None on success."""
    try:
        old_lines = read_file_lines(path)
        result = old_lines[:start_line - 1] + new_lines + old_lines[end_line:]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(result)
        return None
    except Exception as e:
        return str(e)


def apply_insert(path: Path, after_line: int, new_lines: list[str]) -> str | None:
    """Apply insertion. Returns error string or None on success."""
    try:
        old_lines = read_file_lines(path)
        result = old_lines[:after_line] + new_lines + old_lines[after_line:]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(result)
        return None
    except Exception as e:
        return str(e)


def make_diff(old_lines: list[str], new_lines: list[str],
              fromfile: str = "before", tofile: str = "after") -> str:
    """Create a coloured unified diff."""
    diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=fromfile, tofile=tofile, lineterm=""
    ))
    if not diff:
        return f"{DIM}  (no changes){RESET}"
    parts = []
    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            parts.append(f"{BOLD}{BR_WHITE}{line}{RESET}")
        elif line.startswith("@@"):
            parts.append(f"{CYAN}{line}{RESET}")
        elif line.startswith("+"):
            parts.append(f"{BR_GREEN}{line}{RESET}")
        elif line.startswith("-"):
            parts.append(f"{BR_RED}{line}{RESET}")
        else:
            parts.append(f"{DIM}{line}{RESET}")
    return "\n".join(parts)


# ─── Configuration ───────────────────────────────────────────────────────────

BACKENDS = {
    "anthropic":       ("claude-opus-4-5", "ANTHROPIC_API_KEY"),
    "openai-realtime": ("gpt-4o-realtime-preview", "OPENAI_API_KEY"),
    "mock":            ("mock-model", None),
}


def fetch_models_with_pricing(
    modelrelay_path: Path, backend: str
) -> list[dict] | None:
    """Run `modelrelay --backend <b> --list-models` and return parsed models list.

    Each entry: {"id": str, "input_usd_per_1m": float, "output_usd_per_1m": float}
    Returns None on any failure (binary not found, timeout, bad JSON, API error).
    """
    import subprocess, json as _json
    try:
        result = subprocess.run(
            [str(modelrelay_path), "--backend", backend, "--list-models"],
            capture_output=True, text=True, timeout=10,
        )
        data = _json.loads(result.stdout)
        models = data.get("models", [])
        if not models:
            return None
        return models
    except Exception:
        return None


def _fmt_pricing(m: dict) -> str:
    """Format pricing for one model entry as a short inline string."""
    inp = m.get("input_usd_per_1m")
    out = m.get("output_usd_per_1m")
    if inp is None or out is None:
        return ""
    def _fmt(v):
        if v < 1:
            return f"${v:.2f}"
        return f"${v:.0f}"
    return f"{_fmt(inp)} in / {_fmt(out)} out per 1M tokens"


def prompt_model(backend: str, default_model: str,
                 modelrelay_path: Path | None) -> str:
    """Prompt for model selection.

    If modelrelay_path is provided, runs --list-models and shows a numbered
    menu with pricing.  Falls back to a plain prompt_text on any failure.
    """
    models: list[dict] | None = None
    if modelrelay_path is not None:
        print(f"  {DIM}Fetching available models…{RESET}", end="\r", flush=True)
        models = fetch_models_with_pricing(modelrelay_path, backend)
        print(" " * 35, end="\r")   # clear the "Fetching…" line

    if not models:
        # Fallback: plain text prompt (API key missing, network error, etc.)
        return prompt_text("\n  Model identifier", default=default_model)

    # Build numbered menu matching the Backend: style
    print(f"\n  {BOLD}{WHITE}Model:{RESET}")
    default_idx = 0
    for i, m in enumerate(models):
        mid = m["id"]
        pricing = _fmt_pricing(m)
        pricing_str = f"  {DIM}{pricing}{RESET}" if pricing else ""
        if mid == default_model:
            default_idx = i
        marker = f"{BR_CYAN}▶{RESET}" if mid == default_model else f"{BR_BLACK}○{RESET}"
        default_tag = f" {DIM}(default){RESET}" if mid == default_model else ""
        print(f"    {marker} {BOLD}{i+1}{RESET}. {mid}{default_tag}{pricing_str}")

    while True:
        try:
            raw = input(
                _rl(f"\n  {CYAN}Choice [{default_idx + 1}]{RESET}: ")
            ).strip()
            if raw == "":
                return models[default_idx]["id"]
            n = int(raw) - 1
            if 0 <= n < len(models):
                return models[n]["id"]
        except (ValueError, EOFError):
            pass
        print(f"  {BR_RED}Please enter a number between 1 and {len(models)}.{RESET}")


@dataclass
class Config:
    backend:             str = "anthropic"
    model:               str = "claude-opus-4-5"
    system_prompt:       str | None = None
    system_prompt_file:  str | None = None
    ide_context:         str | None = None
    ide_context_file:    str | None = None
    log_dir:             str = "~/.modelrelay/logs"
    usage_interval_s:    float = 30.0
    reconnect_attempts:  int = 3
    working_dir:         Path = field(default_factory=Path.cwd)

    def to_dict(self) -> dict:
        return {
            "backend":            self.backend,
            "model":              self.model,
            "system_prompt":      self.system_prompt,
            "system_prompt_file": self.system_prompt_file,
            "ide_context":        self.ide_context,
            "ide_context_file":   self.ide_context_file,
            "log_dir":            self.log_dir,
            "usage_interval_s":   self.usage_interval_s,
            "reconnect_attempts": self.reconnect_attempts,
        }

    @classmethod
    def from_dict(cls, d: dict, working_dir: Path) -> "Config":
        return cls(
            backend            = d.get("backend",            "anthropic"),
            model              = d.get("model",              "claude-opus-4-5"),
            system_prompt      = d.get("system_prompt"),
            system_prompt_file = d.get("system_prompt_file"),
            ide_context        = d.get("ide_context"),
            ide_context_file   = d.get("ide_context_file"),
            log_dir            = d.get("log_dir",            "~/.modelrelay/logs"),
            usage_interval_s   = float(d.get("usage_interval_s", 30.0)),
            reconnect_attempts = int(d.get("reconnect_attempts", 3)),
            working_dir        = working_dir,
        )

    def to_argv(self, modelrelay_path: Path) -> list[str]:
        """Build the argv list for the modelrelay subprocess."""
        args = [
            str(modelrelay_path),
            "--backend", self.backend,
            "--model", self.model,
            "--log-dir", os.path.expanduser(self.log_dir),
            "--usage-interval", str(self.usage_interval_s),
            "--reconnect-attempts", str(self.reconnect_attempts),
        ]
        if self.system_prompt:
            args += ["--system-prompt", self.system_prompt]
        if self.system_prompt_file:
            args += ["--system-prompt-file", self.system_prompt_file]
        if self.ide_context:
            args += ["--ide-context", self.ide_context]
        if self.ide_context_file:
            args += ["--ide-context-file", self.ide_context_file]
        return args


# ─── UI Helpers ──────────────────────────────────────────────────────────────

def print_banner():
    w = term_width()
    lines = [
        f"{BR_CYAN}",
        r"  ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗      ██████╗  ██████╗ ██╗  ██╗",
        r"  ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     ██╔══██╗██╔════╝ ██║ ██╔╝",
        r"  ██╔████╔██║██║   ██║██║  ██║█████╗  ██║     ██████╔╝██║  ███╗█████╔╝ ",
        r"  ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     ██╔══██╗██║   ██║██╔═██╗ ",
        r"  ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗██║  ██║╚██████╔╝██║  ██╗",
        r"  ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝",
        f"{RESET}",
        f"  {BR_BLACK}RELAY{RESET}  {DIM}Line-Oriented JSON Protocol Host  •  v0.6.0{RESET}",
        "",
    ]
    print()
    for l in lines:
        print(l)
    print(hr())


def clear_screen():
    print("\033[2J\033[H", end="")


def print_section(title: str, color=BR_YELLOW):
    print(f"\n{color}{BOLD}  {title}{RESET}")
    print(f"  {color}{'─' * (len(title) + 2)}{RESET}")


def prompt_choice(question: str, choices: list[tuple[str, str]],
                  default_idx: int = 0) -> str:
    """Show a numbered menu, return chosen value."""
    print(f"\n  {BOLD}{WHITE}{question}{RESET}")
    for i, (key, label) in enumerate(choices):
        marker = f"{BR_CYAN}▶{RESET}" if i == default_idx else f"{BR_BLACK}○{RESET}"
        default_tag = f" {DIM}(default){RESET}" if i == default_idx else ""
        print(f"    {marker} {BOLD}{i+1}{RESET}. {label}{default_tag}")
    while True:
        try:
            raw = input(_rl(f"\n  {CYAN}Choice [{default_idx+1}]{RESET}: ")).strip()
            if raw == "":
                return choices[default_idx][0]
            n = int(raw) - 1
            if 0 <= n < len(choices):
                return choices[n][0]
        except ValueError:
            pass
        print(f"  {BR_RED}Please enter a number between 1 and {len(choices)}.{RESET}")


def prompt_text(question: str, default: str = "", hint: str = "") -> str:
    hint_str = f" {DIM}[{hint}]{RESET}" if hint else ""
    default_str = f" {DIM}(default: {default}){RESET}" if default else ""
    try:
        val = input(_rl(f"  {CYAN}{question}{hint_str}{default_str}: {RESET}")).strip()
        return val if val else default
    except EOFError:
        return default


def prompt_optional(question: str, hint: str = "") -> str | None:
    hint_str = f" {DIM}[{hint}]{RESET}" if hint else ""
    try:
        val = input(_rl(f"  {DIM}{question}{hint_str} (leave blank to skip): {RESET}")).strip()
        return val if val else None
    except EOFError:
        return None


def prompt_directory(question: str, default: Path | None = None) -> Path:
    """Prompt for a directory path, with tab-completion hint."""
    default_str = str(default) if default is not None else ""
    while True:
        raw = prompt_text(question, default=default_str, hint="directory path")
        if not raw:
            print(f"  {BR_RED}  Please enter a directory path.{RESET}")
            continue
        p = Path(os.path.expanduser(raw)).resolve()
        if p.exists() and p.is_dir():
            return p
        elif not p.exists():
            yn = input(_rl(f"  {YELLOW}  Directory does not exist. Create it? [y/N]: {RESET}")).strip().lower()
            if yn == "y":
                p.mkdir(parents=True, exist_ok=True)
                return p
        else:
            print(f"  {BR_RED}  Not a directory.{RESET}")


def confirm(question: str, default_yes: bool = True) -> bool:
    yN = "Y/n" if default_yes else "y/N"
    ans = input(_rl(f"  {YELLOW}  {question} [{yN}]: {RESET}")).strip().lower()
    if ans == "":
        return default_yes
    return ans in ("y", "yes")


# ─── TUI ─────────────────────────────────────────────────────────────────────

import curses as _curses
import queue  as _queue


def make_diff_plain(
    old_lines: list[str], new_lines: list[str],
    fromfile: str = "before", tofile: str = "after"
) -> list[tuple[str, str]]:
    """Return (text, style) pairs for unified diff, suitable for curses rendering."""
    diff = list(difflib.unified_diff(
        old_lines, new_lines, fromfile=fromfile, tofile=tofile, lineterm=""))
    if not diff:
        return [("  (no changes)", "dim")]
    result = []
    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            result.append((line, "diff_hdr"))
        elif line.startswith("@@"):
            result.append((line, "cyan"))
        elif line.startswith("+"):
            result.append((line, "diff_add"))
        elif line.startswith("-"):
            result.append((line, "diff_rem"))
        else:
            result.append((line, "diff_ctx"))
    return result


class TUI:
    """Three-region split-screen terminal UI powered by curses.

    Regions (top to bottom):
      • Status  – fixed height, shows state / model / token / file count
      • Model   – ~2/3 of remaining height, scrolling output from the model
                  and tool events
      • Input   – remainder, dimmed while waiting, highlighted on user turn
    """

    _STATUS_H = 3   # rows for the status region (border + content + border)

    # Color pair names → index
    _CP: dict[str, int] = {}

    def __init__(self, stdscr):
        self._scr = stdscr

        # Thread-safe queue: async thread → main thread draw updates
        self._draw_q: _queue.Queue = _queue.Queue()

        # Set from async thread once its loop is running
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._prompt_q:   asyncio.Queue | None             = None
        self._confirm_q:  asyncio.Queue | None             = None

        # ── Mutable session state (read only on main thread) ─────────────────
        self._state      = "connecting"
        self._model_name = ""
        self._tokens     = 0
        self._cost       = 0.0
        self._desc       = ""
        self._files: dict[str, int] = {}   # resource → line count

        # Startup errors to echo to the terminal after curses exits
        self._startup_errors: list[str] = []

        # Model output buffer: list of (text, style_name)
        self._model_lines: list[tuple[str, str]] = []
        self._text_buf = ""   # accumulated streaming text; word-wrapped on render

        # Scroll state: offset > 0 means the user is reading backscroll.
        # When scrolled, _scroll_total is snapshotted so that new content
        # streaming in does not shift the frozen view.
        self._scroll_offset: int = 0   # lines from live bottom; 0 = pinned
        self._scroll_total:  int = 0   # len(wrapped) when scroll mode entered

        # Input state
        self._input_active    = False
        self._input_confirming = False
        self._input_buf       = ""
        self._input_cursor    = 0

        # ── curses init ───────────────────────────────────────────────────────
        _curses.curs_set(1)
        _curses.start_color()
        _curses.use_default_colors()
        self._init_colors()
        stdscr.keypad(True)
        # Enable mouse event reporting for scroll-wheel support.
        # Wrapped in try/except: not all terminals or curses builds support it.
        try:
            _curses.mousemask(
                getattr(_curses, "ALL_MOUSE_EVENTS",        0) |
                getattr(_curses, "REPORT_MOUSE_POSITION",   0))
        except Exception:
            pass

        self._status_win = None
        self._model_win  = None
        self._input_win  = None
        self._layout()
        self._redraw_all()

    # ── Color setup ──────────────────────────────────────────────────────────

    def _init_colors(self):
        bright_black = (_curses.COLOR_BLACK + 8
                        if _curses.COLORS >= 16 else _curses.COLOR_BLACK)
        pairs = [
            ("normal",          _curses.COLOR_WHITE,   -1),
            ("dim",             bright_black,           -1),
            ("green",           _curses.COLOR_GREEN,   -1),
            ("yellow",          _curses.COLOR_YELLOW,  -1),
            ("red",             _curses.COLOR_RED,     -1),
            ("cyan",            _curses.COLOR_CYAN,    -1),
            ("magenta",         _curses.COLOR_MAGENTA, -1),
            ("diff_add",        _curses.COLOR_GREEN,   -1),
            ("diff_rem",        _curses.COLOR_RED,     -1),
            ("diff_hdr",        _curses.COLOR_WHITE,   -1),
            ("diff_ctx",        bright_black,           -1),
            ("status_idle",     _curses.COLOR_GREEN,   -1),
            ("status_gen",      _curses.COLOR_YELLOW,  -1),
            ("status_tool",     _curses.COLOR_CYAN,    -1),
            ("status_rate",     _curses.COLOR_RED,     -1),
            ("status_conn",     _curses.COLOR_MAGENTA, -1),
            ("status_ended",    bright_black,           -1),
            ("input_active",    _curses.COLOR_CYAN,    -1),
            ("input_idle",      bright_black,           -1),
        ]
        for i, (name, fg, bg) in enumerate(pairs, start=1):
            try:
                _curses.init_pair(i, fg, bg)
            except _curses.error:
                pass
            TUI._CP[name] = i

    def _attr(self, style: str, bold: bool = False, dim: bool = False) -> int:
        a = _curses.color_pair(TUI._CP.get(style, 0))
        if bold: a |= _curses.A_BOLD
        if dim:  a |= _curses.A_DIM
        return a

    # ── Layout ────────────────────────────────────────────────────────────────

    def _layout(self, is_resize: bool = False):
        if is_resize:
            # Read the authoritative terminal size from the kernel rather than
            # curses.LINES/COLS, which are not reliably updated by ncurses in
            # all environments before KEY_RESIZE is delivered.
            try:
                import struct as _struct
                import fcntl as _fcntl
                import termios as _termios
                # fd 0 is the controlling terminal; curses windows have no
                # fileno() method so we go to the underlying fd directly.
                _ws = _fcntl.ioctl(0, _termios.TIOCGWINSZ, b'\x00' * 8)
                _rows, _cols = _struct.unpack('HHHH', _ws)[:2]
            except Exception:
                _rows, _cols = _curses.LINES, _curses.COLS
            try:
                _curses.resizeterm(_rows, _cols)
            except AttributeError:
                pass  # some curses builds omit resizeterm
            self._scr.clearok(True)
            # Use the ioctl-derived dimensions directly.  In some ncurses
            # builds resizeterm() does not update what getmaxyx() returns, so
            # reading back through getmaxyx() would give the stale pre-resize
            # size, leaving self._h/_w unchanged and causing the TIOCGWINSZ
            # poll to fire again on the next tick (infinite redraw loop).
            h, w = _rows, _cols
            # Resize changes wrap metrics (different column width), so any
            # saved scroll position would point to a different spot in the new
            # wrapping.  Snap back to the live view rather than show garbage.
            self._scroll_offset = 0
            self._scroll_total  = 0
        else:
            h, w = self._scr.getmaxyx()
        self._h, self._w = h, w

        status_h = self._STATUS_H
        rem      = h - status_h
        input_h  = 2
        model_h  = max(4, rem - input_h)

        self._status_h = status_h
        self._model_h  = model_h
        self._input_h  = input_h

        self._status_win = _curses.newwin(status_h, w, 0, 0)
        self._model_win  = _curses.newwin(model_h,  w, status_h, 0)
        self._input_win  = _curses.newwin(input_h,  w, status_h + model_h, 0)
        for win in (self._status_win, self._model_win, self._input_win):
            win.keypad(True)
        self._input_win.timeout(50)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _place_cursor(self):
        """Move the physical cursor to the input text line so it never
        appears in the status or model window after a partial redraw.
        """
        try:
            if self._input_active:
                prompt = "> "
                avail  = self._w - len(prompt) - 3
                scroll = max(0, self._input_cursor - avail)
                cx = 1 + len(prompt) + (self._input_cursor - scroll)
                self._input_win.move(1, min(cx, self._w - 2))
            else:
                self._input_win.move(1, 1)
            self._input_win.noutrefresh()
        except _curses.error:
            pass

    def _redraw_all(self):
        self._draw_status()
        self._draw_model()
        self._draw_input()
        _curses.doupdate()

    def _safe_addstr(self, win, y: int, x: int, text: str, attr: int = 0):
        try:
            win.addstr(y, x, text, attr)
        except _curses.error:
            pass

    def _draw_status(self):
        win = self._status_win
        w   = self._w
        win.erase()

        rule = self._attr("dim")

        # Top and bottom borders
        self._safe_addstr(win, 0, 0, "─" * (w - 1), rule)
        self._safe_addstr(win, 2, 0, "─" * (w - 1), rule)

        # State badge
        STATE_STYLE = {
            "idle":             "status_idle",
            "generating":       "status_gen",
            "waiting_for_tool": "status_tool",
            "rate_limited":     "status_rate",
            "connecting":       "status_conn",
            "ended":            "status_ended",
        }
        ss    = STATE_STYLE.get(self._state, "normal")
        badge = f" {self._state.upper()} "
        fc    = len(self._files)

        segments = [
            (badge,                              self._attr(ss, bold=True)),
            ("  ",                               self._attr("dim")),
            (f"model: {self._model_name}",       self._attr("dim")),
            ("   ",                              self._attr("dim")),
            (f"tokens: {self._tokens:,}",        self._attr("dim")),
            ("   ",                              self._attr("dim")),
            (f"${self._cost:.4f}",               self._attr("dim")),
            ("   ",                              self._attr("dim")),
            (f"files: {fc}",                     self._attr("dim")),
        ]
        if self._desc:
            segments += [("   ", self._attr("dim")),
                         (self._desc[:35], self._attr("dim"))]

        x = 1
        for text, attr in segments:
            if x + len(text) >= w - 1:
                break
            self._safe_addstr(win, 1, x, text, attr)
            x += len(text)

        win.noutrefresh()

    def _build_wrapped(self) -> list[tuple[str, str]]:
        """Return the complete word-wrapped display-line list for current content.

        This is extracted from _draw_model so that the scroll methods can call
        it without triggering a redraw.
        """
        import textwrap
        w     = self._w
        max_w = max(1, w - 3)
        wrapped: list[tuple[str, str]] = []
        source = list(self._model_lines)
        if self._text_buf:
            source.append((self._text_buf, "text"))

        for raw, style in source:
            if style == "text":
                paragraphs = raw.split("\n\n")
                for i, para in enumerate(paragraphs):
                    for line in para.split("\n"):
                        line = line.rstrip()
                        if line:
                            indent     = len(line) - len(line.lstrip())
                            indent_str = line[:indent]
                            inner_w    = max(1, max_w - indent)
                            for wline in textwrap.wrap(
                                    line.lstrip(), inner_w,
                                    initial_indent=indent_str,
                                    subsequent_indent=indent_str):
                                wrapped.append((wline, "normal"))
                        else:
                            wrapped.append(("", "normal"))
                    if i < len(paragraphs) - 1:
                        wrapped.append(("", "normal"))
            else:
                while len(raw) > max_w:
                    wrapped.append((raw[:max_w], style))
                    raw = raw[max_w:]
                wrapped.append((raw, style))

        return wrapped

    def _draw_model(self):
        win = self._model_win
        w   = self._w
        h   = self._model_h
        win.erase()

        content_h = h - 1   # bottom row reserved for border

        wrapped = self._build_wrapped()
        total   = len(wrapped)

        if self._scroll_offset == 0:
            # Live/pinned: always show the newest content
            visible   = wrapped[-content_h:] if wrapped else []
            new_below = 0
        else:
            # Frozen view: anchored at _scroll_total so that new output
            # arriving during reading does not shift the visible region.
            anchor    = min(self._scroll_total, total)
            new_below = max(0, total - anchor)
            end       = max(0, anchor - self._scroll_offset)
            start     = max(0, end - content_h)
            visible   = wrapped[start:end]

        STYLE_ATTRS = {
            "normal":   self._attr("normal"),
            "dim":      self._attr("dim",    dim=True),
            "green":    self._attr("green"),
            "yellow":   self._attr("yellow"),
            "red":      self._attr("red",    bold=True),
            "cyan":     self._attr("cyan"),
            "magenta":  self._attr("magenta"),
            "bold":     self._attr("normal", bold=True),
            "diff_add": self._attr("diff_add", bold=True),
            "diff_rem": self._attr("diff_rem", bold=True),
            "diff_hdr": self._attr("diff_hdr", bold=True),
            "diff_ctx": self._attr("diff_ctx", dim=True),
            "rule":     self._attr("dim"),
        }

        for i, (text, style) in enumerate(visible):
            if i >= content_h:
                break
            attr = STYLE_ATTRS.get(style, STYLE_ATTRS["normal"])
            self._safe_addstr(win, i, 1, text[:w - 2], attr)

        # Bottom border
        self._safe_addstr(win, h - 1, 0, "─" * (w - 1), self._attr("dim"))

        # Scroll indicator overlaid right-aligned on the border when scrolled
        if self._scroll_offset > 0:
            parts = [f"↑ {self._scroll_offset}"]
            if new_below:
                parts.append(f"{new_below} new ↓")
            indicator = "  " + "  ".join(parts) + "  "
            x = max(0, w - len(indicator) - 1)
            self._safe_addstr(win, h - 1, x, indicator, self._attr("yellow"))

        win.noutrefresh()

    def _draw_input(self):
        win = self._input_win
        w   = self._w
        h   = self._input_h
        win.erase()

        if self._input_active:
            bdr_attr  = self._attr("input_active", bold=True)
            text_attr = self._attr("normal")
            label     = " CONFIRM [y/n] " if self._input_confirming else " YOU "
        else:
            bdr_attr  = self._attr("input_idle", dim=True)
            text_attr = self._attr("dim", dim=True)
            label     = " … "

        self._safe_addstr(win, 0, 0, "─" * (w - 1), bdr_attr)
        self._safe_addstr(win, 0, 2, label, bdr_attr | _curses.A_BOLD)

        if self._input_active:
            prompt = "> "
            avail  = w - len(prompt) - 3
            scroll = max(0, self._input_cursor - avail)
            vis    = self._input_buf[scroll:scroll + avail]
            self._safe_addstr(win, 1, 1, prompt + vis, text_attr)
            cx = 1 + len(prompt) + (self._input_cursor - scroll)
            try:
                win.move(1, min(cx, w - 2))
            except _curses.error:
                pass
        else:
            self._safe_addstr(win, 1, 1, "  waiting for model…", text_attr)

        win.noutrefresh()

    # ── Scrolling ─────────────────────────────────────────────────────────────

    def _scroll_by(self, delta: int):
        """Scroll the model window by *delta* lines (positive = up / back).

        Entering scroll mode (offset > 0) snapshots ``_scroll_total`` so that
        new content arriving from the model does not shift the frozen view.
        Scrolling back down to offset 0 returns to live auto-scroll.
        """
        wrapped   = self._build_wrapped()
        total     = len(wrapped)
        content_h = self._model_h - 1

        if self._scroll_offset == 0 and delta > 0:
            # First scroll upward: anchor the bottom so streaming output that
            # arrives while the user is reading stays below the visible region.
            self._scroll_total = total

        anchor     = self._scroll_total if self._scroll_offset > 0 else total
        max_offset = max(0, anchor - content_h)
        self._scroll_offset = max(0, min(max_offset, self._scroll_offset + delta))

        if self._scroll_offset == 0:
            self._scroll_total = 0   # back to live mode

        self._draw_model()
        self._place_cursor()
        _curses.doupdate()

    def _scroll_to_bottom(self):
        """Return to live auto-scroll mode immediately."""
        self._scroll_offset = 0
        self._scroll_total  = 0
        self._draw_model()
        self._place_cursor()
        _curses.doupdate()

    def _page_up(self):
        """Scroll back one page (content_h - 1 lines for one-line overlap)."""
        self._scroll_by(max(1, self._model_h - 2))

    def _page_down(self):
        """Scroll forward one page."""
        self._scroll_by(-max(1, self._model_h - 2))

    # ── Thread-safe public API ────────────────────────────────────────────────

    def set_async(self, loop: asyncio.AbstractEventLoop,
                  prompt_q: "asyncio.Queue", confirm_q: "asyncio.Queue"):
        """Called from the async thread before it starts processing."""
        self._async_loop = loop
        self._prompt_q   = prompt_q
        self._confirm_q  = confirm_q

    def post(self, kind: str, **data):
        """Post an update from any thread to the main-thread draw loop."""
        self._draw_q.put_nowait({"kind": kind, **data})

    # ── Update processing (main thread only) ─────────────────────────────────

    def _apply(self, msg: dict):
        kind = msg["kind"]

        if kind == "status":
            dirty = False
            for key in ("state", "desc", "model"):
                if key in msg:
                    setattr(self, f"_{key}" if key != "model"
                            else "_model_name", msg[key])
                    dirty = True
            if dirty:
                self._draw_status()

        elif kind == "usage":
            self._tokens = msg.get("total_tokens", self._tokens)
            self._cost   = msg.get("cost_usd",     self._cost)
            self._draw_status()

        elif kind == "text_delta":
            self._text_buf += msg.get("text", "")
            self._draw_model()

        elif kind == "flush_line":
            # Commit the accumulated text buffer as a prose entry, then blank line
            if self._text_buf:
                self._model_lines.append((self._text_buf, "text"))
                self._text_buf = ""
            self._draw_model()

        elif kind == "model_line":
            # Flush any pending text before a structured line
            if self._text_buf:
                self._model_lines.append((self._text_buf, "text"))
                self._text_buf = ""
            self._model_lines.append((msg.get("text", ""), msg.get("style", "normal")))
            self._draw_model()

        elif kind == "file_stat":
            r = msg.get("resource", "")
            n = msg.get("lines")
            if n:
                self._files[r] = n
            else:
                self._files.pop(r, None)
            self._draw_status()

        elif kind == "startup_error":
            msg_text = msg.get("message", "")
            if msg_text:
                self._startup_errors.append(msg_text)

        elif kind == "input_active":
            self._input_active     = msg.get("active", False)
            self._input_confirming = msg.get("confirming", False)
            if not self._input_active:
                self._input_buf    = ""
                self._input_cursor = 0
            self._draw_input()

    # ── Input event loop (main thread) ───────────────────────────────────────

    def run_input_loop(self):
        """Block, processing keyboard and draw updates, until 'quit' is posted."""
        self._input_win.timeout(50)

        while True:
            # Drain the draw queue
            needs_refresh = False
            while True:
                try:
                    msg = self._draw_q.get_nowait()
                except _queue.Empty:
                    break
                if msg["kind"] == "quit":
                    return
                self._apply(msg)
                needs_refresh = True

            if needs_refresh:
                self._place_cursor()
                _curses.doupdate()

            try:
                ch = self._input_win.getch()
            except Exception:
                ch = -1

            if ch == -1:
                # Poll for terminal size changes as a fallback for
                # environments where SIGWINCH is not delivered to getch().
                try:
                    import struct as _s, fcntl as _f, termios as _t
                    _ws = _f.ioctl(0, _t.TIOCGWINSZ, b'\x00'*8)
                    _rh, _rw = _s.unpack('HHHH', _ws)[:2]
                    if _rh != self._h or _rw != self._w:
                        self._layout(is_resize=True)
                        self._redraw_all()
                        # resizeterm() enqueues KEY_RESIZE events on some
                        # ncurses builds; flush them so the KEY_RESIZE branch
                        # below doesn't fire and loop back into another resize.
                        try:
                            _curses.flushinp()
                        except Exception:
                            pass
                except Exception:
                    pass
                continue
            if ch == 3:                      # Ctrl+C
                raise KeyboardInterrupt
            if ch == _curses.KEY_RESIZE:
                self._layout(is_resize=True)
                self._redraw_all()
                try:
                    _curses.flushinp()
                except Exception:
                    pass
                continue

            # ── Scroll keys (work regardless of input_active state) ───────────
            if ch == _curses.KEY_PPAGE:          # Page Up
                self._page_up()
                continue
            if ch == _curses.KEY_NPAGE:          # Page Down
                self._page_down()
                continue
            if ch == _curses.KEY_MOUSE:
                try:
                    _, _, _, _, bstate = _curses.getmouse()
                    _b4 = getattr(_curses, "BUTTON4_PRESSED", 0)
                    _b5 = getattr(_curses, "BUTTON5_PRESSED", 0)
                    if _b4 and (bstate & _b4):   # wheel up → scroll back
                        self._scroll_by(3)
                    elif _b5 and (bstate & _b5): # wheel down → scroll forward
                        self._scroll_by(-3)
                except Exception:
                    pass
                continue

            if not self._input_active:
                continue

            needs_refresh = True

            if ch in (_curses.KEY_ENTER, 10, 13):
                # Snap to live view so the response is immediately visible.
                self._scroll_to_bottom()
                text               = self._input_buf
                self._input_buf    = ""
                self._input_cursor = 0
                self._draw_input()
                self._place_cursor()
                _curses.doupdate()
                if self._async_loop:
                    q = (self._confirm_q if self._input_confirming
                         else self._prompt_q)
                    if q is not None:
                        self._async_loop.call_soon_threadsafe(
                            q.put_nowait, text)
                continue

            elif ch in (_curses.KEY_BACKSPACE, 127, 8):
                if self._input_cursor > 0:
                    self._input_buf = (
                        self._input_buf[:self._input_cursor - 1] +
                        self._input_buf[self._input_cursor:]
                    )
                    self._input_cursor -= 1

            elif ch == _curses.KEY_DC:
                if self._input_cursor < len(self._input_buf):
                    self._input_buf = (
                        self._input_buf[:self._input_cursor] +
                        self._input_buf[self._input_cursor + 1:]
                    )

            elif ch == _curses.KEY_LEFT:
                self._input_cursor = max(0, self._input_cursor - 1)

            elif ch == _curses.KEY_RIGHT:
                self._input_cursor = min(len(self._input_buf), self._input_cursor + 1)

            elif ch in (_curses.KEY_HOME, 1):   # Home or Ctrl-A
                self._input_cursor = 0

            elif ch in (_curses.KEY_END, 5):    # End or Ctrl-E
                self._input_cursor = len(self._input_buf)

            elif 32 <= ch <= 126:
                self._input_buf = (
                    self._input_buf[:self._input_cursor] +
                    chr(ch) +
                    self._input_buf[self._input_cursor:]
                )
                self._input_cursor += 1

            else:
                needs_refresh = False

            if needs_refresh:
                self._draw_input()
                self._place_cursor()
                _curses.doupdate()


# ─── Session ─────────────────────────────────────────────────────────────────

class ModelRelaySession:
    def __init__(self, config: Config, sandbox: Sandbox,
                 modelrelay_path: Path, tui: TUI):
        self.config          = config
        self.sandbox         = sandbox
        self.modelrelay_path = modelrelay_path
        self.tui             = tui
        self.proc:           asyncio.subprocess.Process | None = None
        self._send_queue:    asyncio.Queue[dict] = asyncio.Queue()
        self._prompt_queue:  asyncio.Queue[str | None] = asyncio.Queue()
        self._confirm_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._confirm_lock  = asyncio.Lock()
        self._killed        = False   # set by kill(); tells stop() to just reap
        self._session_ended_event = asyncio.Event()  # set by _on_session_ended
        # Count of active _ask_confirm calls. _on_activity must not
        # deactivate the input box while this is non-zero.
        self._confirming    = 0
        # Last activity state received; used to re-evaluate input state
        # after a confirm dialog closes.
        self._activity_state = ""

    # ── Subprocess management ─────────────────────────────────────────────────

    async def start(self) -> tuple[bool, str]:
        """Start the subprocess. Returns (success, error_message)."""
        p = self.modelrelay_path
        if not p.exists():
            msg = f"modelrelay not found: {p}"
            self.tui.post("model_line", text=f"ERROR: {msg}", style="red")
            return False, msg
        argv = self.config.to_argv(p)
        env  = dict(os.environ)
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin  = asyncio.subprocess.PIPE,
                stdout = asyncio.subprocess.PIPE,
                stderr = asyncio.subprocess.PIPE,
                env    = env,
                cwd    = str(self.sandbox.root),
            )
            return True, ""
        except Exception as e:
            msg = f"failed to start modelrelay: {e}"
            self.tui.post("model_line", text=f"ERROR: {msg}", style="red")
            return False, msg

    def kill(self):
        """Kill the subprocess immediately (synchronous; safe to call from any
        thread).  Sets _killed so stop() knows it only needs to reap.
        """
        self._killed = True
        if self.proc and self.proc.returncode is None:
            try:
                self.proc.kill()
            except Exception:
                pass

    _STOP_TIMEOUT = 5.0   # seconds before forcing subprocess exit

    async def stop(self):
        if not (self.proc and self.proc.returncode is None):
            return
        try:
            if self._killed:
                # kill() was already called synchronously (Ctrl+C path).
                self.tui.post("model_line",
                    text="  ⏹ subprocess killed", style="dim")
                await self.proc.wait()
            else:
                # ── clean /quit path ──────────────────────────────────────
                self.tui.post("model_line",
                    text="  → sending quit to modelrelay…", style="dim")
                await self._send(cmd_quit())

                # Close stdin so run_inbound readline() gets EOF immediately.
                if self.proc.stdin:
                    self.proc.stdin.close()

                self.tui.post("model_line",
                    text="  → waiting for modelrelay to finish…", style="dim")
                try:
                    # Wait for session.ended (set by _on_session_ended via
                    # _stdout_reader, which is kept alive until after stop()).
                    await asyncio.wait_for(
                        self._session_ended_event.wait(),
                        timeout=self._STOP_TIMEOUT)
                    self.tui.post("model_line",
                        text="  ✓ modelrelay finished cleanly", style="dim")
                    # Give the process a moment to actually exit, then reap.
                    await self.proc.wait()
                except asyncio.TimeoutError:
                    self.tui.post("model_line",
                        text=f"  ⚠ modelrelay did not finish within "
                             f"{self._STOP_TIMEOUT:.0f}s — forcing kill",
                        style="red")
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
                    await self.proc.wait()
        except Exception as e:
            self.tui.post("model_line",
                text=f"  ⚠ stop error: {e}", style="red")

    # ── I/O ───────────────────────────────────────────────────────────────────

    async def _send(self, envelope: dict):
        if self.proc and self.proc.stdin:
            self.proc.stdin.write(encode(envelope))
            await self.proc.stdin.drain()

    async def _sender_loop(self):
        while True:
            env = await self._send_queue.get()
            await self._send(env)

    async def _stdout_reader(self):
        while self.proc and self.proc.stdout:
            line = await self.proc.stdout.readline()
            if not line:
                break
            try:
                env = decode(line.decode("utf-8", errors="replace"))
                await self._dispatch(env)
            except Exception as e:
                import traceback
                tb = traceback.format_exc().strip().splitlines()
                self.tui.post("model_line",
                    text=f"[dispatch error] {e}", style="red")
                for tbline in tb[-5:]:
                    self.tui.post("model_line", text=f"  {tbline}", style="red")

    async def _stderr_reader(self):
        while self.proc and self.proc.stderr:
            line = await self.proc.stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip("\n\r")
            self.tui.post("model_line", text=f"[stderr] {text}", style="red")

    # ── Dispatch ──────────────────────────────────────────────────────────────

    async def _dispatch(self, env: dict):
        t = env.get("type", "")
        if   t == "model.text_delta":      self._on_text_delta(env["payload"].get("text", ""))
        elif t == "status.activity":       await self._on_activity(env["payload"])
        elif t == "status.usage":          self._on_usage(env["payload"])
        elif t == "tool.content_request":  await self._on_content_request(env)
        elif t == "tool.stat_request":     await self._on_stat_request(env)
        elif t == "tool.replace_request":  asyncio.create_task(self._on_replace_request(env))
        elif t == "tool.insert_request":   asyncio.create_task(self._on_insert_request(env))
        elif t == "session.ended":         await self._on_session_ended(env["payload"])
        elif t == "error":                 self._on_error(env["payload"])

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_text_delta(self, text: str):
        self.tui.post("text_delta", text=text)

    def _flush_text(self):
        self.tui.post("flush_line")

    async def _on_activity(self, payload: dict):
        state = payload.get("state", "")
        desc  = payload.get("description", "")
        # Only flush the text buffer when leaving the generating state;
        # flushing on every activity event (including mid-stream generating
        # updates) would commit each token as its own _model_lines entry.
        if state != "generating":
            self._flush_text()
        self._activity_state = state
        self.tui.post("status", state=state, desc=desc)
        if state == "idle":
            # Only activate the prompt input when no confirm dialog is open
            if not self._confirming:
                self.tui.post("input_active", active=True)
        else:
            # Don't collapse a confirm dialog that is already in progress
            if not self._confirming:
                self.tui.post("input_active", active=False)

    def _on_usage(self, payload: dict):
        self.tui.post("usage",
            total_tokens = payload.get("total_tokens", 0),
            cost_usd     = payload.get("cost_usd", 0.0))

    def _rel(self, resource: str) -> str:
        return self.sandbox.relative(
            self.sandbox.resolve(resource) or Path(resource))

    async def _on_stat_request(self, env: dict):
        resource = env["payload"]["resource"]
        req_id   = env["id"]
        rel      = self._rel(resource)
        exists, total_lines, last_mod, error = stat_file(self.sandbox, resource)

        if error:
            self.tui.post("model_line", text=f"⚙ stat → {rel}", style="cyan")
            self.tui.post("model_line", text=f"  ⚠ {error}", style="red")
        elif not exists:
            self.tui.post("model_line", text=f"⚙ stat → {rel}  (not found)", style="cyan")
        else:
            self.tui.post("model_line",
                text=f"⚙ stat → {rel}  {total_lines:,} lines", style="cyan")
            self.tui.post("file_stat", resource=resource, lines=total_lines)

        await self._send_queue.put(
            tool_stat_response(req_id, resource, exists, total_lines, last_mod, error))

    async def _on_content_request(self, env: dict):
        payload  = env["payload"]
        resource = payload["resource"]
        req_id   = env["id"]
        rel      = self._rel(resource)

        if "pattern" in payload:
            pattern       = payload["pattern"]
            max_results   = payload.get("max_results", 20)
            context_lines = payload.get("context_lines", 2)
            total, regions, truncated, error = search_file(
                self.sandbox, resource, pattern, max_results, context_lines)
            if error:
                self.tui.post("model_line",
                    text=f"⚙ search → {rel}  {pattern!r}", style="magenta")
                self.tui.post("model_line", text=f"  ⚠ {error}", style="red")
            else:
                hits  = sum(len(r.get("match_lines", [])) for r in regions)
                trunc = "  (truncated)" if truncated else ""
                self.tui.post("model_line",
                    text=f"⚙ search → {rel}  {pattern!r}  {hits} match(es){trunc}",
                    style="magenta")
            await self._send_queue.put(
                tool_content_response(req_id, resource, total, regions, truncated, error))
        else:
            ranges = payload.get("ranges", [])
            total, regions, error = read_file_ranges(self.sandbox, resource, ranges)
            if error:
                self.tui.post("model_line",
                    text=f"⚙ read → {rel}  {len(ranges)} range(s)", style="cyan")
                self.tui.post("model_line", text=f"  ⚠ {error}", style="red")
            else:
                n = sum(len(r["lines"]) for r in regions)
                self.tui.post("model_line",
                    text=f"⚙ read → {rel}  {n} lines  ({total:,} total)",
                    style="cyan")
            await self._send_queue.put(
                tool_content_response(req_id, resource, total, regions, False, error))

    async def _on_replace_request(self, env: dict):
        # Runs as a task from _dispatch so _stdout_reader is never blocked
        # waiting for user input.  Exceptions are caught here because they
        # would otherwise be silently swallowed by the task.
        try:
            await self._on_replace_request_inner(env)
        except Exception as e:
            import traceback
            self.tui.post("model_line",
                text=f"[replace error] {e}", style="red")
            for ln in traceback.format_exc().strip().splitlines()[-5:]:
                self.tui.post("model_line", text=f"  {ln}", style="red")

    async def _on_replace_request_inner(self, env: dict):
        payload    = env["payload"]
        resource   = payload["resource"]
        start_line = payload["start_line"]
        end_line   = payload["end_line"]
        new_lines  = payload["new_lines"]
        rel        = self._rel(resource)

        async with self._confirm_lock:
            req_id = env["id"]
            p = self.sandbox.resolve(resource)
            if p is None:
                self.tui.post("model_line",
                    text=f"⚙ replace → {rel}  DENIED (outside sandbox)", style="red")
                await self._send_queue.put(
                    tool_replace_response(req_id, "Access denied: path outside sandbox"))
                await self._send_queue.put(
                    cmd_invalidate(resource, "Access denied: path outside sandbox"))
                return
            if not p.exists():
                self.tui.post("model_line",
                    text=f"⚙ replace → {rel}  file not found", style="red")
                await self._send_queue.put(tool_replace_response(req_id, "File not found"))
                await self._send_queue.put(cmd_invalidate(resource, "File not found"))
                return

            old_lines = read_file_lines(p)
            old_chunk = old_lines[start_line - 1:end_line]
            diff      = make_diff_plain(old_chunk, new_lines,
                            fromfile=f"{rel} lines {start_line}–{end_line}",
                            tofile=f"{rel} (proposed)")

            self.tui.post("model_line",
                text=f"⚙ replace → {rel}  lines {start_line}–{end_line}", style="yellow")
            await self._confirm_and_apply(
                diff          = diff,
                question      = "Apply this change?",
                apply_fn      = lambda: apply_replace(p, start_line, end_line, new_lines),
                resource      = resource,
                reject_reason = "User rejected the proposed replacement",
                reject_line   = start_line,
                respond_fn    = lambda err: tool_replace_response(req_id, err),
            )

    async def _on_insert_request(self, env: dict):
        # Runs as a task from _dispatch so _stdout_reader is never blocked
        # waiting for user input.  Exceptions are caught here because they
        # would otherwise be silently swallowed by the task.
        try:
            await self._on_insert_request_inner(env)
        except Exception as e:
            import traceback
            self.tui.post("model_line",
                text=f"[insert error] {e}", style="red")
            for ln in traceback.format_exc().strip().splitlines()[-5:]:
                self.tui.post("model_line", text=f"  {ln}", style="red")

    async def _on_insert_request_inner(self, env: dict):
        payload    = env["payload"]
        resource   = payload["resource"]
        after_line = payload["after_line"]
        new_lines  = payload["new_lines"]
        rel        = self._rel(resource)

        async with self._confirm_lock:
            req_id = env["id"]
            p = self.sandbox.resolve(resource)
            if p is None:
                self.tui.post("model_line",
                    text=f"⚙ insert → {rel}  DENIED (outside sandbox)", style="red")
                await self._send_queue.put(
                    tool_insert_response(req_id, "Access denied: path outside sandbox"))
                await self._send_queue.put(
                    cmd_invalidate(resource, "Access denied: path outside sandbox"))
                return

            old_lines = read_file_lines(p) if p.exists() else []
            new_full  = old_lines[:after_line] + new_lines + old_lines[after_line:]
            diff      = make_diff_plain(old_lines, new_full,
                            fromfile=f"{rel} (before)",
                            tofile=f"{rel} (after insert)")

            self.tui.post("model_line",
                text=f"⚙ insert → {rel}  after line {after_line}", style="yellow")

            def _do_insert():
                if not p.exists():
                    p.parent.mkdir(parents=True, exist_ok=True)
                return apply_insert(p, after_line, new_lines)

            await self._confirm_and_apply(
                diff          = diff,
                question      = "Apply this insertion?",
                apply_fn      = _do_insert,
                resource      = resource,
                reject_reason = "User rejected the proposed insertion",
                reject_line   = after_line,
                respond_fn    = lambda err: tool_insert_response(req_id, err),
            )

    async def _on_session_ended(self, payload: dict):
        self._flush_text()
        self._session_ended_event.set()
        logfile = payload.get("logfile", "")
        summary = payload.get("summary", {})
        self.tui.post("status", state="ended", desc="session ended")
        self.tui.post("model_line", text="─" * 60, style="rule")
        self.tui.post("model_line", text="Session ended", style="green")
        if logfile:
            self.tui.post("model_line", text=f"  Log: {logfile}", style="dim")
        if summary:
            pt   = summary.get("prompt_tokens", 0)
            ct   = summary.get("completion_tokens", 0)
            tt   = summary.get("total_tokens", 0)
            cost = summary.get("cost_usd", 0.0)
            self.tui.post("model_line",
                text=f"  {pt:,} prompt + {ct:,} completion = {tt:,} tokens  ${cost:.6f}",
                style="dim")
        self.tui.post("input_active", active=False)

    def _on_error(self, payload: dict):
        code  = payload.get("code", "UNKNOWN")
        msg   = payload.get("message", "")
        fatal = payload.get("fatal", False)
        tag   = "FATAL ERROR" if fatal else "ERROR"
        self.tui.post("model_line", text=f"  {tag}  [{code}] {msg}", style="red")

    async def _ask_confirm(self, question: str) -> bool:
        """Post a y/n prompt and block until the user answers.

        Must be called while self._confirm_lock is held so that
        _prompt_loop cannot race on the same queue item.
        _confirming is incremented for the duration so that _on_activity
        does not collapse the confirm box when generating resumes.
        """
        self._confirming += 1
        try:
            self.tui.post("model_line", text=f"❓ {question}  [y/n]", style="yellow")
            self.tui.post("input_active", active=True, confirming=True)
            while True:
                ans = await self._confirm_queue.get()
                if ans is None:
                    return True
                ans = ans.strip().lower()
                if ans in ("y", "yes"):
                    return True
                if ans in ("n", "no"):
                    return False
                self.tui.post("model_line", text="Please type y or n", style="yellow")
        finally:
            self._confirming -= 1
            if self._confirming == 0 and self._activity_state == "idle":
                # activity(idle) arrived while we were confirming and was
                # suppressed; re-activate the prompt input now.
                self.tui.post("input_active", active=True)
            else:
                self.tui.post("input_active", active=False)

    async def _confirm_and_apply(
        self,
        diff:          list[tuple[str, str]],
        question:      str,
        apply_fn,                           # () -> str | None  (error msg)
        resource:      str,
        reject_reason: str,
        reject_line:   int = 0,
        respond_fn = None,                  # (error: str|None) -> dict
    ) -> None:
        """Show a diff, ask for confirmation, then apply and send the response.

        Must be called while self._confirm_lock is held.
        Factors out the identical tail shared by _on_replace_request and
        _on_insert_request.

        respond_fn, if given, is called with the outcome error string (None on
        success) and the resulting envelope is written to _send_queue.  This is
        required for request/response tools (replace, insert) so that modelrelay's
        pending future is resolved.  On rejection, cmd.invalidate is also sent
        to inject the "don't retry" note into the model context.
        """
        for text, style in diff:
            self.tui.post("model_line", text=f"  {text}", style=style)

        if await self._ask_confirm(question):
            err = apply_fn()
            if err:
                self.tui.post("model_line", text=f"  ⚠ write failed: {err}", style="red")
            else:
                self.tui.post("model_line", text="  ✓ applied", style="green")
            # Always send a response so the pending future resolves
            if respond_fn is not None:
                await self._send_queue.put(respond_fn(err))
        else:
            self.tui.post("model_line", text="  ✗ rejected by user", style="yellow")
            if respond_fn is not None:
                # Unblock the pending future with an error payload …
                await self._send_queue.put(respond_fn(reject_reason))
            # … and tell the model not to retry.
            await self._send_queue.put(
                cmd_invalidate(resource, reject_reason, start_line=reject_line))

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self):
        ok, err_msg = await self.start()
        if not ok:
            # Keep the screen up so the user can read the error, then quit
            # when they press any key (or after a fallback timeout).
            self.tui.post("model_line",
                text="Press any key to exit.", style="dim")
            self.tui.post("input_active", active=True)
            await self._prompt_queue.get()
            self.tui.post("startup_error", message=err_msg)
            self.tui.post("quit")
            return

        self.tui.post("status", model=self.config.model, state="connecting")

        stdout_task = asyncio.create_task(self._stdout_reader())
        tasks = [
            stdout_task,
            asyncio.create_task(self._stderr_reader()),
            asyncio.create_task(self._sender_loop()),
        ]
        try:
            await self._prompt_loop()
        finally:
            # Cancel sender and stderr immediately — they're no longer needed.
            # Keep stdout_task alive so _on_session_ended can fire during stop().
            for t in tasks:
                if t is not stdout_task:
                    t.cancel()
            await self.stop()
            stdout_task.cancel()
            self.tui.post("quit")

    async def _prompt_loop(self):
        self.tui.post("model_line",
            text="/quit  /invalidate <file>  /help", style="dim")

        while True:
            text = await self._prompt_queue.get()
            if text is None:
                break
            text = text.strip()
            if not text:
                continue
            if text.startswith("/"):
                if await self._handle_command(text):
                    break
                continue
            self.tui.post("input_active", active=False)
            self.tui.post("model_line", text=f"you › {text}", style="bold")
            await self._send_queue.put(cmd_prompt(text))

    async def _handle_command(self, line: str) -> bool:
        """Returns True if the session should end."""
        parts = line.split(None, 1)
        cmd   = parts[0].lower()
        arg   = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/quit":
            self.tui.post("model_line", text="Shutting down…", style="dim")
            self.tui.post("input_active", active=False)
            return True
        elif cmd == "/invalidate":
            if not arg:
                self.tui.post("model_line",
                    text="Usage: /invalidate <resource>", style="red")
            else:
                await self._send_queue.put(
                    cmd_invalidate(arg, "User signalled file change"))
                self.tui.post("model_line",
                    text=f"✓ invalidate sent for {arg}", style="green")
        elif cmd == "/help":
            for ln in [
                "Commands:",
                "/quit                  End the session cleanly",
                "/invalidate <file>     Notify modelrelay that a file changed",
                "/help                  Show this message",
            ]:
                self.tui.post("model_line", text=ln, style="dim")
        else:
            self.tui.post("model_line",
                text=f"Unknown command: {cmd!r}  (try /help)", style="red")
        return False


# ─── .modelrelay file helpers ────────────────────────────────────────────────

DOTFILE = ".modelrelay"


def load_dotfile(directory: Path) -> Config | None:
    """Load a Config from <directory>/.modelrelay, or None if absent/invalid."""
    p = directory / DOTFILE
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = Config.from_dict(data, working_dir=directory)
        return cfg
    except Exception:
        return None


def save_dotfile(cfg: Config, directory: Path):
    """Write cfg to <directory>/.modelrelay."""
    p = directory / DOTFILE
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    return p


_STATE_DIR = Path("~/.modelrelay").expanduser()
_LAST_DIR_FILE = _STATE_DIR / "last_working_dir"


def load_last_working_dir() -> Path | None:
    """Return the last-used working directory, or None if never recorded."""
    try:
        text = _LAST_DIR_FILE.read_text(encoding="utf-8").strip()
        p = Path(text)
        return p if p.is_dir() else None
    except Exception:
        return None


def save_last_working_dir(directory: Path):
    """Persist the working directory for next startup."""
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        _LAST_DIR_FILE.write_text(str(directory) + "\n", encoding="utf-8")
    except Exception:
        pass  # non-fatal


def _detect_default_backend() -> str | None:
    """Return the backend implied by available env vars, or None if ambiguous."""
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai    = bool(os.environ.get("OPENAI_API_KEY"))
    if has_anthropic and not has_openai:
        return "anthropic"
    if has_openai and not has_anthropic:
        return "openai-realtime"
    return None   # both or neither → no default


def _print_api_key_status(backend: str):
    _, api_key_var = BACKENDS[backend]
    if api_key_var:
        key_val = os.environ.get(api_key_var, "")
        if key_val:
            masked = key_val[:8] + "…" + key_val[-4:] if len(key_val) > 12 else "****"
            print(f"  {BR_GREEN}✓{RESET}  {api_key_var} is set  {DIM}({masked}){RESET}")
        else:
            print(f"  {BR_YELLOW}⚠{RESET}  {api_key_var} is not set in environment")
            print(f"  {DIM}  modelrelay will fail unless you export it before running.{RESET}")


def _show_config_summary(cfg: Config):
    """Print the confirmation box."""
    print(box_top("SESSION CONFIGURATION", color=BR_CYAN))
    items: list[tuple[str, str]] = [
        ("Backend",        cfg.backend),
        ("Model",          cfg.model),
        ("Working dir",    str(cfg.working_dir)),
        ("Log dir",        cfg.log_dir),
        ("Usage interval", f"{cfg.usage_interval_s}s"),
    ]
    if cfg.system_prompt_file:
        items.append(("Sys prompt file", cfg.system_prompt_file))
    elif cfg.system_prompt:
        preview = cfg.system_prompt[:40] + ("…" if len(cfg.system_prompt) > 40 else "")
        items.append(("Sys prompt", preview))
    else:
        items.append(("Sys prompt", "(built-in default)"))
    for k, v in items:
        print(box_row(f"{BOLD}{k:<20}{RESET}  {BR_WHITE}{v}{RESET}", color=BR_CYAN))
    print(box_bot(color=BR_CYAN))


def _prompt_settings(defaults: Config, modelrelay_path: "Path | None" = None) -> Config:
    """Ask the user all configuration questions, using `defaults` for pre-fills."""
    # ── Backend ──────────────────────────────────────────────────────────────
    env_default = _detect_default_backend()
    backend_choices = [
        ("anthropic",       "Anthropic (claude-opus-4-5)"),
        ("openai-realtime", "OpenAI Realtime (gpt-4o-realtime-preview)"),
        ("mock",            "Mock  (for testing, no API key needed)"),
    ]
    # Env detection takes priority; saved setting is only a fallback for when
    # neither or both API keys are present (i.e. env_default is None).
    keys = [k for k, _ in backend_choices]
    if env_default in keys:
        default_idx = keys.index(env_default)
    elif defaults.backend in keys:
        default_idx = keys.index(defaults.backend)
    else:
        default_idx = None  # no default → force explicit choice

    if default_idx is None:
        # No env vars detected and no saved setting: ask without a default
        print(f"\n  {BOLD}{WHITE}Backend:{RESET}")
        for i, (key, label) in enumerate(backend_choices):
            print(f"    {BR_BLACK}○{RESET} {BOLD}{i+1}{RESET}. {label}")
        while True:
            try:
                raw = input(_rl(f"\n  {CYAN}Choice (required): {RESET}")).strip()
                n = int(raw) - 1
                if 0 <= n < len(backend_choices):
                    backend = backend_choices[n][0]
                    break
            except ValueError:
                pass
            print(f"  {BR_RED}Please enter a number between 1 and {len(backend_choices)}.{RESET}")
    else:
        backend = prompt_choice("Backend:", backend_choices, default_idx=default_idx)

    print()
    _print_api_key_status(backend)

    # ── Model ─────────────────────────────────────────────────────────────────
    default_model = defaults.model if defaults.backend == backend else BACKENDS[backend][0]
    model = prompt_model(backend, default_model, modelrelay_path)

    print()
    print_section("OPTIONAL SETTINGS", BR_CYAN)

    # ── System prompt ─────────────────────────────────────────────────────────
    sys_prompt_file = prompt_optional(
        "System prompt file path",
        hint=defaults.system_prompt_file or "path to .txt",
    )
    if not sys_prompt_file and defaults.system_prompt_file:
        if confirm(f"  Keep saved file '{defaults.system_prompt_file}'?", default_yes=True):
            sys_prompt_file = defaults.system_prompt_file

    sys_prompt_inline = None
    if not sys_prompt_file:
        default_inline = defaults.system_prompt or ""
        sys_prompt_inline = prompt_optional(
            "System prompt inline text",
            hint="leave blank for built-in default",
        )
        if sys_prompt_inline is None and default_inline:
            preview = default_inline[:40] + ("…" if len(default_inline) > 40 else "")
            if confirm(f"  Keep saved inline prompt '{preview}'?", default_yes=True):
                sys_prompt_inline = default_inline

    # ── IDE context ───────────────────────────────────────────────────────────
    ide_ctx_file = prompt_optional(
        "IDE / project context file",
        hint=defaults.ide_context_file or "path to .txt",
    )
    if not ide_ctx_file and defaults.ide_context_file:
        if confirm(f"  Keep saved context file '{defaults.ide_context_file}'?", default_yes=True):
            ide_ctx_file = defaults.ide_context_file

    ide_ctx_inline = None
    if not ide_ctx_file:
        ide_ctx_inline = prompt_optional("IDE context inline text")
        if ide_ctx_inline is None and defaults.ide_context:
            preview = defaults.ide_context[:40] + ("…" if len(defaults.ide_context) > 40 else "")
            if confirm(f"  Keep saved IDE context '{preview}'?", default_yes=True):
                ide_ctx_inline = defaults.ide_context

    # ── Numeric options ───────────────────────────────────────────────────────
    print_section("Advanced Options")

    usage_interval = prompt_text(
        "Usage-status interval (seconds)",
        default=str(defaults.usage_interval_s),
    )
    try:
        usage_interval_f = float(usage_interval)
    except ValueError:
        usage_interval_f = defaults.usage_interval_s

    reconnect = prompt_text(
        "WebSocket reconnect attempts",
        default=str(defaults.reconnect_attempts),
    )
    try:
        reconnect_i = int(reconnect)
    except ValueError:
        reconnect_i = defaults.reconnect_attempts

    log_dir = prompt_text("Log directory", default=defaults.log_dir)

    return Config(
        backend            = backend,
        model              = model,
        system_prompt      = sys_prompt_inline,
        system_prompt_file = sys_prompt_file,
        ide_context        = ide_ctx_inline,
        ide_context_file   = ide_ctx_file,
        log_dir            = log_dir,
        usage_interval_s   = usage_interval_f,
        reconnect_attempts = reconnect_i,
        working_dir        = defaults.working_dir,
    )


# ─── Config wizard ───────────────────────────────────────────────────────────

def run_config_wizard(modelrelay_path: Path | None = None) -> tuple[Config, Sandbox]:
    clear_screen()
    print_banner()

    # ── Step 1: Working directory (always first) ──────────────────────────────
    print_section("WORKING DIRECTORY", BR_MAGENTA)
    print(f"  {DIM}File tools (read/write/stat/search) will be sandboxed to this directory.{RESET}\n")
    working_dir = prompt_directory("Working directory", default=load_last_working_dir())
    sandbox = Sandbox(working_dir)

    # ── Step 2: Check for saved .modelrelay ──────────────────────────────────
    saved = load_dotfile(working_dir)
    cfg: Config | None = None
    used_saved = False

    if saved is not None:
        print()
        print(f"  {BR_GREEN}✓{RESET}  Found {BOLD}{DOTFILE}{RESET} in {working_dir}")
        print()
        _show_config_summary(saved)
        print()
        if confirm(f"Use these saved settings?", default_yes=True):
            cfg = saved
            used_saved = True

    # ── Step 3: Full wizard (either no file, or user declined saved settings) ─
    if cfg is None:
        print()
        print_section("CONFIGURATION", BR_YELLOW)
        print(f"  {DIM}Set startup arguments for modelrelay. Press Enter to accept defaults.{RESET}\n")

        # Use saved config as defaults if it existed (user just declined to use
        # it verbatim), otherwise fall back to a blank Config for the directory.
        wizard_defaults = saved if saved is not None else Config(working_dir=working_dir)
        cfg = _prompt_settings(wizard_defaults, modelrelay_path=modelrelay_path)

    # ── Step 4: Confirmation (skipped when user already confirmed saved settings) ─
    if not used_saved:
        clear_screen()
        print_banner()
        print(f"\n  {BOLD}{BR_GREEN}Ready to launch!{RESET}\n")
        _show_config_summary(cfg)
        print()

        if not confirm("Launch modelrelay with these settings?", default_yes=True):
            print(f"\n  {BR_YELLOW}Aborted.{RESET}\n")
            sys.exit(0)

    # ── Step 5: Persist settings ──────────────────────────────────────────
    save_last_working_dir(working_dir)
    if not used_saved:
        # New settings: always offer to save
        if saved is not None:
            save_prompt = "Replace the existing .modelrelay file with these new settings?"
        else:
            save_prompt = f"Save these settings to {working_dir}/{DOTFILE} for next time?"
        if confirm(save_prompt, default_yes=True):
            dotfile_path = save_dotfile(cfg, working_dir)
            print(f"  {BR_GREEN}✓{RESET}  Settings saved to {DIM}{dotfile_path}{RESET}")
            print()

    return cfg, sandbox



# ─── Entry point ─────────────────────────────────────────────────────────────

async def _async_session(cfg: Config, sandbox: Sandbox,
                         modelrelay_path: Path, tui: TUI,
                         session_holder: list | None = None):
    """Full session coroutine, run in a background thread's event loop."""
    loop = asyncio.get_running_loop()
    session = ModelRelaySession(cfg, sandbox, modelrelay_path, tui)
    if session_holder is not None:
        session_holder.append(session)
    tui.set_async(loop, session._prompt_queue, session._confirm_queue)
    await session.run()


def main():
    import argparse

    ap = argparse.ArgumentParser(
        prog="modelrelay_tui",
        description="Interactive terminal host for the modelrelay subprocess.",
    )
    ap.add_argument(
        "modelrelay",
        metavar="MODELRELAY_PATH",
        help="Path to the modelrelay executable.",
    )
    args = ap.parse_args()
    modelrelay_path = Path(args.modelrelay).resolve()

    # Readline for the wizard's input() calls (plain terminal, pre-curses)
    try:
        import readline  # noqa: F401
    except ImportError:
        pass

    # ── Wizard runs in normal terminal mode ───────────────────────────────────
    try:
        cfg, sandbox = run_config_wizard(modelrelay_path=modelrelay_path)
    except (KeyboardInterrupt, EOFError):
        print(f"\n\n  {BR_YELLOW}Bye!{RESET}\n")
        return

    # ── Session runs inside curses ────────────────────────────────────────────
    exc_holder:  list[BaseException] = []
    err_lines:   list[str]           = []  # startup errors to echo after curses

    def _run_curses(stdscr):
        tui = TUI(stdscr)
        loop = asyncio.new_event_loop()
        # Holds the main Task once the event loop has created it, so the
        # curses thread can cancel it on Ctrl+C without stopping the loop.
        main_task_holder: list[asyncio.Task] = []

        session_holder: list["ModelRelaySession"] = []

        def _run_async():
            asyncio.set_event_loop(loop)
            try:
                async def _run_with_task():
                    task = asyncio.current_task()
                    if task is not None:
                        main_task_holder.append(task)
                    await _async_session(cfg, sandbox, modelrelay_path, tui,
                                        session_holder=session_holder)

                loop.run_until_complete(_run_with_task())
            except asyncio.CancelledError:
                pass   # clean cancellation from Ctrl+C — not an error
            except Exception as e:
                exc_holder.append(e)
                err_lines.append(f"Session error: {e}")
                tui.post("quit")
            finally:
                loop.close()

        t = threading.Thread(target=_run_async, daemon=True)
        t.start()

        try:
            tui.run_input_loop()
        except KeyboardInterrupt:
            # Kill the subprocess immediately (no waiting, no cmd.quit).
            if session_holder:
                session_holder[0].kill()
            # Then cancel the coroutine so run()'s finally block runs
            # (it will call stop(), which just reaps the already-dead process).
            if main_task_holder and not loop.is_closed():
                loop.call_soon_threadsafe(main_task_holder[0].cancel)
        finally:
            t.join(timeout=5)
            # Propagate startup errors to the post-curses display
            err_lines.extend(tui._startup_errors)

    try:
        _curses.wrapper(_run_curses)
    except KeyboardInterrupt:
        pass

    # Collect errors: startup errors from the TUI and runtime exceptions
    all_errors = list(err_lines)
    # _run_curses creates tui locally; we can't access it here directly.
    # startup_error messages are written into err_lines via _run_curses.
    for line in all_errors:
        print(f"  {BR_RED}{line}{RESET}")
    if all_errors:
        print()

    if exc_holder:
        raise exc_holder[0]

    print(f"\n  {BR_GREEN}Session complete.{RESET}\n")


if __name__ == "__main__":
    main()
