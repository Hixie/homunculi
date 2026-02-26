#!/usr/bin/env python3
"""
remote_codex.py

Runs a model-driven remote session over SSH, with OpenAI Realtime WebSocket API tool-calling.
Local process policy: this program must not exec any local program except `ssh`.
UI: split terminal (curses) with SSH I/O (top, live + rate status panels) and model notes (bottom).

Requirements:
  - Python 3.10+
  - pip install websockets

Environment:
  - OPENAI_API_KEY (required)
  - OPENAI_ORG_ID (optional)
  - OPENAI_PROJECT_ID (optional)

Usage:
  ./remote_codex.py --host mybox.example.com --user codex [--log-file session.log] "Install nginx and verify it's running"

Logging:
  - All model messages, SSH commands, and SSH terminal output are written to --log-file
    (defaults to remote_codex_<host>_<timestamp>.log in the current directory).

This program was mainly written by ChatGPT and Claude.
"""

from __future__ import annotations

import argparse
import asyncio
import curses
import json
import os
import pty
import re
import shlex
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Deque
from collections import deque
import textwrap

import websockets


REALTIME_MODEL_DEFAULT = "gpt-realtime"  # shown in OpenAI WS examples
REALTIME_URL = "wss://api.openai.com/v1/realtime?model={model}"

# Marker protocol to delimit command output on the SSH stream
_MARKER_PREFIX = "__RCODEX_MARKER__"
_MARKER_RE = re.compile(rf"{re.escape(_MARKER_PREFIX)}(?P<id>[A-Za-z0-9_-]+)__EXIT__(?P<code>\d+)__")


@dataclass
class CmdResult:
    ok: bool
    exit_code: int
    output: str
    blocked_reason: Optional[str] = None


class SessionLogger:
    """Asynchronous session logger for model messages and SSH activity."""

    def __init__(self, path: str) -> None:
        expanded = os.path.abspath(os.path.expanduser(path))
        directory = os.path.dirname(expanded)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        self.path = expanded
        self._file = open(self.path, "a", encoding="utf-8")
        self._lock = asyncio.Lock()
        self._closed = False

    async def log_model_message(self, text: str) -> None:
        await self._log("MODEL", text)

    async def log_ssh_command(self, text: str) -> None:
        await self._log("SSH_CMD", text)

    async def log_ssh_output(self, text: str) -> None:
        await self._log("SSH_OUT", text)

    async def _log(self, category: str, text: str) -> None:
        if self._closed or not text:
            return
        normalized = text.replace("\r", "")
        lines = normalized.splitlines()
        if not lines:
            if normalized:
                lines = [normalized]
            else:
                return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        async with self._lock:
            for line in lines:
                self._file.write(f"[{timestamp}] {category}: {line}\n")
            self._file.flush()

    async def close(self) -> None:
        async with self._lock:
            if not self._closed:
                self._file.close()
                self._closed = True


class SplitUI:
    """
    Simple curses UI:
      - Top pane: SSH stream (rolling) with STATUS and RATE LIMITS panels
      - Bottom pane: model notes stream (rolling)
    """

    def __init__(self) -> None:
        self._ssh_lines: Deque[str] = deque(maxlen=2000)
        self._note_lines: Deque[str] = deque(maxlen=2000)
        self._lock = asyncio.Lock()
        self._stdscr = None
        self._note_streaming = False
        self._status = "initializing"
        self._rate_status = "no rate limit data"

    async def add_ssh(self, text: str) -> None:
        async with self._lock:
            for line in text.splitlines():
                self._ssh_lines.append(line)

    async def add_note(self, text: str) -> None:
        async with self._lock:
            self._end_note_stream_locked()
            if not text:
                return
            for line in text.splitlines():
                self._note_lines.append(line)

    async def append_note_stream(self, text: str) -> None:
        if not text:
            return
        cleaned = text.replace("\r", "")
        if not cleaned:
            return
        async with self._lock:
            if not self._note_streaming:
                self._note_streaming = True
                if not self._note_lines or self._note_lines[-1] != "":
                    self._note_lines.append("")
            if not self._note_lines:
                self._note_lines.append("")
            segments = cleaned.split("\n")
            self._note_lines[-1] += segments[0]
            for segment in segments[1:]:
                self._note_lines.append(segment)

    async def end_note_stream(self) -> None:
        async with self._lock:
            self._end_note_stream_locked()

    async def set_status(self, text: str) -> None:
        sanitized = (text or "").replace("\n", " ").strip()
        if not sanitized:
            sanitized = "idle"
        async with self._lock:
            self._status = sanitized

    async def set_rate_status(self, text: str) -> None:
        sanitized = (text or "").replace("\n", " ").strip()
        if not sanitized:
            sanitized = "no rate limit data"
        async with self._lock:
            self._rate_status = sanitized

    def _end_note_stream_locked(self) -> None:
        if self._note_streaming:
            self._note_streaming = False
            if self._note_lines and self._note_lines[-1] != "":
                self._note_lines.append("")

    def _wrap_lines(self, lines: list[str], width: int) -> list[str]:
        if width <= 0:
            return lines
        wrapped: list[str] = []
        for line in lines:
            if not line:
                wrapped.append("")
                continue
            parts = textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False)
            if not parts:
                wrapped.append("")
                continue
            wrapped.extend(parts)
        return wrapped

    def _render(self) -> None:
        stdscr = self._stdscr
        if stdscr is None:
            return
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        split = max(3, int(h * 0.65))
        if split >= h:
            split = max(0, h - 1)
        top_h = split
        bot_h = h - split
        usable_width = max(1, w - 1)

        # Headers + status lines
        stdscr.addnstr(0, 0, "== SSH I/O ".ljust(w, "="), usable_width)
        status = self._status or "idle"
        stdscr.addnstr(1, 0, f" STATUS: {status} ".ljust(w), usable_width)
        has_rate_panel = top_h >= 3
        if has_rate_panel:
            rate_status = self._rate_status or "no rate limit data"
            stdscr.addnstr(2, 0, f" RATE LIMITS: {rate_status} ".ljust(w), usable_width)
        stdscr.addnstr(split, 0, "== MODEL NOTES ".ljust(w, "="), usable_width)

        reserved_rows = 2 + (1 if has_rate_panel else 0)
        ssh_view_h = max(0, top_h - reserved_rows)
        ssh_start_row = reserved_rows
        ssh_wrapped = self._wrap_lines(list(self._ssh_lines), usable_width)
        ssh_lines = ssh_wrapped[-ssh_view_h:]
        for i, line in enumerate(ssh_lines, start=ssh_start_row):
            stdscr.addnstr(i, 0, line, usable_width)

        # Render notes
        note_view_h = bot_h - 1
        note_wrapped = self._wrap_lines(list(self._note_lines), usable_width)
        note_lines = note_wrapped[-note_view_h:]
        for j, line in enumerate(note_lines, start=split + 1):
            stdscr.addnstr(j, 0, line, usable_width)

        stdscr.refresh()

    async def run(self, stop_evt: asyncio.Event) -> None:
        def _curses_main(stdscr) -> None:
            self._stdscr = stdscr
            curses.curs_set(0)
            stdscr.nodelay(True)
            stdscr.timeout(50)

            while not stop_evt.is_set():
                try:
                    _ = stdscr.getch()
                except Exception:
                    pass
                self._render()
                time.sleep(0.05)

        await asyncio.to_thread(curses.wrapper, _curses_main)


class SSHSession:
    """
    Maintains a single interactive SSH session using a local `ssh` process in a PTY.

    Local process policy: only spawns `ssh` (no other subprocesses).
    """

    def __init__(self, user: str, host: str, ui: SplitUI, logger: Optional[SessionLogger] = None) -> None:
        self.user = user
        self.host = host
        self.ui = ui
        self.logger = logger
        self.master_fd: Optional[int] = None
        self.pid: Optional[int] = None

        self._read_task: Optional[asyncio.Task] = None
        self._buffer = ""
        self._pending: dict[str, asyncio.Future[CmdResult]] = {}

    async def start(self) -> None:
        master_fd, slave_fd = pty.openpty()
        self.master_fd = master_fd

        # Start ssh in a PTY. -tt forces tty allocation.
        argv = ["ssh", "-tt", f"{self.user}@{self.host}"]
        env = {
            "HOME": os.environ.get("HOME", ""),
            "TERM": "dumb",
        }

        pid = os.fork()
        if pid == 0:
            # Child: connect slave PTY to stdio then exec ssh
            try:
                os.setsid()
                os.dup2(slave_fd, 0)
                os.dup2(slave_fd, 1)
                os.dup2(slave_fd, 2)
                os.close(master_fd)
                os.close(slave_fd)
                os.execvpe("ssh", argv, env)
            except Exception:
                os._exit(127)

        # Parent
        self.pid = pid
        os.close(slave_fd)
        await self.ui.add_ssh(f"[local] spawned ssh pid={pid} to {self.user}@{self.host}")

        self._read_task = asyncio.create_task(self._reader_loop())

    async def run_command(self, command: str, timeout_s: int = 120) -> CmdResult:
        """
        Runs a single remote command by writing it into the interactive shell and reading until marker.
        Returns combined output observed on the SSH stream during that command window.
        """
        if self.master_fd is None:
            return CmdResult(ok=False, exit_code=125, output="", blocked_reason="SSH session not started")

        cmd_id = f"{int(time.time()*1000)}_{os.getpid()}"
        marker = f"{_MARKER_PREFIX}{cmd_id}__EXIT__$?__"
        wrapped = f"\nset -o pipefail; {command}\nEC=$?; echo '{_MARKER_PREFIX}{cmd_id}__EXIT__'${{EC}}'__'\n"
        fut: asyncio.Future[CmdResult] = asyncio.get_running_loop().create_future()
        self._pending[cmd_id] = fut

        os.write(self.master_fd, wrapped.encode("utf-8", errors="ignore"))
        await self.ui.add_ssh(f"[model->ssh] {command}")
        if self.logger:
            await self.logger.log_ssh_command(command)

        try:
            return await asyncio.wait_for(fut, timeout=timeout_s)
        except asyncio.TimeoutError:
            self._pending.pop(cmd_id, None)
            return CmdResult(ok=False, exit_code=124, output="[timeout waiting for command completion]\n")

    async def _reader_loop(self) -> None:
        """
        Reads the PTY continuously, appends to UI, and resolves pending command futures when markers appear.
        """
        if self.master_fd is None:
            return

        loop = asyncio.get_running_loop()
        fd = self.master_fd

        while True:
            try:
                data = await loop.run_in_executor(None, os.read, fd, 4096)
                if not data:
                    await self.ui.add_ssh("[ssh] EOF")
                    break
            except OSError:
                break

            chunk = data.decode("utf-8", errors="ignore")
            if self.logger:
                await self.logger.log_ssh_output(chunk)
            self._buffer += chunk
            await self.ui.add_ssh(chunk)

            # Resolve markers; include buffered output for the command.
            while True:
                m = _MARKER_RE.search(self._buffer)
                if not m:
                    break
                cmd_id = m.group("id")
                exit_code = int(m.group("code"))

                # Split buffer: everything up to end of marker belongs to that command window.
                end = m.end()
                before = self._buffer[:end]
                after = self._buffer[end:]
                self._buffer = after

                fut = self._pending.pop(cmd_id, None)
                if fut and not fut.done():
                    # Provide the "before" text; callers can parse/ignore prompts etc.
                    fut.set_result(CmdResult(ok=(exit_code == 0), exit_code=exit_code, output=before))

    async def stop(self, force: bool = False) -> None:
        pid = self.pid
        sig = signal.SIGKILL if force else signal.SIGHUP

        if pid:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except Exception:
                pass

        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None

        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
            self._read_task = None

        if self._pending:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_result(CmdResult(ok=False, exit_code=255, output="", blocked_reason="SSH session closed"))
            self._pending.clear()

        if pid:
            wait_flag = 0 if force else os.WNOHANG
            try:
                await asyncio.to_thread(os.waitpid, pid, wait_flag)
            except ChildProcessError:
                pass
            except Exception:
                pass
        self.pid = None


class RealtimeAgent:
    """
    OpenAI Realtime WS session that:
      - sets up tool schema for `run_remote` and `finish`
      - sends initial user prompt
      - listens for tool calls in `response.done` outputs (function_call items)
      - streams output_text deltas to the "notes" pane (by convention)
      - detects idle responses (no tool usage / finish) and prompts the model to continue, logging the reminder in model notes
    """

    def __init__(self, prompt: str, ssh: SSHSession, ui: SplitUI, model: str, logger: Optional[SessionLogger] = None) -> None:
        self.prompt = prompt
        self.ssh = ssh
        self.ui = ui
        self.model = model
        self.logger = logger

        self.final_summary: Optional[str] = None
        self._stop_evt = asyncio.Event()
        self._model_streaming = False
        self._current_response_text = ""
        self._consecutive_idle_responses = 0
        self._waiting_for_continue_response = False

    @property
    def stop_evt(self) -> asyncio.Event:
        return self._stop_evt

    def _headers(self) -> dict[str, str]:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY env var is required")

        headers = {"Authorization": f"Bearer {key}"}
        org = os.environ.get("OPENAI_ORG_ID")
        proj = os.environ.get("OPENAI_PROJECT_ID")
        # These are optional; the WS docs show org/project as subprotocol options for browsers,
        # but for server-to-server we keep it minimal and rely on standard auth.
        if org:
            headers["OpenAI-Organization"] = org
        if proj:
            headers["OpenAI-Project"] = proj
        return headers

    async def run(self) -> None:
        url = REALTIME_URL.format(model=self.model)
        await self.ui.set_status("connecting to model")
        await self.ui.add_note(f"[openai] connecting {url}")

        try:
            async with websockets.connect(url, extra_headers=self._headers(), ping_interval=20) as ws:
                await self._wait_for_session_created(ws)
                await self.ui.set_status("configuring model session")
                await self._session_update(ws)
                await self._send_user_prompt(ws, self.prompt)
                await self._create_response(ws)
                await self.ui.set_status("waiting for model")

                async for raw in ws:
                    evt = json.loads(raw)

                    t = evt.get("type", "")
                    if not t:
                        await self.ui.add_note(f"[openai:event] no event")
                    if t == "response.function_call_arguments.delta":
                        pass  # Ignored
                    elif t == "response.function_call_arguments.done":
                        pass  # Ignored
                    elif t == "conversation.item.added":
                        pass  # Ignored
                    elif t == "conversation.item.done":
                        pass  # Ignored
                    elif t == "response.created":
                        pass  # Ignored
                    elif t == "response.content_part.added":
                        pass  # Ignored
                    elif t == "response.content_part.done":
                        pass  # Ignored
                    elif t == "response.output_item.done":
                        pass  # Ignored
                    elif t == "response.output_item.added":
                        pass  # Ignored
                    elif t == "response.output_text.done":
                        pass  # Ignored
                    elif t == "response.output_text.delta":
                        delta = evt.get("delta", "")
                        if delta:
                            if self._waiting_for_continue_response:
                                self._waiting_for_continue_response = False
                            if not self._model_streaming:
                                self._model_streaming = True
                                await self.ui.set_status("model responding")
                            await self.ui.append_note_stream(delta)
                            self._current_response_text += delta
                            if self.logger:
                                await self.logger.log_model_message(delta)
                    elif t == "response.done":
                        self._model_streaming = False
                        await self.ui.end_note_stream()
                        await self._handle_response_done(ws, evt)
                        if self._stop_evt.is_set():
                            break
                    elif t.startswith("rate_limits"):
                        await self.ui.set_rate_status(self._format_rate_limit_event(evt))
                    elif t == "error":
                        await self.ui.add_note(f"[openai:error] {json.dumps(evt, ensure_ascii=False)}")
                        await self.ui.set_status("model error")
                    else:
                        # Report other lifecycle events (session.created, response.created, etc.)
                        await self.ui.add_note(f"[openai:event] unknown event: {t}")
        finally:
            if not self._stop_evt.is_set():
                await self.ui.set_status("model connection closed")

    async def _session_update(self, ws) -> None:
        """
        Configure tools + instructions for the session.
        Tools schema is RealtimeFunctionTool: {type:"function", name, description, parameters}.
        """
        instructions = (
            "You are operating a remote Linux shell via an SSH session.\n"
            "You MUST use the tool run_remote to execute commands.\n"
            "This is a non-interactive session, there is no user to answer questions.\n"
            "When you have enough information, call finish with:\n"
            "  - summary: 3-8 bullet lines of what you did\n"
            "  - result: 1-3 lines stating the final outcome/state\n"
            "\n"
            "For logging purposes, write short 'working notes' as plain text while you operate.\n"
            "You do not have administrative access on the remote host, so calls to sudo will fail.\n"
            # "Do NOT request or reveal hidden chain-of-thought; provide concise operational notes.\n"
            # "Avoid any command that uses ssh/scp/sftp or pivots to another host.\n"
        )

        tools = [
            {
                "type": "function",
                "name": "run_remote",
                "description": "Run a shell command on the connected SSH host and return stdout/stderr and exit code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to run on the remote host."},
                        "timeout_s": {"type": "integer", "minimum": 1, "maximum": 600, "default": 120},
                    },
                    "required": ["command"],
                },
            },
            {
                "type": "function",
                "name": "finish",
                "description": "Indicate the task is complete and provide a short summary and final result for the user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "result": {"type": "string"},
                    },
                    "required": ["summary", "result"],
                },
            },
        ]

        event = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["text"],
                "instructions": instructions,
                "tools": tools,
            },
        }
        await ws.send(json.dumps(event))
        await self.ui.add_note("[openai] session configured (tools + instructions)")

    async def _send_user_prompt(self, ws, prompt: str) -> None:
        # Create a user message item.
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        }
        await ws.send(json.dumps(event))
        await self.ui.add_note("[user] prompt sent")

    async def _create_response(self, ws) -> None:
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {"output_modalities": ["text"]}
        }))

    async def _wait_for_session_created(self, ws) -> None:
        # Consume messages until we see session.created, then return.
        # If something else arrives first, keep it (basic handling/logging).
        while True:
            raw = await ws.recv()
            evt = json.loads(raw)
            t = evt.get("type", "")
            if t == "session.created":
                await self.ui.add_note("[openai] session.created received")
                return
            # Not session.created; keep minimal visibility.
            if t:
                await self.ui.add_note(f"[openai:event-before-ready] {t}")
        
    async def _handle_response_done(self, ws, evt: dict) -> None:
        """
        Tool calls arrive inside response.output items as type:function_call.
        We execute them and send back function_call_output items.
        """
        resp = evt.get("response", {}) or {}
        output = resp.get("output", []) or []

        last_response_text = self._current_response_text.strip()
        self._current_response_text = ""
        handled_tool_call = False

        # Also capture any final assistant text (if present as message output items).
        # For simplicity, we rely on output_text.delta streaming for notes and on `finish` to end.

        for item in output:
            if item.get("type") == "function_call":
                name = item.get("name")
                call_id = item.get("call_id")
                args_raw = item.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                except json.JSONDecodeError:
                    args = {}

                if name == "run_remote":
                    handled_tool_call = True
                    self._waiting_for_continue_response = False
                    cmd = str(args.get("command", ""))
                    timeout_s = int(args.get("timeout_s", 120))
                    cmd_label = self._summarize_command(cmd)
                    await self.ui.set_status(f"waiting for SSH: {cmd_label}")
                    res = await self.ssh.run_command(cmd, timeout_s=timeout_s)

                    await self.ui.set_status("sending SSH result to model")
                    payload = {
                        "ok": res.ok,
                        "exit_code": res.exit_code,
                        "blocked_reason": res.blocked_reason,
                        "output": res.output,
                    }
                    await self._send_tool_output(ws, call_id, payload)
                    await self._create_response(ws)
                    await self.ui.set_status("waiting for model")

                elif name == "finish":
                    handled_tool_call = True
                    self._waiting_for_continue_response = False
                    summary = str(args.get("summary", "")).strip()
                    result = str(args.get("result", "")).strip()
                    await self.ui.set_status("session complete")
                    self.final_summary = f"Summary:\n{summary}\n\nResult:\n{result}\n"
                    await self._send_tool_output(ws, call_id, {"ok": True})
                    if self.logger:
                        if summary:
                            await self.logger.log_model_message(f"[finish.summary]\n{summary}")
                        if result:
                            await self.logger.log_model_message(f"[finish.result]\n{result}")
                    self._stop_evt.set()
                    return

                else:
                    await self._send_tool_output(ws, call_id, {"ok": False, "error": f"Unknown tool: {name}"})
                    await self._create_response(ws)
                    await self.ui.set_status("waiting for model")

        if handled_tool_call:
            self._consecutive_idle_responses = 0
        elif not self._stop_evt.is_set():
            self._consecutive_idle_responses += 1
            if not self._waiting_for_continue_response:
                await self._prompt_model_to_continue(ws, last_response_text)
            else:
                await self.ui.add_note(
                    "[monitor] Continue prompt already pending; waiting for model response before prompting again"
                )

        if not self._stop_evt.is_set():
            await self.ui.set_status("waiting for model")

    async def _prompt_model_to_continue(self, ws, last_response_text: str) -> None:
        count = self._consecutive_idle_responses
        count_note = f" (idle response #{count})" if count else ""
        await self.ui.add_note(
            f"[monitor] No run_remote/finish() tool calls detected in the last response; prompting the model to continue{count_note}"
        )
        if last_response_text:
            excerpt = textwrap.shorten(last_response_text, width=200, placeholder="…")
            await self.ui.add_note(f"[monitor] Last response excerpt: {excerpt}")
        message = (
            "Please continue executing the steps you described. The previous response did not invoke run_remote or "
            "finish(). Use run_remote to run shell commands, and call finish() when the task is complete."
        )
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": message}],
            },
        }
        await ws.send(json.dumps(event))
        await self._create_response(ws)
        self._waiting_for_continue_response = True
        await self.ui.set_status("prompted model to continue")

    async def _send_tool_output(self, ws, call_id: str, output_obj: dict) -> None:
        # Return tool results via conversation.item.create with type=function_call_output.
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(output_obj, ensure_ascii=False),
            },
        }
        await ws.send(json.dumps(event))

    def _format_rate_limit_event(self, evt: dict) -> str:
        rate_limits = evt.get("rate_limits")
        if not isinstance(rate_limits, list):
            return f"[openai:rate_limits] {json.dumps(evt, ensure_ascii=False)}"

        parts = []
        for entry in rate_limits:
            if not isinstance(entry, dict):
                continue
            scope = entry.get("name") or entry.get("type") or entry.get("scope") or "unknown"
            limit = entry.get("limit")
            remaining = entry.get("remaining")
            reset_seconds = entry.get("reset_seconds")
            period = entry.get("period") or entry.get("window")

            details = []
            if remaining is not None and limit is not None:
                details.append(f"{remaining}/{limit} remaining")
            elif limit is not None:
                details.append(f"limit {limit}")
            if period:
                details.append(f"window {period}")
            if reset_seconds is not None:
                if isinstance(reset_seconds, (int, float)):
                    details.append(f"reset in {reset_seconds:.2f}s")
                else:
                    details.append(f"reset {reset_seconds}")

            if not details:
                details.append(json.dumps(entry, ensure_ascii=False))

            parts.append(f"{scope}: {', '.join(details)}")

        if not parts:
            return f"[openai:rate_limits] {json.dumps(evt, ensure_ascii=False)}"

        return "[openai:rate_limits] " + "; ".join(parts)

    def _summarize_command(self, command: str) -> str:
        text = (command or "").strip()
        if not text:
            return "<empty>"
        first_line = text.splitlines()[0].strip()
        if len(first_line) > 60:
            return first_line[:57] + "..."
        return first_line


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True, help="Remote SSH host")
    p.add_argument("--user", default="codex", help='SSH username (default: "codex")')
    p.add_argument("--model", default=REALTIME_MODEL_DEFAULT, help=f"Realtime model (default: {REALTIME_MODEL_DEFAULT})")
    p.add_argument("--log-file", help="Path to session log file (default: remote_codex_<host>_<timestamp>.log)")
    p.add_argument("prompt", help="Task prompt for the model")
    return p.parse_args(argv)


async def main_async(argv: list[str]) -> int:
    args = parse_args(argv)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    host_token = re.sub(r"[^A-Za-z0-9._-]", "_", args.host)
    default_log = f"remote_codex_{host_token}_{timestamp}.log"
    session_logger = SessionLogger(args.log_file or default_log)

    ui = SplitUI()
    ssh = SSHSession(user=args.user, host=args.host, ui=ui, logger=session_logger)
    agent = RealtimeAgent(prompt=args.prompt, ssh=ssh, ui=ui, model=args.model, logger=session_logger)

    stop_ui = asyncio.Event()

    loop = asyncio.get_running_loop()
    def _request_stop(signame: str):
        stop_ui.set()
        agent.stop_evt.set()
        try:
            loop.create_task(ui.set_status(f"stopping ({signame})"))
        except RuntimeError:
            pass
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            loop.add_signal_handler(sig, _request_stop, sig.name)
        except NotImplementedError:
            pass

    await ui.set_status(f"connecting to SSH: {args.user}@{args.host}")
    await ui.add_note(f"[log] session log: {session_logger.path}")

    ui_task: Optional[asyncio.Task] = None
    agent_task: Optional[asyncio.Task] = None
    stop_wait_task: Optional[asyncio.Task] = None

    try:
        await ssh.start()

        ui_task = asyncio.create_task(ui.run(stop_ui))
        agent_task = asyncio.create_task(agent.run())
        stop_wait_task = asyncio.create_task(stop_ui.wait())

        done, _ = await asyncio.wait({agent_task, stop_wait_task}, return_when=asyncio.FIRST_COMPLETED)

        if stop_wait_task in done and agent_task and not agent_task.done():
            agent_task.cancel()
    finally:
        stop_ui.set()
        agent.stop_evt.set()

        if stop_wait_task:
            try:
                await stop_wait_task
            except asyncio.CancelledError:
                pass

        try:
            await ssh.stop(force=True)
        except Exception:
            pass

        if agent_task:
            if not agent_task.done():
                agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        if ui_task:
            try:
                await asyncio.wait_for(ui_task, timeout=1.0)
            except Exception:
                pass

        await session_logger.close()

    if agent.final_summary:
        print(agent.final_summary)
        print(f"Session log saved to: {session_logger.path}")
        return 0

    print("Session ended without a finish() call from the model.")
    print(f"Session log saved to: {session_logger.path}")
    return 2


def main() -> int:
    try:
        return asyncio.run(main_async(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        print(f"\nFatal: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
