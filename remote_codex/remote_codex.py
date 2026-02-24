#!/usr/bin/env python3
"""
remote_codex.py

Runs a model-driven remote session over SSH, with OpenAI Realtime WebSocket API tool-calling.
Local process policy: this program must not exec any local program except `ssh`.
UI: split terminal (curses) with SSH I/O (top) and model notes (bottom).

Requirements:
  - Python 3.10+
  - pip install websockets

Environment:
  - OPENAI_API_KEY (required)
  - OPENAI_ORG_ID (optional)
  - OPENAI_PROJECT_ID (optional)

Usage:
  ./remote_codex.py --host mybox.example.com --user codex "Install nginx and verify it's running"

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


class SplitUI:
    """
    Simple curses UI:
      - Top pane: SSH stream (rolling)
      - Bottom pane: model notes stream (rolling)
    """

    def __init__(self) -> None:
        self._ssh_lines: Deque[str] = deque(maxlen=2000)
        self._note_lines: Deque[str] = deque(maxlen=2000)
        self._lock = asyncio.Lock()
        self._stdscr = None

    async def add_ssh(self, text: str) -> None:
        async with self._lock:
            for line in text.splitlines():
                self._ssh_lines.append(line)

    async def add_note(self, text: str) -> None:
        async with self._lock:
            for line in text.splitlines():
                self._note_lines.append(line)

    def _render(self) -> None:
        stdscr = self._stdscr
        if stdscr is None:
            return
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        split = max(3, int(h * 0.65))
        top_h = split
        bot_h = h - split

        # Headers
        stdscr.addnstr(0, 0, " SSH I/O ".ljust(w, "="), w - 1)
        stdscr.addnstr(split, 0, " MODEL NOTES ".ljust(w, "="), w - 1)

        # Render SSH lines
        ssh_view_h = top_h - 1
        ssh_lines = list(self._ssh_lines)[-ssh_view_h:]
        for i, line in enumerate(ssh_lines, start=1):
            stdscr.addnstr(i, 0, line, w - 1)

        # Render notes
        note_view_h = bot_h - 1
        note_lines = list(self._note_lines)[-note_view_h:]
        for j, line in enumerate(note_lines, start=split + 1):
            stdscr.addnstr(j, 0, line, w - 1)

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

    def __init__(self, user: str, host: str, ui: SplitUI) -> None:
        self.user = user
        self.host = host
        self.ui = ui
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
            # "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            # "USER": os.environ.get("USER", ""),
            # "LOGNAME": os.environ.get("LOGNAME", ""),
            # "SSH_AUTH_SOCK": os.environ.get("SSH_AUTH_SOCK", ""),
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

    async def stop(self) -> None:
        if self.pid:
            try:
                os.kill(self.pid, signal.SIGHUP)
            except Exception:
                pass
        if self._read_task:
            self._read_task.cancel()


class RealtimeAgent:
    """
    OpenAI Realtime WS session that:
      - sets up tool schema for `run_remote` and `finish`
      - sends initial user prompt
      - listens for tool calls in `response.done` outputs (function_call items)
      - streams output_text deltas to the "notes" pane (by convention)
    """

    def __init__(self, prompt: str, ssh: SSHSession, ui: SplitUI, model: str) -> None:
        self.prompt = prompt
        self.ssh = ssh
        self.ui = ui
        self.model = model

        self.final_summary: Optional[str] = None
        self._stop_evt = asyncio.Event()

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
        await self.ui.add_note(f"[openai] connecting {url}")

        async with websockets.connect(url, extra_headers=self._headers(), ping_interval=20) as ws:
            await self._wait_for_session_created(ws)
            await self._session_update(ws)
            await self._send_user_prompt(ws, self.prompt)
            await self._create_response(ws)

            async for raw in ws:
                evt = json.loads(raw)

                t = evt.get("type", "")
                if not t:
                    await self.ui.add_note(f"[openai:event] no event")
                if t == "response.function_call_arguments.delta":
                    pass # Ignored
                if t == "response.function_call_arguments.done":
                    pass # Ignored
                elif t == "conversation.item.added":
                    pass # Ignored
                elif t == "conversation.item.done":
                    pass # Ignored
                elif t == "response.content_part.done":
                    pass # Ignored
                elif t == "response.output_item.done":
                    pass # Ignored
                elif t == "response.output_text.done":
                    pass # Ignored
                elif t == "response.output_text.delta":
                    # The docs identify this as the streaming text delta event.
                    delta = evt.get("delta", "")
                    if delta:
                        await self.ui.add_note(delta)
                elif t == "response.done":
                    await self._handle_response_done(ws, evt)
                    if self._stop_evt.is_set():
                        break
                elif t.startswith("rate_limits"):
                    await self.ui.add_note(self._format_rate_limit_event(evt))
                elif t == "error":
                    await self.ui.add_note(f"[openai:error] {json.dumps(evt, ensure_ascii=False)}")
                else:
                    # Report other lifecycle events (session.created, response.created, etc.)
                    await self.ui.add_note(f"[openai:event] unknown event: {t}")

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
                    cmd = str(args.get("command", ""))
                    timeout_s = int(args.get("timeout_s", 120))
                    res = await self.ssh.run_command(cmd, timeout_s=timeout_s)

                    payload = {
                        "ok": res.ok,
                        "exit_code": res.exit_code,
                        "blocked_reason": res.blocked_reason,
                        "output": res.output,
                    }
                    await self._send_tool_output(ws, call_id, payload)
                    await self._create_response(ws)

                elif name == "finish":
                    summary = str(args.get("summary", "")).strip()
                    result = str(args.get("result", "")).strip()
                    self.final_summary = f"Summary:\n{summary}\n\nResult:\n{result}\n"
                    await self._send_tool_output(ws, call_id, {"ok": True})
                    self._stop_evt.set()
                    return

                else:
                    await self._send_tool_output(ws, call_id, {"ok": False, "error": f"Unknown tool: {name}"})
                    await self._create_response(ws)

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


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True, help="Remote SSH host")
    p.add_argument("--user", default="codex", help='SSH username (default: "codex")')
    p.add_argument("--model", default=REALTIME_MODEL_DEFAULT, help=f"Realtime model (default: {REALTIME_MODEL_DEFAULT})")
    p.add_argument("prompt", help="Task prompt for the model")
    return p.parse_args(argv)


async def main_async(argv: list[str]) -> int:
    args = parse_args(argv)

    ui = SplitUI()
    ssh = SSHSession(user=args.user, host=args.host, ui=ui)
    agent = RealtimeAgent(prompt=args.prompt, ssh=ssh, ui=ui, model=args.model)

    stop_ui = asyncio.Event()

    # Ensure signals restore UI + exit cleanly.
    loop = asyncio.get_running_loop()
    def _request_stop(signame: str):
        stop_ui.set()
        agent.stop_evt.set()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            loop.add_signal_handler(sig, _request_stop, sig.name)
        except NotImplementedError:
            # Some platforms/contexts don't support add_signal_handler (rare on Linux).
            pass

    await ssh.start()

    ui_task = asyncio.create_task(ui.run(stop_ui))
    agent_task = asyncio.create_task(agent.run())

    done, pending = await asyncio.wait({agent_task}, return_when=asyncio.FIRST_COMPLETED)

    stop_ui.set()
    try:
        await ssh.stop()
    except Exception:
        pass

    try:
        await asyncio.wait_for(ui_task, timeout=1.0)
    except Exception:
        pass

    # Print final summary to user (outside curses UI).
    if agent.final_summary:
        print(agent.final_summary)
        return 0

    print("Session ended without a finish() call from the model.")
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
