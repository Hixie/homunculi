"""OpenAI Realtime WebSocket backend."""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from typing import AsyncIterator

import websockets

from .base import (BackendEvent, BackendSession, CanonicalToolSchema,
                   EventType, NormalizedUsage)

log = logging.getLogger(__name__)

WS_URL_TEMPLATE = "wss://api.openai.com/v1/realtime?model={model}"



class OpenAIRealtimeBackend(BackendSession):
    """
    Persistent WebSocket backend for the OpenAI Realtime API.

    Session lifecycle:
      connect()      → opens WS, sends session.update (tools + system prompt)
      send_turn()    → conversation.item.create + response.create
      submit_tool_result() → conversation.item.create (function_call_output) + response.create
      inject_note()  → conversation.item.create (system message)
      close()        → close WS

    Events emitted via events():
      CONNECTED, TEXT_DELTA, TOOL_CALL, TURN_DONE, RATE_LIMITED, ERROR, DISCONNECTED
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-realtime-preview",
                 system_prompt: str = "",
                 tools: list[CanonicalToolSchema] | None = None,
                 max_reconnect_attempts: int = 3,
                 logger=None):
        self._api_key              = api_key
        self._model                = model
        self._system_prompt        = system_prompt
        self._tools                = tools or []
        self._max_reconnect        = max_reconnect_attempts
        self._logger               = logger
        self._ws                   = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._closed               = False
        self._recv_task: asyncio.Task | None = None

    # ── connect ──────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        url = WS_URL_TEMPLATE.format(model=self._model)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        attempt = 0
        delay   = 1.0
        while True:
            try:
                self._ws = await websockets.connect(url, additional_headers=headers)
                break
            except Exception as exc:
                attempt += 1
                if attempt >= self._max_reconnect:
                    raise RuntimeError(
                        f"Failed to connect to OpenAI Realtime after "
                        f"{attempt} attempt(s): {exc}") from exc
                await asyncio.sleep(delay)
                delay *= 2

        # Configure session: tools + system prompt
        session_cfg: dict = {
            "type":             "realtime",
            "output_modalities": ["text"],
            "instructions":     self._system_prompt,
            "tools":            self.translate_tools(self._tools),
            "tool_choice":      "auto",
        }
        await self._send({"type": "session.update", "session": session_cfg})

        # Start background receive loop
        self._recv_task = asyncio.create_task(self._recv_loop())

        await self._event_queue.put(BackendEvent(EventType.CONNECTED, {"model": self._model}))

    # ── events ────────────────────────────────────────────────────────────────

    async def events(self) -> AsyncIterator[BackendEvent]:
        while not self._closed:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                continue

    # ── send_turn ─────────────────────────────────────────────────────────────

    async def send_turn(self, text: str) -> None:
        await self._send({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": text}]},
        })
        await self._send({"type": "response.create"})

    # ── submit_tool_result ────────────────────────────────────────────────────

    async def submit_tool_result(self, call_id: str, result: str) -> None:
        await self._send({
            "type": "conversation.item.create",
            "item": {"type":    "function_call_output",
                     "call_id": call_id,
                     "output":  result},
        })
        await self._send({"type": "response.create"})

    # ── inject_note ───────────────────────────────────────────────────────────

    async def inject_note(self, text: str) -> None:
        await self._send({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "system",
                     "content": [{"type": "input_text", "text": text}]},
        })

    # ── close ─────────────────────────────────────────────────────────────────

    async def close(self) -> None:
        self._closed = True
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

    # ── translate_tools ───────────────────────────────────────────────────────

    @classmethod
    def translate_tools(cls, tools: list[CanonicalToolSchema]) -> list[dict]:
        return [
            {
                "type":        "function",
                "name":        t.name,
                "description": t.description,
                "parameters":  t.parameters,
            }
            for t in tools
        ]

    @classmethod
    def list_models(cls, api_key: str) -> list[str]:
        """Fetch realtime-capable model IDs from the OpenAI API."""
        req = urllib.request.Request(
            "https://api.openai.com/v1/models",
            method="GET",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            return sorted(
                m["id"] for m in data.get("data", [])
                if "realtime" in m["id"]
            )
        except Exception:
            return []

    # ── internal ──────────────────────────────────────────────────────────────

    async def _send(self, frame: dict) -> None:
        if self._ws is None:
            return
        try:
            await self._ws.send(json.dumps(frame))
        except Exception as exc:
            await self._event_queue.put(BackendEvent(
                EventType.ERROR, {"message": str(exc), "fatal": False}))

    async def _recv_loop(self) -> None:
        """Receive frames from the WebSocket and translate to BackendEvents."""
        # Accumulate per-call_id tool-call argument strings
        pending_calls: dict[str, dict] = {}   # call_id → {name, args_str}
        reconnect_attempts = 0
        delay = 1.0

        while not self._closed:
            try:
                raw = await self._ws.recv()
            except Exception as exc:
                # Connection lost — attempt reconnect
                if self._closed:
                    return
                reconnect_attempts += 1
                if reconnect_attempts > self._max_reconnect:
                    await self._event_queue.put(BackendEvent(
                        EventType.ERROR,
                        {"message": f"WebSocket disconnected after {reconnect_attempts} reconnect attempts: {exc}",
                         "fatal": True}))
                    return
                await self._event_queue.put(BackendEvent(EventType.DISCONNECTED, {}))
                await asyncio.sleep(delay)
                delay *= 2
                try:
                    url = WS_URL_TEMPLATE.format(model=self._model)
                    reconnect_headers = {
                        "Authorization": f"Bearer {self._api_key}",
                    }
                    self._ws = await websockets.connect(url, additional_headers=reconnect_headers)
                    # Re-configure session after reconnect (mirrors connect())
                    reconnect_session: dict = {
                        "type":             "realtime",
                        "output_modalities": ["text"],
                        "instructions":     self._system_prompt,
                        "tools":            self.translate_tools(self._tools),
                        "tool_choice":      "auto",
                    }
                    await self._send({
                        "type": "session.update",
                        "session": reconnect_session,
                    })
                    reconnect_attempts = 0
                    delay = 1.0
                except Exception:
                    pass
                continue

            reconnect_attempts = 0
            delay = 1.0

            try:
                frame = json.loads(raw)
            except json.JSONDecodeError as exc:
                if self._logger:
                    self._logger.log_wire_rx_error("openai-realtime", raw, str(exc))
                continue

            etype = frame.get("type", "")
            disposition = "ignored"

            if etype == "response.output_text.delta":
                disposition = "processed"
                text = frame.get("delta", "")
                await self._event_queue.put(
                    BackendEvent(EventType.TEXT_DELTA, {"text": text}))

            elif etype == "response.output_item.added":
                disposition = "processed"
                # The function name is delivered here, before any argument deltas.
                # Seed pending_calls so that delta frames (which omit the name)
                # can look it up by call_id.
                item = frame.get("item", {})
                if item.get("type") == "function_call":
                    call_id = item.get("call_id", "")
                    name    = item.get("name", "")
                    if call_id:
                        pending_calls[call_id] = {"name": name, "args_str": ""}

            elif etype == "response.function_call_arguments.delta":
                disposition = "processed"
                call_id = frame.get("call_id", "")
                delta   = frame.get("delta", "")
                if call_id not in pending_calls:
                    # Fallback: name may still appear on the delta frame in some
                    # API versions; use it if present.
                    pending_calls[call_id] = {"name": frame.get("name", ""), "args_str": ""}
                pending_calls[call_id]["args_str"] += delta

            elif etype == "response.function_call_arguments.done":
                disposition = "processed"
                call_id = frame.get("call_id", "")
                tc = pending_calls.pop(call_id, {"name": "", "args_str": "{}"})
                try:
                    args = json.loads(tc["args_str"])
                except json.JSONDecodeError:
                    args = {"_error": "invalid JSON in tool arguments",
                            "_raw":   tc["args_str"]}
                await self._event_queue.put(BackendEvent(EventType.TOOL_CALL, {
                    "call_id": call_id,
                    "name":    tc["name"],
                    "args":    args,
                }))

            elif etype == "response.done":
                disposition = "processed"
                usage_raw = frame.get("response", {}).get("usage", {})
                usage = NormalizedUsage(
                    input_tokens  = usage_raw.get("input_tokens", 0),
                    output_tokens = usage_raw.get("output_tokens", 0),
                )
                await self._event_queue.put(
                    BackendEvent(EventType.TURN_DONE, {"usage": usage}))

            elif etype == "rate_limits.updated":
                disposition = "processed"
                # rate_limits.updated is informational; the server sends it after
                # every response.done to report current quota usage.  Only emit
                # RATE_LIMITED when requests are genuinely exhausted (remaining == 0
                # with a known reset time) so we don't overwrite the "idle" state
                # that TURN_DONE just set.
                limits = frame.get("rate_limits", [])
                for lim in limits:
                    if lim.get("name") == "requests" and lim.get("remaining", 1) == 0:
                        retry_after = lim.get("reset_seconds")
                        if retry_after is not None:
                            await self._event_queue.put(
                                BackendEvent(EventType.RATE_LIMITED,
                                             {"retry_after_s": retry_after}))
                        break

            elif etype == "error":
                disposition = "processed"
                err = frame.get("error", {})
                msg = err.get("message", str(frame))
                # OpenAI Realtime errors are generally non-fatal unless we
                # can't continue (fatal is inferred from error code).
                fatal = err.get("code", "") in ("invalid_api_key", "model_not_found")
                await self._event_queue.put(
                    BackendEvent(EventType.ERROR, {"message": msg, "fatal": fatal}))

            # All other frame types (session.created, session.updated,
            # conversation.item.created, conversation.item.added,
            # conversation.item.done, response.created, etc.) are
            # informational and not surfaced as BackendEvents — disposition stays "ignored".

            if self._logger:
                self._logger.log_wire_rx("openai-realtime", frame, disposition)
