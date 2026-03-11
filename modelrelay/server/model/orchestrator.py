"""Backend event loop; TEXT_DELTA → model.text_delta; tool dispatch."""
from __future__ import annotations

import asyncio

from ..backends.base import EventType
from ..stdio.protocol import (activity, error_msg, model_text_delta,
                               usage_msg)
from ..tools.context import ToolContext


class Orchestrator:
    def __init__(self, backend, stdio, registry, usage, pending,
                 usage_interval_s: float = 30.0,
                 logger=None):
        self._backend         = backend
        self._stdio           = stdio
        self._registry        = registry
        self._usage           = usage
        self._pending         = pending
        self._usage_interval  = usage_interval_s
        self._logger          = logger
        self._ctx             = ToolContext(stdio, pending)
        self._poll_task: asyncio.Task | None = None
        self._last_sent_usage: dict | None = None

    async def _send_usage_if_changed(self) -> None:
        """Send status.usage only when the payload differs from the last sent."""
        data = self._usage.as_dict()
        if data != self._last_sent_usage:
            self._last_sent_usage = data
            await self._stdio.send(usage_msg(data))

    def _rx(self, event_type: str, data: dict):
        """Log a BackendEvent received from the model (no-op if no logger)."""
        if self._logger:
            self._logger.log_model_rx(event_type, data)

    def _tx(self, kind: str, **fields):
        """Log a message sent to the model (no-op if no logger)."""
        if self._logger:
            self._logger.log_model_tx(kind, **fields)

    async def run(self) -> None:
        self._poll_task = asyncio.create_task(self._usage_poll_loop())
        try:
            async for event in self._backend.events():
                await self._dispatch(event)
        finally:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

    async def _usage_poll_loop(self) -> None:
        """Emit status.usage on a regular interval, skipping if unchanged."""
        while True:
            await asyncio.sleep(self._usage_interval)
            await self._send_usage_if_changed()

    async def _dispatch(self, event) -> None:
        match event.type:
            case EventType.CONNECTED:
                self._rx("CONNECTED", event.data)
                await self._stdio.send(activity("idle",
                    f"Connected ({event.data.get('model', '?')})"))

            case EventType.TEXT_DELTA:
                self._rx("TEXT_DELTA", event.data)
                text = event.data.get("text", "")
                await self._stdio.send(activity("generating", "Generating…"))
                if text:
                    await self._stdio.send(model_text_delta(text))

            case EventType.TOOL_CALL:
                self._rx("TOOL_CALL", event.data)
                name = event.data["name"]
                await self._stdio.send(activity("waiting_for_tool",
                    f"Calling: {name}(…)"))
                result = await self._invoke_tool(event.data)
                self._tx("tool_result",
                         call_id=event.data["call_id"],
                         name=name,
                         args=event.data.get("args"),
                         result=result)
                await self._backend.submit_tool_result(event.data["call_id"], result)
                await self._stdio.send(activity("generating",
                    f"{name} returned; continuing…"))

            case EventType.TURN_DONE:
                self._rx("TURN_DONE", {
                    "input_tokens":  event.data["usage"].input_tokens,
                    "output_tokens": event.data["usage"].output_tokens,
                })
                self._usage.update(event.data["usage"])
                await self._send_usage_if_changed()
                await self._stdio.send(activity("idle", "Turn complete"))

            case EventType.RATE_LIMITED:
                self._rx("RATE_LIMITED", event.data)
                s = event.data.get("retry_after_s")
                desc = f"Rate limited{f'; retrying in {s:.0f}s' if s else ''}"
                await self._stdio.send(activity("rate_limited", desc))

            case EventType.ERROR:
                self._rx("ERROR", event.data)
                await self._stdio.send(error_msg(
                    "BACKEND_ERROR", event.data.get("message", ""),
                    event.data.get("fatal", False)))

    async def on_prompt(self, envelope: dict) -> None:
        text = envelope["payload"]["text"]
        self._tx("send_turn", text=text)
        await self._backend.send_turn(text)

    async def on_invalidate(self, envelope: dict) -> None:
        resource   = envelope["payload"].get("resource", "")
        start_line = envelope["payload"].get("start_line")
        reason     = envelope["payload"].get("reason", "")
        if start_line:
            note = (f"Lines {start_line}–… of '{resource}' have been modified "
                    "externally. Do not retry your previous change; the user "
                    "may have chosen not to apply it. Re-read that region if "
                    "you need to make further edits.")
        else:
            note = (f"'{resource}' has been modified externally. Do not retry "
                    "your previous change; the user may have chosen not to "
                    "apply it. Re-read the file if you need to make further edits.")
        if reason:
            note += f" Reason: {reason}"
        self._tx("inject_note", text=note)
        await self._backend.inject_note(note)

    async def _invoke_tool(self, data: dict) -> str:
        try:
            return await self._registry.invoke(
                data["name"], data["args"], self._ctx)
        except Exception as exc:
            return f"ERROR: tool '{data['name']}' raised: {exc}"
