"""Anthropic HTTP/SSE backend."""
from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import AsyncIterator

from .base import (BackendEvent, BackendSession, CanonicalToolSchema,
                   EventType, NormalizedUsage)

ANTHROPIC_API    = "https://api.anthropic.com/v1/messages"
MAX_5XX_RETRIES  = 3


class AnthropicBackend(BackendSession):
    def __init__(self, api_key: str, model: str = "claude-opus-4-5",
                 system_prompt: str = "",
                 tools: list[CanonicalToolSchema] | None = None,
                 retry_delay_s: float = 1.0,
                 logger=None):
        self._api_key        = api_key
        self._model          = model
        self._system         = system_prompt
        self._tools_schemas  = tools or []
        self._retry_delay_s  = retry_delay_s
        self._logger         = logger
        self._history: list[dict] = []
        self._queue: asyncio.Queue = asyncio.Queue()
        self._closed = False

    async def connect(self) -> None:
        await self._queue.put(BackendEvent(EventType.CONNECTED, {"model": self._model}))

    async def events(self) -> AsyncIterator[BackendEvent]:
        while not self._closed:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                if event.type == EventType.ASSISTANT_CONTENT:
                    # Internal event: append assistant turn to history, don't yield
                    content = event.data.get("content", [])
                    if content:
                        self._history.append({"role": "assistant", "content": content})
                    continue
                yield event
            except asyncio.TimeoutError:
                continue

    async def send_turn(self, text: str) -> None:
        self._history.append({"role": "user", "content": text})
        asyncio.create_task(self._do_request())

    async def _do_request(self, attempt: int = 0) -> None:
        body = {
            "model":    self._model,
            "system":   self._system,
            "messages": self._history,
            "tools":    self.translate_tools(self._tools_schemas),
            "stream":   True,
            "max_tokens": 4096,
        }
        main_loop = asyncio.get_running_loop()
        try:
            await main_loop.run_in_executor(
                None, self._stream_response, json.dumps(body).encode(), main_loop, self._logger)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                try:
                    retry_after = float(exc.headers.get("Retry-After", 0))
                except (TypeError, ValueError):
                    retry_after = None
                await self._queue.put(BackendEvent(
                    EventType.RATE_LIMITED, {"retry_after_s": retry_after}))
            elif exc.code >= 500:
                if attempt < MAX_5XX_RETRIES - 1:
                    await asyncio.sleep(self._retry_delay_s * (2 ** attempt))
                    await self._do_request(attempt + 1)
                else:
                    await self._queue.put(BackendEvent(
                        EventType.ERROR,
                        {"message": f"HTTP {exc.code}: {exc.reason}", "fatal": True}))
            else:
                await self._queue.put(BackendEvent(
                    EventType.ERROR,
                    {"message": f"HTTP {exc.code}: {exc.reason}", "fatal": True}))
        except Exception as exc:
            await self._queue.put(BackendEvent(
                EventType.ERROR, {"message": str(exc), "fatal": True}))

    def _stream_response(self, body: bytes, main_loop: asyncio.AbstractEventLoop,
                          logger=None) -> None:
        """Run in a thread-pool executor. Posts events back via call_soon_threadsafe."""
        def _put(event: BackendEvent) -> None:
            main_loop.call_soon_threadsafe(self._queue.put_nowait, event)
        def _log_frame(frame: dict, disposition: str) -> None:
            if logger:
                main_loop.call_soon_threadsafe(
                    logger.log_wire_rx, "anthropic", frame, disposition)
        def _log_frame_error(raw: str, error: str) -> None:
            if logger:
                main_loop.call_soon_threadsafe(logger.log_wire_rx_error, "anthropic", raw, error)

        req = urllib.request.Request(ANTHROPIC_API, data=body, method="POST", headers={
            "x-api-key":         self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        })
        try:
            with urllib.request.urlopen(req) as resp:
                # Read rate-limit headers
                requests_remaining       = _int_header(resp, "anthropic-ratelimit-requests-remaining")
                tokens_remaining_per_min = _int_header(resp, "anthropic-ratelimit-tokens-remaining")
                reset_at                 = _str_header(resp, "anthropic-ratelimit-requests-reset")

                input_tokens  = 0
                output_tokens = 0
                tool_calls:   dict[str, dict] = {}
                current_block = None

                # Accumulate the full assistant response for history
                assistant_text_blocks: list[str] = []
                current_text: list[str] = []

                for raw in resp:
                    line = raw.decode("utf-8").rstrip("\n")
                    if line.startswith(":") or not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError as exc:
                        _log_frame_error(data_str, str(exc))
                        continue

                    etype = data.get("type", "")
                    disposition = "ignored"

                    if etype == "message_start":
                        disposition = "processed"
                        usage = data.get("message", {}).get("usage", {})
                        input_tokens = usage.get("input_tokens", 0)

                    elif etype == "content_block_start":
                        disposition = "processed"
                        block = data.get("content_block", {})
                        current_block = block
                        if block.get("type") == "tool_use":
                            tool_calls[block["id"]] = {"name": block["name"], "args_str": ""}
                        elif block.get("type") == "text":
                            current_text = []

                    elif etype == "content_block_delta":
                        disposition = "processed"
                        delta = data.get("delta", {})
                        dtype = delta.get("type", "")
                        if dtype == "text_delta":
                            text = delta.get("text", "")
                            current_text.append(text)
                            _put(BackendEvent(EventType.TEXT_DELTA, {"text": text}))
                        elif dtype == "input_json_delta" and current_block:
                            bid = current_block.get("id", "")
                            if bid in tool_calls:
                                tool_calls[bid]["args_str"] += delta.get("partial_json", "")

                    elif etype == "content_block_stop" and current_block:
                        disposition = "processed"
                        btype = current_block.get("type", "")
                        if btype == "text":
                            assistant_text_blocks.append("".join(current_text))
                            current_text = []
                        elif btype == "tool_use":
                            bid = current_block["id"]
                            tc  = tool_calls.get(bid, {})
                            try:
                                args = json.loads(tc.get("args_str", "{}"))
                            except json.JSONDecodeError:
                                args = {}
                            _put(BackendEvent(EventType.TOOL_CALL, {
                                "call_id": bid,
                                "name":    tc["name"],
                                "args":    args,
                            }))

                    elif etype == "message_delta":
                        disposition = "processed"
                        usage = data.get("usage", {})
                        output_tokens = usage.get("output_tokens", 0)

                    elif etype == "message_stop":
                        disposition = "processed"
                        # Build assistant content blocks for history
                        assistant_content: list[dict] = []
                        for txt in assistant_text_blocks:
                            assistant_content.append({"type": "text", "text": txt})
                        for bid, tc in tool_calls.items():
                            try:
                                args = json.loads(tc.get("args_str", "{}"))
                            except json.JSONDecodeError:
                                args = {}
                            assistant_content.append({
                                "type":  "tool_use",
                                "id":    bid,
                                "name":  tc["name"],
                                "input": args,
                            })
                        _put(BackendEvent(EventType.ASSISTANT_CONTENT,
                                          {"content": assistant_content}))
                        u = NormalizedUsage(
                            input_tokens             = input_tokens,
                            output_tokens            = output_tokens,
                            requests_remaining       = requests_remaining,
                            tokens_remaining_per_min = tokens_remaining_per_min,
                            reset_at                 = reset_at,
                        )
                        _put(BackendEvent(EventType.TURN_DONE, {"usage": u}))

                    _log_frame(data, disposition)

        except urllib.error.HTTPError:
            raise   # re-raise so _do_request can handle it
        except Exception as exc:
            _put(BackendEvent(EventType.ERROR, {"message": str(exc), "fatal": False}))

    async def submit_tool_result(self, call_id: str, result: str) -> None:
        self._history.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call_id, "content": result}
        ]})
        asyncio.create_task(self._do_request())

    async def inject_note(self, text: str) -> None:
        self._history.append({"role": "user", "content": f"[System note: {text}]"})

    async def close(self) -> None:
        self._closed = True

    @classmethod
    def translate_tools(cls, tools: list[CanonicalToolSchema]) -> list[dict]:
        return [{"name": t.name, "description": t.description,
                 "input_schema": t.parameters} for t in tools]


    @classmethod
    def list_models(cls, api_key: str) -> list[str]:
        """Fetch available model IDs from the Anthropic API."""
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/models",
            method="GET",
            headers={
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []

def _int_header(resp, name: str) -> int | None:
    try:
        val = resp.headers.get(name)
        return int(val) if val is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def _str_header(resp, name: str) -> str | None:
    try:
        return resp.headers.get(name)
    except AttributeError:
        return None

