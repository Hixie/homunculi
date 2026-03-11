"""Scripted mock backend for offline testing."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from .base import (BackendEvent, BackendScriptExhausted, BackendSession,
                   CanonicalToolSchema, EventType, NormalizedUsage,
                   ScriptedTurn)


class MockBackend(BackendSession):
    def __init__(self, script: list[ScriptedTurn] | None = None):
        self._script = list(script or [])
        self._idx = 0
        self._tool_result_futures: dict[str, asyncio.Future] = {}
        self.received_notes: list[str] = []
        self._closed = False
        self._event_queue: asyncio.Queue = asyncio.Queue()

    async def connect(self) -> None:
        await self._event_queue.put(BackendEvent(EventType.CONNECTED,
                                                  {"model": "mock-model"}))

    async def events(self) -> AsyncIterator[BackendEvent]:
        while not self._closed:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                continue

    async def send_turn(self, text: str) -> None:
        if self._idx >= len(self._script):
            await self._event_queue.put(BackendEvent(
                EventType.ERROR,
                {"message": "MockBackend script exhausted", "fatal": True}))
            return
        turn = self._script[self._idx]
        self._idx += 1
        asyncio.create_task(self._play_turn(turn))

    async def _play_turn(self, turn: ScriptedTurn):
        if turn.rate_limited is not None:
            await self._event_queue.put(BackendEvent(
                EventType.RATE_LIMITED, {"retry_after_s": turn.rate_limited}))
        for chunk in turn.text_chunks:
            await self._event_queue.put(BackendEvent(EventType.TEXT_DELTA, {"text": chunk}))
        for tc in turn.tool_calls:
            fut: asyncio.Future = asyncio.get_event_loop().create_future()
            self._tool_result_futures[tc.call_id] = fut
            await self._event_queue.put(BackendEvent(EventType.TOOL_CALL, {
                "call_id": tc.call_id, "name": tc.name, "args": tc.args}))
            await fut  # wait for submit_tool_result
        await self._event_queue.put(BackendEvent(
            EventType.TURN_DONE, {"usage": turn.usage}))

    async def submit_tool_result(self, call_id: str, result: str) -> None:
        fut = self._tool_result_futures.pop(call_id, None)
        if fut and not fut.done():
            fut.set_result(result)

    async def inject_note(self, text: str) -> None:
        self.received_notes.append(text)

    async def close(self) -> None:
        self._closed = True

    @classmethod
    def translate_tools(cls, tools: list[CanonicalToolSchema]) -> list[dict]:
        return [{"name": t.name, "description": t.description,
                 "parameters": t.parameters} for t in tools]

    @classmethod
    def list_models(cls, api_key: str) -> list[str]:
        return ["mock-model"]
