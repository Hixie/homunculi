"""ToolContext: shared pending_requests dict and send/request helpers."""
from __future__ import annotations

import asyncio
import uuid
from typing import Any


class ToolContext:
    def __init__(self, stdio, pending_requests: dict):
        self._stdio = stdio
        self._pending = pending_requests

    async def send(self, envelope: dict) -> None:
        await self._stdio.send(envelope)

    async def request(self, envelope: dict) -> dict:
        """Send envelope, register Future, await response."""
        id_ = envelope["id"]
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[id_] = fut
        await self._stdio.send(envelope)
        try:
            return await fut
        finally:
            self._pending.pop(id_, None)
