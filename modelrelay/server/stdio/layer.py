"""Async stdin reader / stdout writer. Resolves pending_requests futures."""
from __future__ import annotations

import asyncio
import sys

from .protocol import decode, encode, error_msg

# All message types the host is permitted to send
KNOWN_INBOUND_TYPES = frozenset({
    "tool.content_response",
    "tool.stat_response",
    "tool.insert_response",
    "tool.replace_response",
    "cmd.prompt",
    "cmd.invalidate",
    "cmd.quit",
})


class StdioLayer:
    def __init__(self, pending_requests: dict, bus, logger=None):
        self._pending = pending_requests
        self._bus = bus
        self._logger = logger
        self._lock = asyncio.Lock()
        self._reader: asyncio.StreamReader | None = None

    async def start(self):
        loop = asyncio.get_event_loop()
        self._reader = asyncio.StreamReader()
        proto = asyncio.StreamReaderProtocol(self._reader)
        await loop.connect_read_pipe(lambda: proto, sys.stdin)

    async def send(self, envelope: dict):
        data = encode(envelope)
        async with self._lock:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        await self._bus.publish("_outbound", envelope)

    async def run_inbound(self):
        """Read lines from stdin and dispatch to bus."""
        while True:
            line = await self._reader.readline()
            if not line:
                break
            try:
                env = decode(line)
            except Exception as exc:
                raw = line.decode("utf-8", errors="replace").rstrip("\n")
                if self._logger:
                    self._logger.log_protocol_inbound_error(raw, "PARSE_ERROR", str(exc))
                await self.send(error_msg("PARSE_ERROR", str(exc), fatal=False))
                continue

            type_ = env.get("type", "")

            if type_ not in KNOWN_INBOUND_TYPES:
                raw = line.decode("utf-8", errors="replace").rstrip("\n")
                if self._logger:
                    self._logger.log_protocol_inbound_error(raw, "UNKNOWN_TYPE", type_)
                await self.send(error_msg(
                    "UNKNOWN_TYPE",
                    f"Unrecognised inbound message type: {type_!r}",
                    fatal=False))
                continue

            await self._bus.publish("_inbound", env)
            if type_ in ("tool.content_response", "tool.stat_response",
                         "tool.insert_response", "tool.replace_response"):
                id_ = env.get("id")
                fut = self._pending.get(id_)
                if fut and not fut.done():
                    fut.set_result(env)
            await self._bus.publish(type_, env)
