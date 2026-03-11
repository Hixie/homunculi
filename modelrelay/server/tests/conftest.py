"""conftest.py — InMemorySession, BackendContractMixin, and make_session."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ..backends.mock import MockBackend
from ..backends.base import EventType, ScriptedTurn
from ..bus.command_bus import CommandBus
from ..model.orchestrator import Orchestrator
from ..model.usage_tracker import UsageTracker
from ..stdio.protocol import make_envelope
from ..tools import build_registry
from ..commands import register_all


class FakeStdioLayer:
    """In-memory stdio: writes go to an asyncio.Queue."""
    def __init__(self, pending, bus):
        self._pending = pending
        self._bus = bus
        self._outbound: asyncio.Queue = asyncio.Queue()

    async def send(self, envelope: dict):
        await self._outbound.put(envelope)
        await self._bus.publish("_outbound", envelope)

    async def inject(self, type_: str, payload: dict, id_: str | None = None):
        """Simulate host sending a message."""
        env = make_envelope(type_, payload, id_=id_)
        if type_ in ("tool.content_response", "tool.stat_response",
                     "tool.insert_response", "tool.replace_response"):
            eid = env.get("id")
            fut = self._pending.get(eid)
            if fut and not fut.done():
                fut.set_result(env)
        await self._bus.publish("_inbound", env)
        await self._bus.publish(type_, env)

    async def recv(self, timeout: float = 2.0) -> dict:
        return await asyncio.wait_for(self._outbound.get(), timeout=timeout)

    async def recv_until(self, type_: str, timeout: float = 5.0) -> dict:
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError(f"Never received {type_!r}")
            try:
                env = await asyncio.wait_for(
                    self._outbound.get(), timeout=min(remaining, 0.5))
                if env.get("type") == type_:
                    return env
            except asyncio.TimeoutError:
                continue


@dataclass
class InMemorySession:
    stdio:    FakeStdioLayer
    backend:  MockBackend
    bus:      CommandBus
    orch:     Orchestrator
    shutdown: asyncio.Event

    async def send_host(self, type_: str, payload: dict, id_: str | None = None):
        await self.stdio.inject(type_, payload, id_=id_)

    async def recv(self, timeout: float = 2.0) -> dict:
        return await self.stdio.recv(timeout)

    async def recv_until(self, type_: str, timeout: float = 5.0) -> dict:
        return await self.stdio.recv_until(type_, timeout)


async def make_session(
    script: list | None = None,
    usage_interval_s: float = 3600.0,   # default: no polling during tests
) -> tuple[InMemorySession, asyncio.Task]:
    pending: dict = {}
    bus     = CommandBus()
    backend = MockBackend(script or [])
    await backend.connect()
    stdio   = FakeStdioLayer(pending, bus)
    registry = build_registry()
    usage   = UsageTracker()
    orch    = Orchestrator(backend, stdio, registry, usage, pending,
                           usage_interval_s=usage_interval_s)
    shutdown = asyncio.Event()
    register_all(bus, orch, shutdown, backend=backend,
                 stdio=stdio, usage=usage, log_path="")
    task = asyncio.create_task(orch.run())
    return InMemorySession(stdio=stdio, backend=backend, bus=bus,
                           orch=orch, shutdown=shutdown), task


# ── BackendContractMixin ──────────────────────────────────────────────────────

class BackendContractMixin:
    """
    Mix this into a test class that implements make_backend() to verify
    every BackendSession implementation satisfies the basic contract.
    """

    def make_backend(self):
        raise NotImplementedError

    async def test_connect_yields_connected(self):
        b = self.make_backend()
        await b.connect()
        async for ev in b.events():
            self.assertEqual(ev.type, EventType.CONNECTED)
            self.assertIn("model", ev.data)
            break
        await b.close()

    async def test_text_delta_carries_text(self):
        """TEXT_DELTA events must include data['text'] for Orchestrator forwarding."""
        b = self.make_backend()
        await b.connect()
        async for ev in b.events():
            if ev.type == EventType.TEXT_DELTA:
                self.assertIn("text", ev.data)
                break
            if ev.type == EventType.TURN_DONE:
                break  # turn completed without text; that's fine
        await b.close()

    async def test_turn_done_has_usage(self):
        from ..backends.base import NormalizedUsage
        b = self.make_backend()
        await b.connect()
        async for ev in b.events():
            if ev.type == EventType.TURN_DONE:
                self.assertIsInstance(ev.data["usage"], NormalizedUsage)
                break
        await b.close()

    async def test_inject_note_no_error(self):
        b = self.make_backend()
        await b.connect()
        await b.inject_note("a note")   # must not raise
        await b.close()

    async def test_close_stops_events(self):
        b = self.make_backend()
        await b.connect()
        await b.close()
        # After close, events() should stop iterating
        count = 0
        async for _ in b.events():
            count += 1
            if count > 5:
                break
        # We allow up to 5 leftover queued events; what we require is that
        # iterating does not block forever (the test framework will time out)
        self.assertLessEqual(count, 5)
