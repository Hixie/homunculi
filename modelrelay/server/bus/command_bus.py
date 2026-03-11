"""Asyncio pub/sub command bus."""
from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from typing import Callable


class CommandBus:
    def __init__(self):
        self._handlers: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Callable):
        self._handlers[topic].append(handler)

    async def publish(self, topic: str, envelope: dict):
        for h in self._handlers.get(topic, []):
            try:
                result = h(envelope)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass  # isolated
