"""ToolRegistry."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolHandler:
    name: str
    schema: dict  # JSON Schema for parameters
    description: str
    invoke_fn: Any  # async callable(args, ctx) -> str


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolHandler] = {}

    def register(self, handler: ToolHandler):
        self._tools[handler.name] = handler

    def has(self, name: str) -> bool:
        return name in self._tools

    def get_schemas(self):
        from ..backends.base import CanonicalToolSchema
        return [CanonicalToolSchema(name=h.name, description=h.description,
                                    parameters=h.schema)
                for h in self._tools.values()]

    async def invoke(self, name: str, args: dict, ctx) -> str:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return await self._tools[name].invoke_fn(args, ctx)
