"""BackendSession ABC, BackendEvent, NormalizedUsage, CanonicalToolSchema."""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, AsyncIterator


class EventType(Enum):
    CONNECTED         = auto()
    TEXT_DELTA        = auto()
    TOOL_CALL         = auto()
    TURN_DONE         = auto()
    RATE_LIMITED      = auto()
    ERROR             = auto()
    DISCONNECTED      = auto()
    ASSISTANT_CONTENT = auto()   # internal: assistant turn for history (Anthropic only)


@dataclass
class NormalizedUsage:
    input_tokens:             int       = 0
    output_tokens:            int       = 0
    requests_remaining:       int|None  = None
    tokens_remaining_per_min: int|None  = None
    reset_at:                 str|None  = None


@dataclass
class BackendEvent:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalToolSchema:
    name:        str
    description: str
    parameters:  dict


@dataclass
class ScriptedToolCall:
    name:    str
    args:    dict
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ScriptedTurn:
    text_chunks:  list[str]              = field(default_factory=list)
    tool_calls:   list[ScriptedToolCall] = field(default_factory=list)
    usage:        NormalizedUsage        = field(default_factory=NormalizedUsage)
    rate_limited: float | None           = None


class BackendScriptExhausted(Exception):
    pass


class BackendSession(ABC):
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    def events(self) -> AsyncIterator[BackendEvent]: ...

    @abstractmethod
    async def send_turn(self, text: str) -> None: ...

    @abstractmethod
    async def submit_tool_result(self, call_id: str, result: str) -> None: ...

    @abstractmethod
    async def inject_note(self, text: str) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @classmethod
    def translate_tools(cls, tools: list[CanonicalToolSchema]) -> list[dict]:
        raise NotImplementedError
