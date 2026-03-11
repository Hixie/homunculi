# modelrelay

modelrelay is a subprocess bridge between a host application and an LLM backend.
It speaks a line-oriented JSON protocol (LOJP) over stdin/stdout, making it
straightforward to embed a coding assistant into any editor, shell tool, or CI
workflow without linking against a provider SDK.

```
host process
    │  cmd.prompt / cmd.invalidate / cmd.quit
    │  tool.content_response / tool.stat_response
    ▼
  stdin ──► modelrelay ──► Anthropic API (HTTP/SSE)
  stdout ◄──            ◄── OpenAI Realtime API (WebSocket)
    │
    │  model.text_delta
    │  tool.content_request / tool.stat_request
    │  tool.replace_request / tool.insert_request
    │  status.activity / status.usage
    │  session.ended / error
    ▼
host process
```

---

## Requirements

- Python 3.11 or later (uses `match`/`case`, `asyncio.TaskGroup`-style patterns)
- No third-party runtime dependencies for the Anthropic backend
- `websockets >= 12` for the OpenAI Realtime backend (`pip install websockets`)

---

## Installation

```bash
git clone <repo>
cd modelrelay/server   # or whatever the package directory is named
./run --help
```

---

## Usage

### Quick start

```bash
# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
./run --backend anthropic --model claude-opus-4-5

# OpenAI Realtime
export OPENAI_API_KEY=sk-...
./run --backend openai-realtime --model gpt-4o-realtime-preview
```

`run` can be invoked from any directory — it resolves its own location
and starts the server correctly regardless of the current working directory.

> **Why not `python3 modelrelay.py`?**
> Running the file directly detaches it from its package context and breaks
> all relative imports. Use `./run` instead.

modelrelay reads LOJP envelopes from stdin and writes them to stdout.
Both streams are UTF-8, newline-delimited JSON — one envelope per line.

### Config file

All flags can be set in `~/.modelrelay/config.json`. CLI flags take precedence.

```json
{
  "backend":           "anthropic",
  "model":             "claude-opus-4-5",
  "log_dir":           "~/.modelrelay/logs",
  "usage_interval_s":  30,
  "reconnect_attempts": 3
}
```

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--backend NAME` | — | **Required.** `anthropic`, `openai-realtime`, or `mock` |
| `--model NAME` | backend default | Model identifier string |
| `--system-prompt TEXT` | built-in | Override the system prompt inline |
| `--system-prompt-file PATH` | — | Load system prompt from a file |
| `--ide-context TEXT` | — | Append IDE/project context to the prompt |
| `--ide-context-file PATH` | — | Load IDE context from a file |
| `--log-dir PATH` | `~/.modelrelay/logs` | Directory for JSONL session logs |
| `--usage-interval SECONDS` | `30` | Maximum interval between `status.usage` emissions |
| `--reconnect-attempts N` | `3` | WebSocket reconnect attempts (OpenAI backend) |
| `--print-prompt` | — | Print the composed system prompt to stderr and exit |
| `--list-models` | — | Print available models and pricing for `--backend`, then exit |
| `--config PATH` | `~/.modelrelay/config.json` | Config file path |
| `--version` | — | Print version and exit |

### API keys

API keys are read exclusively from environment variables and are never accepted
on the command line or stored in the config file.

| Variable | Used by |
|---|---|
| `ANTHROPIC_API_KEY` | `--backend anthropic` |
| `OPENAI_API_KEY` | `--backend openai-realtime` |

### Listing available models

`--list-models` prints a JSON object to stdout describing every model available
for the selected backend, together with pricing estimates, then exits 0.
It requires `--backend` and reads the API key from the environment.

```bash
./run --backend anthropic --list-models
./run --backend openai-realtime --list-models
```

Output shape:

```json
{
  "backend": "anthropic",
  "models": [
    {
      "id": "claude-opus-4-5",
      "input_usd_per_1m": 15.0,
      "output_usd_per_1m": 75.0
    },
    {
      "id": "claude-sonnet-4-5",
      "input_usd_per_1m": 3.0,
      "output_usd_per_1m": 15.0
    }
  ]
}
```

Pricing is estimated from the prefix table in `model/usage_tracker.py`. Models
whose names do not match any known prefix fall back to the `"default"` rate.

If the API call fails (missing key, network error) the `"models"` array is empty
and an `"error"` key explains why:

```json
{
  "backend": "anthropic",
  "models": [],
  "error": "Could not fetch model list for 'anthropic'. Is ANTHROPIC_API_KEY set?"
}
```

---

## Protocol

Every message on stdin or stdout is a single JSON object followed by a newline.
All messages share a common envelope:

```json
{
  "lojp": "1.0",
  "id":   "a3f7c2d1-...",
  "ts":   "2024-03-15T09:22:11.042Z",
  "type": "<message type>",
  "payload": { ... }
}
```

### Host → modelrelay

| Type | Payload fields | Description |
|---|---|---|
| `cmd.prompt` | `text` | Send a user turn to the model |
| `cmd.invalidate` | `resource`, `start_line?`, `reason?` | Notify that a file was modified; modelrelay injects a note into context |
| `cmd.quit` | — | Shut down cleanly; triggers `session.ended` |
| `tool.content_response` | `resource`, `total_lines`, `regions[]`, `truncated`, `error` | Response to a `tool.content_request` |
| `tool.stat_response` | `resource`, `exists`, `total_lines`, `last_modified`, `error` | Response to a `tool.stat_request` |
| `tool.replace_response` | `error` | Response to a `tool.replace_request` |
| `tool.insert_response` | `error` | Response to a `tool.insert_request` |

### modelrelay → host

| Type | Payload fields | Description |
|---|---|---|
| `model.text_delta` | `text` | Streaming text chunk from the model |
| `tool.content_request` | `resource`, `ranges[]` or `pattern`, … | Model wants to read or search a file |
| `tool.stat_request` | `resource` | Model wants file metadata |
| `tool.replace_request` | `resource`, `start_line`, `end_line`, `new_lines[]`, `total_lines` | Model wants to replace a line range |
| `tool.insert_request` | `resource`, `after_line`, `new_lines[]`, `total_lines` | Model wants to insert lines |
| `status.activity` | `state`, `description` | State change: `idle`, `generating`, `waiting_for_tool`, `rate_limited` |
| `status.usage` | `prompt_tokens`, `completion_tokens`, `total_tokens`, `cost_usd`, … | Token and cost snapshot |
| `session.ended` | `logfile`, `summary` | Emitted on clean shutdown |
| `error` | `code`, `message`, `fatal` | Protocol or backend error |

### Request/response matching

All four tool request types carry an `id` field. The host must echo that same
`id` in the corresponding response envelope. modelrelay uses the id to resolve
the awaiting future and unblock the tool call.

| Request | Response |
|---|---|
| `tool.content_request` | `tool.content_response` |
| `tool.stat_request` | `tool.stat_response` |
| `tool.replace_request` | `tool.replace_response` |
| `tool.insert_request` | `tool.insert_response` |

modelrelay will not return a tool result to the model until the host responds.
If the host rejects a change it should send an error payload in the response,
then optionally follow up with `cmd.invalidate` to inject a note into context.

---

## Architecture

```
modelrelay/
├── modelrelay.py          entry point: wire everything, run the event loop
├── cli.py                 argparse → Config dataclass
├── prompt.py              compose_system_prompt(), DEFAULT_SYSTEM_PROMPT
│
├── stdio/
│   ├── protocol.py        LOJP encode/decode; envelope factory functions
│   └── layer.py           async stdin reader / stdout writer; resolves pending futures
│
├── bus/
│   └── command_bus.py     async pub/sub; routes inbound messages to handlers
│
├── commands/
│   ├── prompt.py          cmd.prompt → orchestrator.on_prompt()
│   ├── invalidate.py      cmd.invalidate → orchestrator.on_invalidate()
│   └── quit.py            cmd.quit → session.ended + shutdown
│
├── model/
│   ├── orchestrator.py    backend event loop; dispatches tool calls; forwards text
│   └── usage_tracker.py   token/cost accumulation; session summary
│
├── backends/
│   ├── base.py            BackendSession ABC, EventType, NormalizedUsage, ScriptedTurn
│   ├── anthropic.py       HTTP/SSE backend (urllib, no SDK)
│   ├── openai_realtime.py WebSocket backend (requires websockets package)
│   └── mock.py            scripted in-process backend for tests
│
├── tools/
│   ├── registry.py        ToolRegistry: register, invoke
│   ├── context.py         ToolContext: send() and request() helpers
│   ├── read.py            read tool (ranges)
│   ├── replace.py         replace tool
│   ├── insert.py          insert tool
│   ├── stat.py            stat tool
│   └── search.py          search tool (substring)
│
├── auditlog/
│   └── logger.py          JSONL session logger (protocol, backend, event categories)
│
└── tests/
    ├── conftest.py         InMemorySession, make_session(), BackendContractMixin
    ├── unit/               one file per module
    ├── backend/            SSE parsing, WebSocket frame translation, class-level tests
    └── integration/        full session flows over in-memory transport
```

### Data flow for a single turn

1. Host writes `cmd.prompt` to stdin.
2. `StdioLayer` reads the line, decodes it, publishes to the `CommandBus`.
3. `commands/prompt.py` handler calls `orchestrator.on_prompt()`.
4. Orchestrator calls `backend.send_turn(text)`.
5. The backend streams events: `TEXT_DELTA` → orchestrator emits `model.text_delta`; `TOOL_CALL` → orchestrator invokes the tool and calls `backend.submit_tool_result()`.
6. `TURN_DONE` → orchestrator emits `status.usage` and `status.activity(idle)`.

### Adding a new backend

1. Subclass `BackendSession` in `backends/base.py` and implement all six abstract methods: `connect`, `events`, `send_turn`, `submit_tool_result`, `inject_note`, `close`.
2. Add a `translate_tools` classmethod that converts `list[CanonicalToolSchema]` to your provider's tool format.
3. Register it in `backends/__init__.py` under a new `cfg.backend` key.
4. Add a backend identifier and default model to `_BACKEND_MODELS` in `cli.py`.
5. Write tests (see below).

### Adding a new tool

1. Create `tools/mytool.py` with a `SCHEMA` dict (JSON Schema) and an `async def invoke(args, ctx)` function.
2. Instantiate a `ToolHandler` and export it from the module.
3. Import and register it in `tools/__init__.py:build_registry()`.
4. If the tool needs a new request/response message pair, add factory functions to `stdio/protocol.py` and add the response type to `KNOWN_INBOUND_TYPES` in `stdio/layer.py`.
5. Write tests (see below).

---

## Running the tests

The test suite requires no additional packages beyond the standard library.

```bash
# Run the full suite
./test

# Run with verbose output
./test -v

# Run a specific module
./test modelrelay.tests.unit.test_protocol -v

# Run a single test method
./test modelrelay.tests.unit.test_protocol.TestProtocol.test_encode_decode_roundtrip
```

`test` is a shell script (next to `run`) that derives the package name from its
own location, so it works regardless of what the directory is called. Any
arguments after `./test` are forwarded directly to `python3 -m unittest`.

The suite currently contains 269 tests across 27 files and completes in under
10 seconds. No network access is required; all HTTP and WebSocket calls are
patched with `unittest.mock`.

---

## Writing tests

### Unit tests

Unit tests live in `tests/unit/` and cover a single module in isolation. They
use only the standard library and avoid async wherever possible.

```python
# tests/unit/test_tool_mytool.py
import unittest
from ..tools.mytool import invoke

class TestMyTool(unittest.IsolatedAsyncioTestCase):

    async def test_basic_invoke(self):
        class FakeCtx:
            async def request(self, env):
                return {"payload": {"result": "ok", "error": None}}
        result = await invoke({"resource": "f.py"}, FakeCtx())
        self.assertIn("ok", result)
```

### Backend tests

Backend tests live in `tests/backend/`. There are two patterns:

**SSE / frame parsing** — extract the parsing logic into a standalone function
and test it without touching the class:

```python
# tests/backend/test_myprovider_parsing.py
import unittest, json
from ..backends.myprovider import _parse_frame

class TestMyProviderParsing(unittest.TestCase):
    def test_text_delta(self):
        frame = {"type": "text.delta", "text": "hello"}
        event = _parse_frame(frame)
        self.assertEqual(event.type, EventType.TEXT_DELTA)
        self.assertEqual(event.data["text"], "hello")
```

**Class-level behaviour** — use `unittest.mock.patch` to intercept network
calls and drive events through the queue directly:

```python
from unittest.mock import patch, MagicMock
from ..backends.myprovider import MyProviderBackend

class TestMyProviderBackend(unittest.IsolatedAsyncioTestCase):
    async def test_turn_done_has_usage(self):
        b = MyProviderBackend(api_key="test")
        await b.connect()
        with patch("urllib.request.urlopen", return_value=_fake_response(...)):
            await b.send_turn("hi")
            async for ev in b.events():
                if ev.type == EventType.TURN_DONE:
                    self.assertIsInstance(ev.data["usage"], NormalizedUsage)
                    break
```

**Backend contract** — mix in `BackendContractMixin` from `tests/conftest.py`
to verify that your backend satisfies the standard interface:

```python
from ..conftest import BackendContractMixin
from ..backends.myprovider import MyProviderBackend

class TestMyProviderContract(BackendContractMixin,
                              unittest.IsolatedAsyncioTestCase):
    def make_backend(self):
        # Return a backend pre-loaded with one scripted turn
        b = MyProviderBackend(api_key="test")
        b._script = [ScriptedTurn(text_chunks=["hello"])]
        return b
```

`BackendContractMixin` provides six tests covering `connect`, text deltas,
`TURN_DONE` usage, `inject_note`, and `close`.

### Integration tests

Integration tests live in `tests/integration/`. They wire the full stack — 
`Orchestrator`, `FakeStdioLayer`, `MockBackend`, real tool registry — over an
in-memory transport. No actual I/O takes place.

Use `make_session()` from `tests/conftest.py` to create a session:

```python
from ..conftest import make_session
from ..backends.base import ScriptedTurn, ScriptedToolCall

class TestMyFlow(unittest.IsolatedAsyncioTestCase):

    async def test_my_tool_called(self):
        tc = ScriptedToolCall(name="mytool", args={"resource": "f.py"})
        session, task = await make_session([ScriptedTurn(tool_calls=[tc])])

        await session.recv_until("status.activity")        # CONNECTED idle
        await session.send_host("cmd.prompt", {"text": "go"})

        env = await session.recv_until("tool.mytool_request", timeout=3.0)
        self.assertEqual(env["payload"]["resource"], "f.py")

        # Reply on behalf of the host
        await session.send_host("tool.mytool_response",
            {"resource": "f.py", "result": "ok", "error": None},
            id_=env["id"])

        await session.recv_until("status.activity", timeout=5.0)  # idle after turn
        task.cancel()
        try: await task
        except: pass
```

`make_session` accepts an optional `usage_interval_s` parameter (default 3600)
so that the polling timer does not fire unexpectedly during short tests.

`session.recv_until(type, timeout)` drains the outbound queue until it sees a
message of the requested type, or raises `asyncio.TimeoutError`. Use it instead
of `recv()` whenever ordering is not guaranteed.

---

## Session logs

modelrelay writes a JSONL log to `~/.modelrelay/logs/` (one file per session,
named by UTC start time). Every line is a self-contained JSON object. Records
are written in strict chronological order with a 1-based `seq` counter.

Every record starts with `seq`, `ts`, and `category`. The five categories are:

| Category | What it covers |
|---|---|
| `protocol` | Every LOJP message crossing the host ↔ modelrelay boundary |
| `wire_rx` | Every raw frame received from the backend before filtering |
| `model_rx` | BackendEvents emitted toward the orchestrator |
| `model_tx` | Messages sent to the backend (turns, tool results, notes) |
| `lifecycle` | Session-level events (`session_start`, `session_end`) |

Every `protocol` and `wire_rx` record includes a `disposition` field:
`"processed"` (message caused a state change or action), `"ignored"` (well-formed
but no handler fired — wire_rx only), or `"erroneous"` (could not be parsed or
type was unknown).

```jsonc
// lifecycle — always the first record
{"seq":1,"ts":"2026-03-09T21:33:14.316408+00:00","category":"lifecycle",
 "event":"session_start","data":{"backend":"anthropic","model":"claude-opus-4-5",
 "log_file":"20260309T213300.jsonl"}}

// protocol — inbound cmd.prompt
{"seq":2,"ts":"…","category":"protocol",
 "direction":"inbound","disposition":"processed",
 "msg_id":"5c98…","msg_ts":"2026-03-09T21:33:14.316Z",
 "msg_type":"cmd.prompt","payload":{"text":"fix the bug"}}

// wire_rx — raw SSE frame from Anthropic, processed
{"seq":3,"ts":"…","category":"wire_rx","backend":"anthropic",
 "disposition":"processed","frame":{"type":"content_block_delta",
 "delta":{"type":"text_delta","text":"Hello"}}}

// wire_rx — informational frame, ignored
{"seq":4,"ts":"…","category":"wire_rx","backend":"openai-realtime",
 "disposition":"ignored","frame":{"type":"conversation.item.created","item":{…}}}
```

The `logfile` path is included in the `session.ended` payload so the host can
surface it to the user.

---

## Contribution guidelines

### Before opening a pull request

- All 186 existing tests must pass with no modifications.
- New features must ship with tests. New modules need unit tests. New protocol
  flows need integration tests. New backends need SSE/frame parsing tests, a
  class-level test, and `BackendContractMixin` coverage.
- No new runtime dependencies. If a feature genuinely requires one, gate it
  behind a try/import (as `websockets` is for the OpenAI backend) and document
  the install step.

### Code style

- Python 3.11+. Type annotations on all public functions and class members.
- `from __future__ import annotations` at the top of every module.
- Async throughout; no blocking I/O on the event loop thread. Use
  `run_in_executor` for blocking calls (see `AnthropicBackend._stream_response`).
- When posting events from a worker thread back to the async queue, always use
  `loop.call_soon_threadsafe(queue.put_nowait, event)` — never
  `loop.run_until_complete(queue.put(...))` from a second event loop.
- Keep modules single-responsibility. The boundary between layers is: protocol
  encoding lives in `stdio/`, routing lives in `bus/` + `commands/`, provider
  I/O lives in `backends/`, and file-operation semantics live in `tools/`.

### What belongs where

| What you're adding | Where it goes |
|---|---|
| New CLI flag | `cli.py`: add to argparse, add field to `Config`, thread through `main()` |
| New outbound message type | `stdio/protocol.py`: add factory function |
| New inbound message type | `stdio/protocol.py` + add to `KNOWN_INBOUND_TYPES` in `stdio/layer.py` |
| New tool | `tools/newtool.py` + register in `tools/__init__.py` |
| New backend | `backends/newprovider.py` + register in `backends/__init__.py` and `cli.py` |
| New command | `commands/newcmd.py` + register in `commands/__init__.py:register_all()` |
