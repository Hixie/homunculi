# modelrelay-standalone-ui

A self-contained interactive terminal host for the [modelrelay](https://github.com/example/modelrelay) subprocess, implementing the full LOJP (Line-Oriented JSON Protocol) host side.

## Requirements

- Python 3.11+
- No third-party dependencies

## Usage

```bash
python modelrelay_tui.py /path/to/modelrelay-executable
```

A configuration wizard runs in normal terminal mode. Once confirmed, the session switches to a split-screen curses TUI.

## Screen layout

```
┌─ STATUS ──────────────────────────────────────────────┐
│  IDLE  model: gpt-4o  tokens: 1,234  $0.0049  files:2 │
├─ MODEL OUTPUT ─────────────────────────────────────────┤
│                                                        │
│  The model's response appears here, word-wrapped.      │
│  Tool invocations (read/write/search) appear inline.   │
│                                                        │
├─ YOU ──────────────────────────────────────────────────┤
│  > your message here                                   │
└────────────────────────────────────────────────────────┘
```

- **Status bar** — state badge (colour-coded), model, token count, cost, live file count
- **Model region** (~2/3 of terminal) — streaming output word-wrapped at terminal width; tool calls shown inline with diffs for edits
- **Input region** — highlighted cyan on your turn; dimmed while the model generates; `CONFIRM [y/n]` during file-edit confirmation

## Commands

| Command | Description |
|---|---|
| `/quit` | End the session cleanly |
| `/invalidate <file>` | Notify modelrelay that a file changed externally |
| `/help` | Show command list |

## Configuration

Settings are saved to `.modelrelay` (JSON) in the chosen working directory. The last-used working directory is remembered in `~/.modelrelay/last_working_dir`.

## Tests

```bash
python -m unittest test_tui_shutdown -v
```

Tests cover the async shutdown mechanics (threading boundary between curses main thread and asyncio worker thread), including a regression test for the `loop.stop()` bug that previously caused `RuntimeError: Event loop stopped before Future completed` on quit.
