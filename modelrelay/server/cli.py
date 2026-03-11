"""argparse → Config dataclass."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

from .prompt import DEFAULT_SYSTEM_PROMPT
from .model.usage_tracker import PRICING, _get_pricing


@dataclass
class Config:
    backend:            str
    model:              str
    anthropic_api_key:  str | None
    openai_api_key:     str | None
    log_dir:            str
    usage_interval_s:   float
    reconnect_attempts: int
    system_prompt:      str
    ide_context:        str | None


BACKENDS: dict[str, str] = {
    "anthropic":       "claude-opus-4-5",
    "openai-realtime": "gpt-4o-realtime-preview",
    "mock":            "mock-model",
}

# Keep the old name for any code that still references it
_BACKEND_MODELS = BACKENDS


def _fetch_models(backend: str) -> list[str]:
    """Ask the backend for its current model list. Returns [] on failure."""
    api_key: str
    if backend == "anthropic":
        from .backends.anthropic import AnthropicBackend
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        return AnthropicBackend.list_models(api_key)
    elif backend == "openai-realtime":
        from .backends.openai_realtime import OpenAIRealtimeBackend
        api_key = os.environ.get("OPENAI_API_KEY", "")
        return OpenAIRealtimeBackend.list_models(api_key)
    elif backend == "mock":
        from .backends.mock import MockBackend
        return MockBackend.list_models("")
    return []


def _model_help(backend: str | None) -> str:
    """Build the --model help string, optionally listing live models."""
    if backend and backend in BACKENDS:
        print(f"  Fetching available models for {backend}...", file=sys.stderr, flush=True)
        models = _fetch_models(backend)
        if models:
            listing = ", ".join(models)
            return f"Model identifier. Available for {backend}: {listing}"
        else:
            return (f"Model identifier. (Could not fetch model list for {backend} "
                    f"— is the API key set?)")
    default_note = "; ".join(f"{b}: default={m}" for b, m in BACKENDS.items()
                             if b != "mock")
    return f"Model identifier ({default_note})"


def _list_models_json(backend: str) -> str:
    """Return a machine-readable JSON string describing available models and
    their pricing estimates for *backend*.  Prints to stdout and exits 0.

    Output shape::

        {
          "backend": "<name>",
          "models": [
            {"id": "<model-id>",
             "input_usd_per_1m": <float>,
             "output_usd_per_1m": <float>},
            ...
          ]
        }

    On API failure the "models" list is empty and "error" explains why.
    """
    api_key: str
    if backend == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    elif backend == "openai-realtime":
        api_key = os.environ.get("OPENAI_API_KEY", "")
    else:
        api_key = ""

    models_raw = _fetch_models(backend)

    result: dict = {"backend": backend}
    if not models_raw and backend not in ("mock",):
        env_var = "ANTHROPIC_API_KEY" if backend == "anthropic" else "OPENAI_API_KEY"
        result["models"] = []
        result["error"]  = (f"Could not fetch model list for {backend!r}. "
                            f"Is {env_var} set?")
    else:
        result["models"] = [
            {
                "id":                 m,
                "input_usd_per_1m":  _get_pricing(m)[0],
                "output_usd_per_1m": _get_pricing(m)[1],
            }
            for m in models_raw
        ]
    return json.dumps(result, indent=2)


def parse_args(argv=None) -> Config:
    # Pre-scan argv to find --backend so we can enrich --model help when
    # --help/-h is also present, before argparse processes anything.
    raw = argv if argv is not None else sys.argv[1:]
    pre_backend: str | None = None
    for i, tok in enumerate(raw):
        if tok in ("--backend", "-b") and i + 1 < len(raw):
            pre_backend = raw[i + 1]
        elif tok.startswith("--backend="):
            pre_backend = tok.split("=", 1)[1]
    help_requested       = any(t in raw for t in ("-h", "--help"))
    list_models_requested = "--list-models" in raw

    # Handle --list-models early, before argparse, so we can use pre_backend.
    if list_models_requested:
        if not pre_backend:
            print("error: --list-models requires --backend", file=sys.stderr)
            sys.exit(2)
        if pre_backend not in BACKENDS:
            backend_list_str = ", ".join(BACKENDS)
            print(f"error: unknown backend {pre_backend!r}. "
                  f"Valid values: {backend_list_str}", file=sys.stderr)
            sys.exit(2)
        print(_list_models_json(pre_backend))
        sys.exit(0)

    backend_list = ", ".join(BACKENDS)
    model_help   = _model_help(pre_backend) if help_requested else _model_help(None)

    p = argparse.ArgumentParser(
        prog="modelrelay",
        description="Subprocess bridge between a host application and an LLM backend. "
                    "Communicates over stdin/stdout using line-oriented JSON (LOJP).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",
                   default="~/.modelrelay/config.json",
                   metavar="PATH",
                   help="Config file path (default: ~/.modelrelay/config.json)")
    p.add_argument("--backend",
                   default=None,
                   metavar="NAME",
                   help=f"Backend to use. Required. Valid values: {backend_list}")
    p.add_argument("--model",
                   default=None,
                   metavar="NAME",
                   help=model_help)
    p.add_argument("--log-dir",
                   default=None,
                   metavar="PATH",
                   help="Directory for JSONL session logs (default: ~/.modelrelay/logs)")
    p.add_argument("--usage-interval",
                   type=float, default=None, dest="usage_interval_s",
                   metavar="SECONDS",
                   help="Maximum interval between status.usage emissions (default: 30)")
    p.add_argument("--reconnect-attempts",
                   type=int, default=None,
                   metavar="N",
                   help="WebSocket reconnect attempts for openai-realtime (default: 3)")
    p.add_argument("--system-prompt",
                   default=None,
                   metavar="TEXT",
                   help="Override the system prompt inline")
    p.add_argument("--system-prompt-file",
                   default=None,
                   metavar="PATH",
                   help="Load system prompt from a file")
    p.add_argument("--ide-context",
                   default=None,
                   metavar="TEXT",
                   help="Append IDE/project context to the system prompt")
    p.add_argument("--ide-context-file",
                   default=None,
                   metavar="PATH",
                   help="Load IDE context from a file")
    p.add_argument("--print-prompt",
                   action="store_true", default=False,
                   help="Print the composed system prompt to stderr and exit")
    p.add_argument("--list-models",
                   action="store_true", default=False,
                   dest="list_models",
                   help="Print a machine-readable JSON list of available models and "
                        "pricing for the selected backend, then exit. "
                        "Requires --backend.")
    p.add_argument("--version",
                   action="version", version="modelrelay 0.6.0")

    args = p.parse_args(argv)

    # Mutual exclusions
    if args.system_prompt and args.system_prompt_file:
        p.error("--system-prompt and --system-prompt-file are mutually exclusive")
    if args.ide_context and args.ide_context_file:
        p.error("--ide-context and --ide-context-file are mutually exclusive")

    # Load config file
    cfg_path = os.path.expanduser(args.config)
    file_cfg: dict = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                file_cfg = json.load(f)
        except Exception:
            pass

    def get(cli_val, cfg_key, default=None):
        if cli_val is not None:
            return cli_val
        return file_cfg.get(cfg_key, default)

    backend = get(args.backend, "backend")
    if not backend:
        p.error("--backend is required (or set 'backend' in config.json)")
    if backend not in BACKENDS:
        p.error(f"Unknown backend: {backend!r}. Valid values: {backend_list}")

    model = get(args.model, "model", BACKENDS[backend])

    # System prompt
    if args.system_prompt_file:
        try:
            with open(args.system_prompt_file) as f:
                system_prompt = f.read()
        except FileNotFoundError:
            p.error(f"System prompt file not found: {args.system_prompt_file}")
    else:
        system_prompt = get(args.system_prompt, "system_prompt", DEFAULT_SYSTEM_PROMPT)

    # IDE context
    if args.ide_context_file:
        try:
            with open(args.ide_context_file) as f:
                ide_context = f.read()
        except FileNotFoundError:
            p.error(f"IDE context file not found: {args.ide_context_file}")
    else:
        ide_context = get(args.ide_context, "ide_context", None)

    cfg = Config(
        backend           = backend,
        model             = model,
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY"),
        openai_api_key    = os.environ.get("OPENAI_API_KEY"),
        log_dir           = get(args.log_dir,           "log_dir",           os.path.expanduser("~/.modelrelay/logs")),
        usage_interval_s  = get(args.usage_interval_s,  "usage_interval_s",  30.0),
        reconnect_attempts= get(args.reconnect_attempts,"reconnect_attempts", 3),
        system_prompt     = system_prompt,
        ide_context       = ide_context,
    )

    if args.print_prompt:
        from .prompt import compose_system_prompt
        composed = compose_system_prompt(cfg.system_prompt, cfg.ide_context)
        print(composed, file=sys.stderr)
        sys.exit(0)

    return cfg
