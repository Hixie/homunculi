"""Entry point: parse args, wire components, event loop."""
from __future__ import annotations

import asyncio
import os
import signal
import sys

from .bus.command_bus import CommandBus
from .cli import parse_args
from .commands import register_all
from .auditlog.logger import Logger
from .model.orchestrator import Orchestrator
from .model.usage_tracker import UsageTracker
from .prompt import compose_system_prompt
from .stdio.layer import StdioLayer
from .stdio.protocol import session_ended


async def main(argv=None):
    cfg = parse_args(argv)
    system_prompt = compose_system_prompt(cfg.system_prompt, cfg.ide_context)

    bus     = CommandBus()
    pending: dict = {}

    from .tools import build_registry
    registry = build_registry()

    logger = Logger(cfg.log_dir)
    logger.write_event("session_start", {
        "backend":  cfg.backend,
        "model":    cfg.model,
        "log_file": os.path.basename(logger.path),
    })

    stdio = StdioLayer(pending, bus, logger=logger)
    await stdio.start()

    bus.subscribe("_inbound",  lambda e: logger.log_protocol(e, "inbound"))
    bus.subscribe("_outbound", lambda e: logger.log_protocol(e, "outbound"))

    from .backends import build_backend
    backend = build_backend(cfg, logger=logger)
    await backend.connect()

    usage = UsageTracker(model=cfg.model)
    orch  = Orchestrator(backend, stdio, registry, usage, pending,
                         usage_interval_s=cfg.usage_interval_s,
                         logger=logger)

    shutdown = asyncio.Event()

    # Register SIGINT / SIGTERM so clean shutdown always fires
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown.set)
        except NotImplementedError:
            pass  # Windows

    register_all(bus, orch, shutdown, backend=backend,
                 stdio=stdio, usage=usage, log_path=logger.path)

    inbound_task = asyncio.create_task(stdio.run_inbound())
    orch_task    = asyncio.create_task(orch.run())

    await shutdown.wait()
    orch_task.cancel()
    inbound_task.cancel()
    try:
        await orch_task
    except asyncio.CancelledError:
        pass
    await backend.close()

    summary = usage.session_summary()
    logger.write_event("session_end", {
        "total_tokens": summary.get("total_tokens", 0),
        "cost_usd":     summary.get("cost_usd", 0.0),
    })
    await stdio.send(session_ended(logger.path, summary))
    logger.close()


if __name__ == "__main__":
    asyncio.run(main())
