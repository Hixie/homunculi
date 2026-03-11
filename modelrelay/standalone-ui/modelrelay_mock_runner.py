#!/usr/bin/env python3
"""Wrapper that stubs websockets and runs modelrelay.server."""
import sys, types

ws = types.ModuleType("websockets")
ws.__version__ = "12.0"
for sub in ["exceptions", "client", "connection", "legacy"]:
    m = types.ModuleType(f"websockets.{sub}")
    sys.modules[f"websockets.{sub}"] = m
sys.modules["websockets"] = ws

sys.path.insert(0, '/home/claude/modelrelay_src3')
# Mimic __main__
import asyncio
from modelrelay.server.modelrelay import main
asyncio.run(main())
