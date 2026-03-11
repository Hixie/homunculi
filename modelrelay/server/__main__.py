import sys

def _check_dependencies():
    try:
        import websockets
    except ImportError:
        sys.exit(
            "error: the 'websockets' package is required but not installed.\n"
            "  pip install 'websockets>=12'"
        )
    from importlib.metadata import version, PackageNotFoundError
    try:
        ws_version = tuple(int(x) for x in version("websockets").split(".")[:2])
        if ws_version < (12, 0):
            sys.exit(
                f"error: websockets>= 12 is required (found {version('websockets')}).\n"
                "  pip install --upgrade 'websockets>=12'"
            )
    except PackageNotFoundError:
        pass  # version metadata unavailable; import succeeded so allow it

_check_dependencies()

import asyncio
from .modelrelay import main

asyncio.run(main())
