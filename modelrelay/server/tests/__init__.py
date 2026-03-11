# websockets is a required runtime dependency but is not available in the
# test environment (no network access to install packages).  Stub it here —
# before any backend module is imported — so that the package can be loaded
# and all tests that don't exercise real WebSocket connections continue to
# pass.  Tests that do exercise connect() inject their own mock via
# patch.object(oai_module, "websockets", ...), which overrides this stub for
# the duration of the test.
import sys
import types
import unittest.mock

if "websockets" not in sys.modules:
    _ws_stub = types.ModuleType("websockets")
    _ws_stub.connect = unittest.mock.AsyncMock()
    sys.modules["websockets"] = _ws_stub
