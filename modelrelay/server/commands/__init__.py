from .prompt     import register_prompt
from .invalidate import register_invalidate
from .quit       import register_quit


def register_all(bus, orchestrator, shutdown_event, backend=None,
                 stdio=None, usage=None, log_path=""):
    register_prompt(bus, orchestrator)
    register_invalidate(bus, orchestrator)
    register_quit(bus, shutdown_event, backend=backend,
                  stdio=stdio, usage=usage, log_path=log_path)
