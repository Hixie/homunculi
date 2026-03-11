from ..stdio.protocol import session_ended


def register_quit(bus, shutdown_event, backend=None, stdio=None,
                  usage=None, log_path=""):
    async def handle(env):
        if backend:
            await backend.close()
        if stdio and usage is not None:
            summary = usage.session_summary()
            await stdio.send(session_ended(log_path, summary))
        shutdown_event.set()
    bus.subscribe("cmd.quit", handle)
