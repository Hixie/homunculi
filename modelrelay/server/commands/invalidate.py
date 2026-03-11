def register_invalidate(bus, orchestrator):
    async def handle(env):
        await orchestrator.on_invalidate(env)
    bus.subscribe("cmd.invalidate", handle)
