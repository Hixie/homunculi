def register_prompt(bus, orchestrator):
    async def handle(env):
        await orchestrator.on_prompt(env)
    bus.subscribe("cmd.prompt", handle)
