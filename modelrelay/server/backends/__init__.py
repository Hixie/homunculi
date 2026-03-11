from .anthropic import AnthropicBackend
from .openai_realtime import OpenAIRealtimeBackend
from .mock import MockBackend
from .base import CanonicalToolSchema


def build_backend(cfg, logger=None):
    from ..tools import build_registry
    registry = build_registry()
    schemas = registry.get_schemas()
    if cfg.backend == "anthropic":
        key = cfg.anthropic_api_key or ""
        return AnthropicBackend(key, cfg.model, cfg.system_prompt, schemas,
                                logger=logger)
    elif cfg.backend == "openai-realtime":
        key = cfg.openai_api_key or ""
        return OpenAIRealtimeBackend(key, cfg.model, cfg.system_prompt, schemas,
                                     max_reconnect_attempts=cfg.reconnect_attempts,
                                     logger=logger)
    elif cfg.backend == "mock":
        return MockBackend()
    else:
        raise ValueError(f"Unknown backend: {cfg.backend}")
