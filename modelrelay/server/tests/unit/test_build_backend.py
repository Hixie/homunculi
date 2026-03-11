"""Tests for build_backend() factory — covers reconnect_attempts wiring."""
import unittest


class TestBuildBackend(unittest.TestCase):

    def _cfg(self, **overrides):
        from ...cli import Config
        defaults = dict(
            backend="mock", model="mock-model",
            anthropic_api_key=None, openai_api_key=None,
            log_dir="/tmp", usage_interval_s=30.0, reconnect_attempts=3,
            system_prompt="", ide_context=None,
        )
        defaults.update(overrides)
        return Config(**defaults)

    def test_reconnect_attempts_passed_to_openai(self):
        """cfg.reconnect_attempts is forwarded to OpenAIRealtimeBackend."""
        from ...backends import build_backend
        cfg = self._cfg(backend="openai-realtime", reconnect_attempts=7,
                        openai_api_key="sk-test")
        backend = build_backend(cfg)
        self.assertEqual(backend._max_reconnect, 7)

    def test_mock_backend_returned_for_mock(self):
        from ...backends import build_backend
        from ...backends.mock import MockBackend
        cfg = self._cfg(backend="mock")
        backend = build_backend(cfg)
        self.assertIsInstance(backend, MockBackend)

    def test_anthropic_key_used(self):
        from ...backends import build_backend
        from ...backends.anthropic import AnthropicBackend
        cfg = self._cfg(backend="anthropic", anthropic_api_key="sk-ant-test")
        backend = build_backend(cfg)
        self.assertIsInstance(backend, AnthropicBackend)
        self.assertEqual(backend._api_key, "sk-ant-test")
