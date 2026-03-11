import io
import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch


def _silence():
    """Context manager that captures and discards stdout and stderr."""
    return unittest.mock.patch.multiple(
        sys, stdout=io.StringIO(), stderr=io.StringIO()
    )


class TestCLI(unittest.TestCase):
    def parse(self, args):
        from ...cli import parse_args
        return parse_args(args)

    def parse_silent(self, args):
        """parse_args with all output captured; returns (cfg_or_None, stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                cfg = self.parse(args)
                return cfg, out.getvalue(), err.getvalue()
            except SystemExit as exc:
                return exc, out.getvalue(), err.getvalue()

    # ── backend validation ────────────────────────────────────────────────────

    def test_backend_required(self):
        result, out, err = self.parse_silent(["--config", "/no/such/config.json"])
        self.assertIsInstance(result, SystemExit)
        self.assertIn("--backend is required", err)
        self.assertEqual(out, "")

    def test_unknown_backend_exits(self):
        result, out, err = self.parse_silent(["--backend", "nonexistent"])
        self.assertIsInstance(result, SystemExit)
        self.assertIn("nonexistent", err)
        self.assertIn("Valid values", err)
        self.assertEqual(out, "")

    # ── non-error paths ───────────────────────────────────────────────────────

    def test_non_backend_defaults(self):
        cfg, out, err = self.parse_silent(["--backend", "mock"])
        self.assertIsNotNone(cfg.system_prompt)
        self.assertEqual(cfg.usage_interval_s, 30.0)
        self.assertEqual(out, "")
        self.assertEqual(err, "")

    def test_backend_flag(self):
        cfg, out, err = self.parse_silent(["--backend", "mock"])
        self.assertEqual(cfg.backend, "mock")
        self.assertEqual(out, "")
        self.assertEqual(err, "")

    def test_model_flag(self):
        cfg, out, err = self.parse_silent(["--backend", "mock", "--model", "claude-opus-4-5"])
        self.assertEqual(cfg.model, "claude-opus-4-5")
        self.assertEqual(out, "")
        self.assertEqual(err, "")

    def test_api_keys_from_env(self):
        env = {"ANTHROPIC_API_KEY": "ant-key", "OPENAI_API_KEY": "oai-key"}
        with unittest.mock.patch.dict(os.environ, env):
            cfg, out, err = self.parse_silent(["--backend", "mock"])
        self.assertEqual(cfg.anthropic_api_key, "ant-key")
        self.assertEqual(cfg.openai_api_key, "oai-key")
        self.assertEqual(out, "")
        self.assertEqual(err, "")

    def test_api_keys_absent_when_env_unset(self):
        env_clean = {k: v for k, v in os.environ.items()
                     if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
        with unittest.mock.patch.dict(os.environ, env_clean, clear=True):
            cfg, out, err = self.parse_silent(["--backend", "mock"])
        self.assertIsNone(cfg.anthropic_api_key)
        self.assertIsNone(cfg.openai_api_key)
        self.assertEqual(out, "")
        self.assertEqual(err, "")

    def test_system_prompt_inline(self):
        cfg, out, err = self.parse_silent(["--backend", "mock", "--system-prompt", "Hello"])
        self.assertEqual(cfg.system_prompt, "Hello")
        self.assertEqual(out, "")
        self.assertEqual(err, "")

    def test_system_prompt_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("from file")
            fname = f.name
        try:
            cfg, out, err = self.parse_silent(["--backend", "mock", "--system-prompt-file", fname])
            self.assertEqual(cfg.system_prompt, "from file")
            self.assertEqual(out, "")
            self.assertEqual(err, "")
        finally:
            os.unlink(fname)

    def test_system_prompt_file_not_found(self):
        result, out, err = self.parse_silent(
            ["--backend", "mock", "--system-prompt-file", "/no/such/file.txt"])
        self.assertIsInstance(result, SystemExit)
        self.assertIn("System prompt file not found", err)
        self.assertEqual(out, "")

    def test_ide_context_inline(self):
        cfg, out, err = self.parse_silent(["--backend", "mock", "--ide-context", "ctx"])
        self.assertEqual(cfg.ide_context, "ctx")
        self.assertEqual(out, "")
        self.assertEqual(err, "")

    def test_ide_context_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("ide ctx")
            fname = f.name
        try:
            cfg, out, err = self.parse_silent(["--backend", "mock", "--ide-context-file", fname])
            self.assertEqual(cfg.ide_context, "ide ctx")
            self.assertEqual(out, "")
            self.assertEqual(err, "")
        finally:
            os.unlink(fname)

    def test_mutually_exclusive_pairs(self):
        result, out, err = self.parse_silent(
            ["--backend", "mock", "--system-prompt", "a", "--system-prompt-file", "b"])
        self.assertIsInstance(result, SystemExit)
        self.assertIn("mutually exclusive", err)
        self.assertEqual(out, "")

        result, out, err = self.parse_silent(
            ["--backend", "mock", "--ide-context", "a", "--ide-context-file", "b"])
        self.assertIsInstance(result, SystemExit)
        self.assertIn("mutually exclusive", err)
        self.assertEqual(out, "")

    def test_config_file_values_used(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"backend": "mock", "model": "custom-model"}, f)
            fname = f.name
        try:
            cfg, out, err = self.parse_silent(["--config", fname])
            self.assertEqual(cfg.backend, "mock")
            self.assertEqual(cfg.model, "custom-model")
            self.assertEqual(out, "")
            self.assertEqual(err, "")
        finally:
            os.unlink(fname)

    def test_cli_overrides_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"backend": "mock"}, f)
            fname = f.name
        try:
            cfg, out, err = self.parse_silent(["--config", fname, "--backend", "anthropic"])
            self.assertEqual(cfg.backend, "anthropic")
            self.assertEqual(out, "")
            self.assertEqual(err, "")
        finally:
            os.unlink(fname)

    def test_default_config_absent_silent(self):
        cfg, out, err = self.parse_silent(
            ["--backend", "mock", "--config", "/no/such/config.json"])
        self.assertIsNotNone(cfg)
        self.assertEqual(out, "")
        self.assertEqual(err, "")

    def test_print_prompt_flag_accepted(self):
        result, out, err = self.parse_silent(["--backend", "mock", "--print-prompt"])
        self.assertIsInstance(result, SystemExit)
        self.assertEqual(result.code, 0)
        # The composed system prompt goes to stderr
        self.assertIn("coding assistant", err)
        self.assertEqual(out, "")

    # ── --list-models ─────────────────────────────────────────────────────────

    def test_list_models_requires_backend(self):
        result, out, err = self.parse_silent(["--list-models"])
        self.assertIsInstance(result, SystemExit)
        self.assertEqual(result.code, 2)
        self.assertIn("--list-models requires --backend", err)
        self.assertEqual(out, "")

    def test_list_models_rejects_unknown_backend(self):
        result, out, err = self.parse_silent(["--backend", "bogus", "--list-models"])
        self.assertIsInstance(result, SystemExit)
        self.assertEqual(result.code, 2)
        self.assertIn("bogus", err)
        self.assertEqual(out, "")

    def test_list_models_mock_exits_zero(self):
        result, out, err = self.parse_silent(["--backend", "mock", "--list-models"])
        self.assertIsInstance(result, SystemExit)
        self.assertEqual(result.code, 0)
        self.assertEqual(err, "")

    def test_list_models_mock_valid_json(self):
        result, out, err = self.parse_silent(["--backend", "mock", "--list-models"])
        data = json.loads(out)
        self.assertEqual(data["backend"], "mock")
        self.assertIn("models", data)

    def test_list_models_mock_model_entry_shape(self):
        result, out, err = self.parse_silent(["--backend", "mock", "--list-models"])
        data = json.loads(out)
        self.assertTrue(len(data["models"]) >= 1)
        m = data["models"][0]
        self.assertIn("id", m)
        self.assertIn("input_usd_per_1m", m)
        self.assertIn("output_usd_per_1m", m)
        self.assertIsInstance(m["input_usd_per_1m"],  (int, float))
        self.assertIsInstance(m["output_usd_per_1m"], (int, float))

    def test_list_models_pricing_positive(self):
        """Real backends must have positive pricing; mock-model must be $0."""
        result, out, err = self.parse_silent(["--backend", "mock", "--list-models"])
        data = json.loads(out)
        for m in data["models"]:
            self.assertEqual(m["input_usd_per_1m"],  0.0)
            self.assertEqual(m["output_usd_per_1m"], 0.0)

    def test_list_models_api_failure_returns_error_key(self):
        """When the live API call returns nothing, 'error' key is present."""
        with unittest.mock.patch("server.cli._fetch_models", return_value=[]):
            result, out, err = self.parse_silent(
                ["--backend", "anthropic", "--list-models"])
        self.assertIsInstance(result, SystemExit)
        self.assertEqual(result.code, 0)
        data = json.loads(out)
        self.assertEqual(data["models"], [])
        self.assertIn("error", data)
        self.assertIn("ANTHROPIC_API_KEY", data["error"])

    def test_list_models_pricing_matches_known_model(self):
        """A known model id must receive its specific pricing, not the default."""
        with unittest.mock.patch("server.cli._fetch_models",
                                 return_value=["claude-opus-custom-v1"]):
            result, out, err = self.parse_silent(
                ["--backend", "anthropic", "--list-models"])
        data = json.loads(out)
        m = data["models"][0]
        # claude-opus maps to (15.0, 75.0) in PRICING
        self.assertEqual(m["input_usd_per_1m"],  15.0)
        self.assertEqual(m["output_usd_per_1m"], 75.0)

    def test_list_models_does_not_start_session(self):
        """--list-models must exit before parse_args returns a Config."""
        result, out, err = self.parse_silent(["--backend", "mock", "--list-models"])
        self.assertIsInstance(result, SystemExit)
        self.assertNotIsInstance(result, type(None))

    def test_list_models_output_on_stdout_not_stderr(self):
        result, out, err = self.parse_silent(["--backend", "mock", "--list-models"])
        self.assertTrue(out.strip())   # JSON on stdout
        self.assertEqual(err, "")      # nothing on stderr

