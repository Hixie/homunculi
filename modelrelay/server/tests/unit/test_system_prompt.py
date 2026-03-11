import unittest
from ...prompt import compose_system_prompt


class TestSystemPrompt(unittest.TestCase):
    def test_no_context_returns_base(self):
        self.assertEqual(compose_system_prompt("base"), "base")

    def test_empty_context_returns_base(self):
        self.assertEqual(compose_system_prompt("base", ""), "base")

    def test_separator_present(self):
        result = compose_system_prompt("base", "ctx")
        self.assertIn("---", result)

    def test_heading_present(self):
        result = compose_system_prompt("base", "ctx")
        self.assertIn("## IDE Context", result)

    def test_ordering(self):
        result = compose_system_prompt("BASE", "CTX")
        self.assertLess(result.index("BASE"), result.index("CTX"))

    def test_base_and_context_unmodified(self):
        b, c = "base content", "ctx content"
        result = compose_system_prompt(b, c)
        self.assertIn(b, result)
        self.assertIn(c, result)

    def test_multiline_inputs(self):
        b = "line1\nline2"
        c = "ctx1\nctx2"
        result = compose_system_prompt(b, c)
        self.assertIn(b, result)
        self.assertIn(c, result)

    def test_special_characters(self):
        b = "Use `backtick` and /path/to/file"
        c = "cargo: target/debug/build.log"
        result = compose_system_prompt(b, c)
        self.assertIn(b, result)
        self.assertIn(c, result)
