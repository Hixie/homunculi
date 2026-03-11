import unittest
from ...backends.anthropic import AnthropicBackend
from ...backends.openai_realtime import OpenAIRealtimeBackend
from ...tools import build_registry


class TestSchemaTranslation(unittest.TestCase):
    def setUp(self):
        self.schemas = build_registry().get_schemas()

    def test_openai_function_type(self):
        tools = OpenAIRealtimeBackend.translate_tools(self.schemas)
        for t in tools:
            self.assertEqual(t["type"], "function")

    def test_openai_parameters_key(self):
        tools = OpenAIRealtimeBackend.translate_tools(self.schemas)
        for t in tools:
            self.assertIn("parameters", t)
            self.assertNotIn("input_schema", t)

    def test_anthropic_input_schema_key(self):
        tools = AnthropicBackend.translate_tools(self.schemas)
        for t in tools:
            self.assertIn("input_schema", t)

    def test_anthropic_no_type_field(self):
        tools = AnthropicBackend.translate_tools(self.schemas)
        for t in tools:
            self.assertNotIn("type", t)

    def test_all_five_tools_translate(self):
        oa = OpenAIRealtimeBackend.translate_tools(self.schemas)
        an = AnthropicBackend.translate_tools(self.schemas)
        self.assertEqual(len(oa), 5)
        self.assertEqual(len(an), 5)

    def test_required_fields_preserved(self):
        an = AnthropicBackend.translate_tools(self.schemas)
        for t in an:
            schema = t["input_schema"]
            self.assertIn("required", schema)
