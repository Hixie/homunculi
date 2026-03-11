import unittest
from ...tools.registry import ToolRegistry, ToolHandler
from ...tools import build_registry


async def _noop(args, ctx): return "ok"


class TestToolRegistry(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.reg = ToolRegistry()
        self.handler = ToolHandler(name="mytool", description="desc",
                                   schema={"type": "object", "properties": {}},
                                   invoke_fn=_noop)

    def test_register_and_has(self):
        self.reg.register(self.handler)
        self.assertTrue(self.reg.has("mytool"))

    def test_get_schemas(self):
        self.reg.register(self.handler)
        schemas = self.reg.get_schemas()
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0].name, "mytool")

    async def test_invoke_delegates(self):
        self.reg.register(self.handler)
        result = await self.reg.invoke("mytool", {}, None)
        self.assertEqual(result, "ok")

    async def test_invoke_unknown_raises(self):
        with self.assertRaises(KeyError):
            await self.reg.invoke("unknown", {}, None)

    def test_build_registry_has_five_tools(self):
        r = build_registry()
        for name in ["read", "replace", "stat", "search", "insert"]:
            self.assertTrue(r.has(name))
