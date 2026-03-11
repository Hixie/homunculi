import asyncio
import unittest


class TestCommandBus(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from ...bus.command_bus import CommandBus
        self.bus = CommandBus()

    async def test_subscribe_and_publish(self):
        received = []
        self.bus.subscribe("test", lambda e: received.append(e))
        await self.bus.publish("test", {"x": 1})
        self.assertEqual(received, [{"x": 1}])

    async def test_multiple_handlers(self):
        a, b = [], []
        self.bus.subscribe("t", lambda e: a.append(e))
        self.bus.subscribe("t", lambda e: b.append(e))
        await self.bus.publish("t", {"v": 2})
        self.assertEqual(len(a), 1)
        self.assertEqual(len(b), 1)

    async def test_handler_exception_isolated(self):
        good = []
        def bad(e): raise RuntimeError("fail")
        self.bus.subscribe("t", bad)
        self.bus.subscribe("t", lambda e: good.append(e))
        await self.bus.publish("t", {})
        self.assertEqual(len(good), 1)

    async def test_unknown_topic_no_op(self):
        await self.bus.publish("no_such_topic", {})  # should not raise

    async def test_sync_handler_accepted(self):
        results = []
        self.bus.subscribe("t", lambda e: results.append("sync"))
        await self.bus.publish("t", {})
        self.assertEqual(results, ["sync"])
