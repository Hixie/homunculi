import unittest
from ...tools.stat import invoke


def make_ctx(payload):
    class FakeCtx:
        def __init__(self): self.sent = []
        async def request(self, env):
            self.sent.append(env)
            return {"id": env["id"], "type": "tool.stat_response", "payload": payload}
    return FakeCtx()


class TestToolStat(unittest.IsolatedAsyncioTestCase):
    async def test_sends_stat_request(self):
        ctx = make_ctx({"resource":"f.py","exists":True,"total_lines":50,"last_modified":"2024-01-01T00:00:00Z","error":None})
        await invoke({"resource": "f.py"}, ctx)
        self.assertEqual(ctx.sent[0]["type"], "tool.stat_request")
        self.assertEqual(ctx.sent[0]["payload"]["resource"], "f.py")

    async def test_exists_true_return_string(self):
        ctx = make_ctx({"resource":"f.py","exists":True,"total_lines":1000,"last_modified":"2024-01-01T00:00:00Z","error":None})
        result = await invoke({"resource":"f.py"}, ctx)
        self.assertIn("1,000", result)
        self.assertIn("2024-01-01", result)

    async def test_exists_false_return_string(self):
        ctx = make_ctx({"resource":"missing.txt","exists":False,"total_lines":None,"last_modified":None,"error":None})
        result = await invoke({"resource":"missing.txt"}, ctx)
        self.assertIn("does not exist", result)

    async def test_no_capabilities_field_expected(self):
        # Response without capabilities field should parse correctly
        payload = {"resource":"f.py","exists":True,"total_lines":5,"last_modified":"now","error":None}
        ctx = make_ctx(payload)
        result = await invoke({"resource":"f.py"}, ctx)
        self.assertNotIn("capabilities", result)

    async def test_host_error_returned(self):
        ctx = make_ctx({"resource":"f.py","exists":False,"total_lines":None,"last_modified":None,"error":"permission denied"})
        result = await invoke({"resource":"f.py"}, ctx)
        self.assertIn("permission denied", result)
