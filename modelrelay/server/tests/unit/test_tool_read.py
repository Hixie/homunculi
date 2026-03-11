import asyncio
import unittest
import uuid
from ...tools.read import invoke


def make_ctx(payload):
    """Fake ToolContext that resolves immediately with given payload."""
    class FakeCtx:
        def __init__(self):
            self.sent = []
        async def request(self, env):
            self.sent.append(env)
            return {"id": env["id"], "type": "tool.content_response",
                    "payload": payload}
        async def send(self, env):
            self.sent.append(env)
    return FakeCtx()


def make_response(resource="f.py", total_lines=100, regions=None, error=None, truncated=False):
    return {"resource": resource, "total_lines": total_lines,
            "regions": regions or [], "truncated": truncated, "error": error}


class TestToolRead(unittest.IsolatedAsyncioTestCase):
    async def test_sends_content_request_with_ranges(self):
        ctx = make_ctx(make_response(regions=[{"start_line":1,"end_line":5,"lines":["a\n"]*5}]))
        await invoke({"resource": "f.py", "ranges": [{"start_line":1,"num_lines":5}]}, ctx)
        req = ctx.sent[0]
        self.assertEqual(req["type"], "tool.content_request")
        self.assertIn("ranges", req["payload"])
        self.assertNotIn("start_line", req["payload"])

    async def test_single_range_in_ranges(self):
        ctx = make_ctx(make_response(regions=[{"start_line":1,"end_line":1,"lines":["x\n"]}]))
        await invoke({"resource": "f.py", "ranges": [{"start_line":1,"num_lines":1}]}, ctx)
        self.assertEqual(len(ctx.sent[0]["payload"]["ranges"]), 1)

    async def test_multi_range_in_ranges(self):
        regions = [{"start_line":1,"end_line":2,"lines":["a\n","b\n"]},
                   {"start_line":10,"end_line":11,"lines":["x\n","y\n"]}]
        ctx = make_ctx(make_response(regions=regions))
        result = await invoke({"resource":"f.py","ranges":[{"start_line":1,"num_lines":2},
                                                            {"start_line":10,"num_lines":2}]}, ctx)
        self.assertEqual(len(ctx.sent[0]["payload"]["ranges"]), 2)

    async def test_return_string_header_has_total_lines(self):
        ctx = make_ctx(make_response(total_lines=500, regions=[{"start_line":1,"end_line":1,"lines":["x\n"]}]))
        result = await invoke({"resource":"f.py","ranges":[{"start_line":1,"num_lines":1}]}, ctx)
        self.assertIn("500", result)

    async def test_return_string_no_per_line_numbers(self):
        ctx = make_ctx(make_response(regions=[{"start_line":5,"end_line":5,"lines":["hello\n"]}]))
        result = await invoke({"resource":"f.py","ranges":[{"start_line":5,"num_lines":1}]}, ctx)
        import re
        self.assertFalse(re.search(r"^\d+:", result, re.MULTILINE))

    async def test_return_string_region_headers(self):
        ctx = make_ctx(make_response(regions=[{"start_line":3,"end_line":5,"lines":["a\n","b\n","c\n"]}]))
        result = await invoke({"resource":"f.py","ranges":[{"start_line":3,"num_lines":3}]}, ctx)
        self.assertIn("Lines 3–5:", result)

    async def test_multi_region_has_dividers(self):
        regions = [{"start_line":1,"end_line":1,"lines":["a\n"]},
                   {"start_line":5,"end_line":5,"lines":["b\n"]}]
        ctx = make_ctx(make_response(regions=regions))
        result = await invoke({"resource":"f.py","ranges":[{"start_line":1,"num_lines":1},
                                                            {"start_line":5,"num_lines":1}]}, ctx)
        self.assertIn("---", result)

    async def test_num_lines_all_forwarded(self):
        ctx = make_ctx(make_response(regions=[{"start_line":1,"end_line":10,"lines":["x\n"]*10}]))
        await invoke({"resource":"f.py","ranges":[{"start_line":1,"num_lines":"all"}]}, ctx)
        sent_range = ctx.sent[0]["payload"]["ranges"][0]
        self.assertEqual(sent_range["num_lines"], "all")

    async def test_host_error_returned(self):
        ctx = make_ctx(make_response(error="permission denied"))
        result = await invoke({"resource":"f.py","ranges":[{"start_line":1,"num_lines":1}]}, ctx)
        self.assertIn("permission denied", result)

    async def test_no_pending_future_after_response(self):
        # ToolContext.request pops the future after resolution
        pending = {}
        class RealCtx:
            async def request(self_, env):
                return {"id": env["id"], "type": "tool.content_response",
                        "payload": make_response(regions=[{"start_line":1,"end_line":1,"lines":["x\n"]}])}
        await invoke({"resource":"f.py","ranges":[{"start_line":1,"num_lines":1}]}, RealCtx())
        self.assertEqual(pending, {})
