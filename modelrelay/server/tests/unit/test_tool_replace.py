import unittest
from ...tools.replace import invoke


class FakeCtx:
    """Simulates ToolContext.request(): captures the envelope and returns a
    configurable response dict."""
    def __init__(self, response=None):
        self.requests = []
        self._response = response or {"payload": {"error": None}}

    async def request(self, env):
        self.requests.append(env)
        return self._response


class TestToolReplace(unittest.IsolatedAsyncioTestCase):

    async def _run(self, args, response=None):
        ctx = FakeCtx(response=response)
        result = await invoke(args, ctx)
        return ctx, result

    async def test_sends_replace_request(self):
        ctx, _ = await self._run({"resource":"f.py","start_line":1,"end_line":3,
                                   "new_content":"x\n","total_lines":10})
        self.assertEqual(ctx.requests[0]["type"], "tool.replace_request")

    async def test_payload_has_all_required_fields(self):
        ctx, _ = await self._run({"resource":"f.py","start_line":1,"end_line":3,
                                   "new_content":"x\n","total_lines":10})
        p = ctx.requests[0]["payload"]
        for field in ("start_line","end_line","new_lines","total_lines"):
            self.assertIn(field, p)

    async def test_new_lines_split_correctly(self):
        ctx, _ = await self._run({"resource":"f.py","start_line":1,"end_line":2,
                                   "new_content":"line1\nline2\n","total_lines":10})
        self.assertEqual(len(ctx.requests[0]["payload"]["new_lines"]), 2)

    async def test_trailing_newline_normalised(self):
        ctx, _ = await self._run({"resource":"f.py","start_line":1,"end_line":1,
                                   "new_content":"no newline","total_lines":5})
        lines = ctx.requests[0]["payload"]["new_lines"]
        self.assertTrue(all(l.endswith("\n") for l in lines))

    async def test_return_string_confirms_range(self):
        _, result = await self._run({"resource":"f.py","start_line":3,"end_line":7,
                                      "new_content":"x\n","total_lines":10})
        self.assertIn("f.py", result)
        self.assertIn("3", result)
        self.assertIn("7", result)

    async def test_awaits_response_before_returning(self):
        """invoke() must call request() so it blocks until the host responds."""
        ctx, _ = await self._run({"resource":"f.py","start_line":1,"end_line":1,
                                   "new_content":"x\n","total_lines":5})
        self.assertEqual(len(ctx.requests), 1, "must use request(), not send()")

    async def test_error_response_surfaced(self):
        _, result = await self._run(
            {"resource":"f.py","start_line":1,"end_line":2,
             "new_content":"x\n","total_lines":5},
            response={"payload": {"error": "line range mismatch"}})
        self.assertIn("ERROR", result)
        self.assertIn("line range mismatch", result)
