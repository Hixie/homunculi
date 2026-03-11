import unittest
from ...tools.insert import invoke


class FakeCtx:
    """Simulates ToolContext.request(): captures the envelope and returns a
    configurable response dict."""
    def __init__(self, response=None):
        self.requests = []
        self._response = response or {"payload": {"error": None}}

    async def request(self, env):
        self.requests.append(env)
        return self._response


class TestToolInsert(unittest.IsolatedAsyncioTestCase):

    async def test_sends_insert_request(self):
        ctx = FakeCtx()
        await invoke({"resource":"f.py","after_line":5,"new_content":"x\n","total_lines":10}, ctx)
        self.assertEqual(ctx.requests[0]["type"], "tool.insert_request")
        p = ctx.requests[0]["payload"]
        self.assertEqual(p["after_line"],  5)
        self.assertEqual(p["total_lines"], 10)

    async def test_prepend_after_line_zero(self):
        ctx = FakeCtx()
        await invoke({"resource":"f.py","after_line":0,"new_content":"x\n","total_lines":5}, ctx)
        self.assertEqual(ctx.requests[0]["payload"]["after_line"], 0)

    async def test_append_at_total_lines(self):
        ctx = FakeCtx()
        await invoke({"resource":"f.py","after_line":10,"new_content":"x\n","total_lines":10}, ctx)
        self.assertEqual(ctx.requests[0]["payload"]["after_line"], 10)

    async def test_trailing_newline_normalised(self):
        ctx = FakeCtx()
        await invoke({"resource":"f.py","after_line":0,"new_content":"no newline","total_lines":5}, ctx)
        lines = ctx.requests[0]["payload"]["new_lines"]
        self.assertTrue(all(l.endswith("\n") for l in lines))

    async def test_total_lines_required(self):
        ctx = FakeCtx()
        with self.assertRaises((KeyError, TypeError)):
            await invoke({"resource":"f.py","after_line":0,"new_content":"x\n"}, ctx)

    async def test_return_string_confirms_insert(self):
        ctx = FakeCtx()
        result = await invoke({"resource":"f.py","after_line":3,"new_content":"a\nb\n","total_lines":10}, ctx)
        self.assertIn("f.py", result)
        self.assertIn("2", result)  # line count

    async def test_awaits_response_before_returning(self):
        """invoke() must call request() (not send()) so it blocks for the host."""
        ctx = FakeCtx()
        await invoke({"resource":"f.py","after_line":1,"new_content":"x\n","total_lines":5}, ctx)
        self.assertEqual(len(ctx.requests), 1, "must use request(), not send()")

    async def test_error_response_surfaced(self):
        ctx = FakeCtx(response={"payload": {"error": "disk full"}})
        result = await invoke({"resource":"f.py","after_line":0,"new_content":"x\n","total_lines":5}, ctx)
        self.assertIn("ERROR", result)
        self.assertIn("disk full", result)

    async def test_success_response_gives_count(self):
        ctx = FakeCtx()
        result = await invoke({"resource":"f.py","after_line":0,
                               "new_content":"a\nb\nc\n","total_lines":5}, ctx)
        self.assertIn("3", result)
