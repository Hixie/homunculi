import unittest
from ...tools.search import invoke, search_tool


def make_ctx(payload):
    class FakeCtx:
        def __init__(self): self.sent = []
        async def request(self, env):
            self.sent.append(env)
            return {"id": env["id"], "type": "tool.content_response", "payload": payload}
    return FakeCtx()


def make_response(regions=None, total_lines=100, truncated=False, error=None):
    return {"resource":"f.py","total_lines":total_lines,
            "regions": regions or [], "truncated": truncated, "error": error}


class TestToolSearch(unittest.IsolatedAsyncioTestCase):
    async def test_sends_content_request_with_pattern(self):
        ctx = make_ctx(make_response())
        await invoke({"resource":"f.py","pattern":"foo"}, ctx)
        p = ctx.sent[0]["payload"]
        self.assertIn("pattern", p)
        self.assertNotIn("ranges", p)
        self.assertEqual(p["pattern"], "foo")

    async def test_max_results_default(self):
        ctx = make_ctx(make_response())
        await invoke({"resource":"f.py","pattern":"x"}, ctx)
        self.assertEqual(ctx.sent[0]["payload"]["max_results"], 20)

    async def test_context_lines_default(self):
        ctx = make_ctx(make_response())
        await invoke({"resource":"f.py","pattern":"x"}, ctx)
        self.assertEqual(ctx.sent[0]["payload"]["context_lines"], 2)

    async def test_match_marker_on_match_lines(self):
        regions = [{"start_line":5,"end_line":7,"lines":["a\n","target\n","b\n"],"match_lines":[6]}]
        ctx = make_ctx(make_response(regions=regions))
        result = await invoke({"resource":"f.py","pattern":"target"}, ctx)
        self.assertIn("▶", result)

    async def test_no_match_marker_on_context_lines(self):
        regions = [{"start_line":5,"end_line":7,"lines":["ctx1\n","match\n","ctx2\n"],"match_lines":[6]}]
        ctx = make_ctx(make_response(regions=regions))
        result = await invoke({"resource":"f.py","pattern":"match"}, ctx)
        lines = result.split("\n")
        ctx_lines = [l for l in lines if "ctx1" in l or "ctx2" in l]
        self.assertTrue(all("▶" not in l for l in ctx_lines))

    async def test_region_header_has_line_range(self):
        regions = [{"start_line":10,"end_line":14,"lines":["x\n"]*5,"match_lines":[12]}]
        ctx = make_ctx(make_response(regions=regions))
        result = await invoke({"resource":"f.py","pattern":"x"}, ctx)
        self.assertIn("Lines 10–14:", result)

    async def test_total_lines_in_header(self):
        regions = [{"start_line":1,"end_line":1,"lines":["x\n"],"match_lines":[1]}]
        ctx = make_ctx(make_response(regions=regions, total_lines=500))
        result = await invoke({"resource":"f.py","pattern":"x"}, ctx)
        self.assertIn("500", result)

    async def test_truncated_flagged(self):
        regions = [{"start_line":1,"end_line":1,"lines":["x\n"],"match_lines":[1]}]
        ctx = make_ctx(make_response(regions=regions, truncated=True))
        result = await invoke({"resource":"f.py","pattern":"x"}, ctx)
        self.assertIn("truncated", result)

    async def test_no_matches_return(self):
        ctx = make_ctx(make_response(regions=[]))
        result = await invoke({"resource":"f.py","pattern":"xyz"}, ctx)
        self.assertIn("no matches", result)

    async def test_host_error_returned(self):
        ctx = make_ctx(make_response(error="file not found"))
        result = await invoke({"resource":"f.py","pattern":"x"}, ctx)
        self.assertIn("file not found", result)

    def test_no_regex_syntax_in_schema(self):
        desc = search_tool.description.lower()
        self.assertNotIn("regex", desc)
