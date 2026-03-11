"""replace tool."""
from __future__ import annotations
from ..stdio.protocol import replace_request as _replace_request
from .registry import ToolHandler

SCHEMA = {
    "type": "object",
    "properties": {
        "resource":    {"type": "string"},
        "start_line":  {"type": "integer"},
        "end_line":    {"type": "integer"},
        "new_content": {"type": "string"},
        "total_lines": {"type": "integer"},
    },
    "required": ["resource", "start_line", "end_line", "new_content", "total_lines"],
}


async def invoke(args: dict, ctx) -> str:
    resource    = args["resource"]
    start_line  = args["start_line"]
    end_line    = args["end_line"]
    new_content = args["new_content"]
    total_lines = args["total_lines"]

    if not new_content.endswith("\n"):
        new_content += "\n"
    new_lines = [l + "\n" for l in new_content.rstrip("\n").split("\n")]

    env = _replace_request(resource, start_line, end_line, new_lines, total_lines)
    resp = await ctx.request(env)
    p = resp.get("payload", {})
    if p.get("error"):
        return f"ERROR: replace(\"{resource}\") — {p['error']}"
    return f"Replaced lines {start_line}–{end_line} in {resource}."


replace_tool = ToolHandler(
    name="replace",
    description="Replace a range of lines in a file. You must provide the start_line, end_line, "
                "and total_lines exactly as returned by the read() or search() tool. Do not guess "
                "or compute these values yourself.",
    schema=SCHEMA,
    invoke_fn=invoke,
)
