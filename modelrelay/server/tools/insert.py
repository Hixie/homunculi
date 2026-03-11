"""insert tool."""
from __future__ import annotations
from ..stdio.protocol import insert_request as _insert_request
from .registry import ToolHandler

SCHEMA = {
    "type": "object",
    "properties": {
        "resource":    {"type": "string", "description": "File path."},
        "after_line":  {"type": "integer", "description": "Insert after this line. 0=prepend."},
        "new_content": {"type": "string",  "description": "Content to insert."},
        "total_lines": {"type": "integer", "description": "Current file length from stat() or content_response."},
    },
    "required": ["resource", "after_line", "new_content", "total_lines"],
}


async def invoke(args: dict, ctx) -> str:
    resource    = args["resource"]
    after_line  = args["after_line"]
    new_content = args["new_content"]
    total_lines = args["total_lines"]

    if not new_content.endswith("\n"):
        new_content += "\n"
    new_lines = [l + "\n" for l in new_content.rstrip("\n").split("\n")]
    count = len(new_lines)

    env = _insert_request(resource, after_line, new_lines, total_lines)
    resp = await ctx.request(env)
    p = resp.get("payload", {})
    if p.get("error"):
        return f"ERROR: insert(\"{resource}\") — {p['error']}"
    return f"Inserted {count} line(s) after line {after_line} in {resource}."


insert_tool = ToolHandler(
    name="insert",
    description="Insert lines into a file without requiring a prior read(). Use after_line=0 "
                "to prepend, after_line=total_lines to append, or after_line=N to insert after "
                "line N.",
    schema=SCHEMA,
    invoke_fn=invoke,
)
