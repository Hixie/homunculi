"""stat tool."""
from __future__ import annotations
from ..stdio.protocol import stat_request as _stat_request
from .registry import ToolHandler

SCHEMA = {
    "type": "object",
    "properties": {
        "resource": {"type": "string", "description": "File path."},
    },
    "required": ["resource"],
}


async def invoke(args: dict, ctx) -> str:
    resource = args["resource"]
    env = _stat_request(resource)
    resp = await ctx.request(env)
    p = resp["payload"]
    if p.get("error"):
        return f'ERROR: stat("{resource}") — {p["error"]}'
    if not p.get("exists", False):
        return f"{resource}: does not exist."
    total = p.get("total_lines")
    modified = p.get("last_modified", "unknown")
    total_str = f"{total:,}" if isinstance(total, int) else str(total)
    return f"{resource}: {total_str} lines, last modified {modified}"


stat_tool = ToolHandler(
    name="stat",
    description="Get file metadata without reading content. Use before read() to check whether "
                "a file exists and how large it is.",
    schema=SCHEMA,
    invoke_fn=invoke,
)
