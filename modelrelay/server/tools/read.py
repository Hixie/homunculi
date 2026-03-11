"""read tool."""
from __future__ import annotations
from ..stdio.protocol import content_request
from .registry import ToolHandler

SCHEMA = {
    "type": "object",
    "properties": {
        "resource": {"type": "string", "description": "File path."},
        "ranges": {
            "type": "array",
            "description": "One or more line ranges to read.",
            "items": {
                "type": "object",
                "properties": {
                    "start_line": {"type": "integer"},
                    "num_lines": {"oneOf": [{"type": "integer"},
                                            {"type": "string", "enum": ["all"]}]},
                },
                "required": ["start_line", "num_lines"],
            },
            "minItems": 1,
        },
    },
    "required": ["resource", "ranges"],
}


async def invoke(args: dict, ctx) -> str:
    resource = args["resource"]
    ranges = args["ranges"]
    env = content_request(resource, ranges=ranges)
    resp = await ctx.request(env)
    payload = resp["payload"]
    if payload.get("error"):
        return f"ERROR: {payload['error']}"
    total = payload.get("total_lines", "?")
    regions = payload.get("regions", [])
    parts = [f"Resource: {resource}  ({total:,} lines total)" if isinstance(total, int)
             else f"Resource: {resource}  ({total} lines total)"]
    for i, region in enumerate(regions):
        sl = region["start_line"]
        el = region["end_line"]
        lines = region.get("lines", [])
        parts.append(f"\nLines {sl}–{el}:")
        parts.append("".join(lines).rstrip("\n"))
        if i < len(regions) - 1:
            parts.append("\n---")
    return "\n".join(parts)


read_tool = ToolHandler(
    name="read",
    description="Read one or more line ranges from a file. Returns the exact lines requested. "
                "Use stat() first to check file size. Use search() to locate target regions in "
                "large files before reading.",
    schema=SCHEMA,
    invoke_fn=invoke,
)
