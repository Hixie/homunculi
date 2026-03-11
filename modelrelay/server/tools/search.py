"""search tool."""
from __future__ import annotations
from ..stdio.protocol import content_request
from .registry import ToolHandler

SCHEMA = {
    "type": "object",
    "properties": {
        "resource":      {"type": "string", "description": "File path."},
        "pattern":       {"type": "string", "description": "Substring to find."},
        "max_results":   {"type": "integer", "description": "Maximum matching regions.", "default": 20},
        "context_lines": {"type": "integer", "description": "Lines of context before/after each match.", "default": 2},
    },
    "required": ["resource", "pattern"],
}


async def invoke(args: dict, ctx) -> str:
    resource      = args["resource"]
    pattern       = args["pattern"]
    max_results   = args.get("max_results", 20)
    context_lines = args.get("context_lines", 2)

    env = content_request(resource, pattern=pattern,
                          max_results=max_results, context_lines=context_lines)
    resp = await ctx.request(env)
    p = resp["payload"]

    if p.get("error"):
        return f"ERROR: {p['error']}"

    regions = p.get("regions", [])
    total   = p.get("total_lines", "?")
    total_str = f"{total:,}" if isinstance(total, int) else str(total)

    if not regions:
        return f'Search "{pattern}" in {resource} — no matches.'

    truncated = p.get("truncated", False)
    header = f'Search "{pattern}" in {resource} ({total_str} lines total) — {len(regions)} region(s):'
    parts = [header]

    for i, region in enumerate(regions):
        sl = region["start_line"]
        el = region["end_line"]
        lines = region.get("lines", [])
        match_lines = set(region.get("match_lines", []))
        parts.append(f"\nLines {sl}–{el}:")
        for j, line_text in enumerate(lines):
            abs_line = sl + j
            stripped = line_text.rstrip("\n")
            if abs_line in match_lines:
                parts.append(f"▶ {stripped}")
            else:
                parts.append(f"  {stripped}")
        if i < len(regions) - 1:
            parts.append("\n---")

    if truncated:
        parts.append("\n(truncated at max_results; narrow the pattern or increase the limit)")

    return "\n".join(parts)


search_tool = ToolHandler(
    name="search",
    description="Search for lines containing a substring in a file. Returns matching regions "
                "with surrounding context lines. Use to locate symbols or functions before "
                "reading their full content.",
    schema=SCHEMA,
    invoke_fn=invoke,
)
