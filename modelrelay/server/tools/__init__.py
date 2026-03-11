from .registry import ToolRegistry
from .read    import read_tool
from .replace import replace_tool
from .stat    import stat_tool
from .search  import search_tool
from .insert  import insert_tool


def build_registry() -> ToolRegistry:
    r = ToolRegistry()
    for tool in [read_tool, replace_tool, stat_tool, search_tool, insert_tool]:
        r.register(tool)
    return r
