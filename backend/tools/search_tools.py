# backend/tools/search_tools.py
from duckduckgo_search import DDGS
from langchain.tools import Tool
import textwrap

def _ddg_search(q: str) -> str:
    with DDGS() as ddg:
        results = ddg.text(q, max_results=5)
    lines = []
    for r in results or []:
        lines.append(f"- {r.get('title')}: {r.get('href')} — {r.get('body')}")
    return textwrap.shorten("\n".join(lines) or "No results", width=2000)

WEB_SEARCH_TOOL = Tool.from_function(
    name="WebSearch",
    func=_ddg_search,
    description=(
        "General web search. Use when the answer likely requires information not in the local docs."
    ),
)
