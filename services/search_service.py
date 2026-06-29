"""
Web search service — DuckDuckGo lookups exposed to Gemini as a tool.
"""
import asyncio

from duckduckgo_search import DDGS


def _duckduckgo_search(query: str, max_results: int = 5) -> list[dict]:
    """Blocking DDG search — always call via run_in_executor."""
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                if isinstance(r, dict) and r.get("title") and r.get("href"):
                    results.append({
                        "title": r["title"],
                        "url": r["href"],
                        "snippet": r.get("body", ""),
                    })
    except Exception as exc:
        print(f"[search error] {exc}", flush=True)
    return results


async def run_web_search(query: str) -> dict:
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, _duckduckgo_search, query, 5)
    if not results:
        return {"results": [], "note": "No results found."}
    return {"results": results}


# Tool declaration handed to Gemini Live so it knows this function exists.
WEB_SEARCH_DECLARATION = {
    "function_declarations": [
        {
            "name": "web_search",
            "description": (
                "Search the web via DuckDuckGo for current information, "
                "documentation, or examples. Returns a list of results with "
                "titles, URLs, and snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    }
                },
                "required": ["query"],
            },
        }
    ]
}