"""Web search via Tavily — exposed through the worker's MCP server.

Requires TAVILY_API_KEY in the environment (provided via the mad-worker-secrets
Modal secret). Returns LLM-ready results: a short answer summary plus a list of
source URLs with titles and snippets.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


TAVILY_ENDPOINT = "https://api.tavily.com/search"
DEFAULT_MAX_RESULTS = 5


def web_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    search_depth: str = "basic",
    include_answer: bool = True,
) -> dict[str, Any]:
    """Search the web with Tavily.

    Args:
        query: Natural-language search query.
        max_results: How many source links to return (1–10).
        search_depth: "basic" (fast, 1 credit) or "advanced" (deeper, 2 credits).
        include_answer: Whether Tavily should synthesize a short answer.

    Returns a dict with keys: query, answer, results (list of {title, url, content, score}).
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {"error": "TAVILY_API_KEY is not set in the environment."}

    max_results = max(1, min(int(max_results or DEFAULT_MAX_RESULTS), 10))
    if search_depth not in ("basic", "advanced"):
        search_depth = "basic"

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": search_depth,
        "include_answer": bool(include_answer),
        "max_results": max_results,
    }

    try:
        resp = httpx.post(TAVILY_ENDPOINT, json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"Tavily HTTP {e.response.status_code}: {e.response.text[:300]}"}
    except httpx.HTTPError as e:
        return {"error": f"Tavily request failed: {type(e).__name__}: {e}"}

    results = []
    for r in data.get("results", []) or []:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score"),
        })

    return {
        "query": data.get("query", query),
        "answer": data.get("answer") or "",
        "results": results,
    }
