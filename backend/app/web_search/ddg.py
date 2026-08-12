"""DuckDuckGo web search — no API key, fast, privacy-respecting."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
from html.parser import HTMLParser
import httpx

logger = logging.getLogger(__name__)

class SimpleSnippetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.snippets = []
        self.in_snippet = False
        self.current_snippet = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        class_name = attrs_dict.get("class", "")
        classes = class_name.split() if class_name else []
        if tag == "a" and "result__snippet" in classes:
            self.in_snippet = True
            self.current_snippet = []

    def handle_endtag(self, tag):
        if tag == "a" and self.in_snippet:
            text = "".join(self.current_snippet).strip()
            if text:
                self.snippets.append(text)
            self.in_snippet = False

    def handle_data(self, data):
        if self.in_snippet:
            self.current_snippet.append(data)

_DDG_HTML_URL = "https://html.duckduckgo.com/html/"
_DDG_INSTANT_URL = "https://api.duckduckgo.com/"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


@dataclass
class SearchResult:
    snippet: str
    title: str = ""
    url: str = ""


async def web_search(query: str, *, max_results: int = 5, timeout: float = 8.0) -> list[SearchResult]:
    """Search DuckDuckGo and return up to max_results snippets.

    Tries instant answer API first (fast), then falls back to HTML scrape.
    """
    query = (query or "").strip().strip('"').strip("'").strip()
    results: list[SearchResult] = []

    # 1. Try instant answer (fast, structured)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                _DDG_INSTANT_URL,
                params={"q": query, "format": "json", "no_redirect": "1", "no_html": "1"},
            )
            if resp.status_code == 200:
                data = resp.json()
                abstract = data.get("AbstractText", "").strip()
                answer = data.get("Answer", "").strip()
                if answer:
                    results.append(SearchResult(snippet=answer, title="DuckDuckGo Instant Answer"))
                if abstract and abstract not in (r.snippet for r in results):
                    results.append(SearchResult(
                        snippet=abstract,
                        title=data.get("AbstractSource", ""),
                        url=data.get("AbstractURL", ""),
                    ))
                for topic in data.get("RelatedTopics", [])[:max_results]:
                    text = topic.get("Text", "").strip()
                    if text and text not in (r.snippet for r in results):
                        results.append(SearchResult(
                            snippet=text,
                            url=topic.get("FirstURL", ""),
                        ))
    except Exception as exc:
        logger.warning("DDG instant answer failed: %s", exc)

    if results:
        return results[:max_results]

    # 2. Fallback: HTML scrape
    try:
        async with httpx.AsyncClient(
            headers=_HEADERS, follow_redirects=True, timeout=timeout
        ) as client:
            resp = await client.post(_DDG_HTML_URL, data={"q": query})
            if resp.status_code == 200:
                parser = SimpleSnippetParser()
                parser.feed(resp.text)
                for snippet in parser.snippets:
                    results.append(SearchResult(snippet=snippet))
                    if len(results) >= max_results:
                        break
    except Exception as exc:
        logger.warning("DDG HTML scrape failed: %s", exc)

    return results[:max_results]


def format_web_results_as_context(results: list[SearchResult]) -> str:
    """Format search results into a readable context string for the LLM."""
    if not results:
        return ""
    lines = ["Web search results:\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"[Result {i}] {r.snippet}")
        if r.url:
            lines.append(f"Source: {r.url}")
        lines.append("")
    return "\n".join(lines)
