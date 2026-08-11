"""Web search tool providers for Agent Router v1."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)


class WebSearchError(Exception):
    """Raised when web search fails in a controlled way."""


@dataclass(frozen=True)
class WebSearchHit:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class WebSearchResult:
    query: str
    hits: list[WebSearchHit] = field(default_factory=list)
    provider: str = "unknown"

    def concise_answer(self) -> str:
        """Format hits into a short answer string for the chat response."""
        if not self.hits:
            return (
                "I could not find reliable web results for that question right now. "
                "Please try again shortly."
            )
        lines: list[str] = []
        for idx, hit in enumerate(self.hits[:5], start=1):
            snippet = hit.snippet.strip() or hit.title
            url_part = f" ({hit.url})" if hit.url else ""
            lines.append(f"{idx}. {hit.title}: {snippet}{url_part}")
        return "Here is what I found:\n" + "\n".join(lines)


@runtime_checkable
class WebSearchProvider(Protocol):
    """Vendor-agnostic web search contract."""

    async def search(self, query: str) -> WebSearchResult: ...


class StubWebSearchProvider:
    """Deterministic provider for tests and offline use — no network calls."""

    async def search(self, query: str) -> WebSearchResult:
        q = (query or "").strip() or "empty"
        logger.info("[WEB SEARCH] provider=stub query_len=%d", len(q))
        return WebSearchResult(
            query=q,
            provider="stub",
            hits=[
                WebSearchHit(
                    title="Stub web result",
                    url="https://example.com/stub",
                    snippet=(
                        f"Stub search result for: {q}. "
                        "Configure WEB_SEARCH_PROVIDER=duckduckgo for live results."
                    ),
                )
            ],
        )


class DuckDuckGoWebSearchProvider:
    """DuckDuckGo web search provider using keyless HTML and Instant Answer endpoints."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 8.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self._client = client
        self._owns_client = client is None

    async def search(self, query: str) -> WebSearchResult:
        q = (query or "").strip()
        if not q:
            raise WebSearchError("Search query must not be empty.")

        logger.info("[WEB SEARCH] provider=duckduckgo query='%s'", q)
        client = await self._get_client()

        # 1. Try HTML web search endpoint (live web results)
        hits = await self._search_html(client, q)

        # 2. Fall back to Instant Answer API if HTML returns 0 hits
        if not hits:
            logger.info("[WEB SEARCH] html search yielded 0 hits, trying Instant Answer API for '%s'", q)
            hits = await self._search_instant_answer(client, q)

        if not hits:
            logger.warning("[WEB SEARCH] provider=duckduckgo returned 0 hits for query='%s'", q)

        return WebSearchResult(query=q, hits=hits, provider="duckduckgo")

    async def _search_html(self, client: httpx.AsyncClient, query: str) -> list[WebSearchHit]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        # 1. Try POST to html.duckduckgo.com/html/
        try:
            url = "https://html.duckduckgo.com/html/"
            response = await client.post(url, data={"q": query}, headers=headers)
            logger.info("[WEB SEARCH] duckduckgo html POST status=%d len=%d", response.status_code, len(response.text))
            
            with open("scratch/ddg_response_dump.txt", "w", encoding="utf-8") as f:
                f.write(f"STATUS: {response.status_code}\nURL: {response.url}\nLEN: {len(response.text)}\n\nBODY:\n{response.text}")
                
            if response.status_code == 200:
                hits = _hits_from_ddg_html(response.text)
                if hits:
                    logger.info("[WEB SEARCH] duckduckgo html POST parsed %d hits", len(hits))
                    return hits
                logger.warning("[WEB SEARCH] duckduckgo html POST returned 200 but 0 hits parsed (len=%d)", len(response.text))
        except Exception as exc:
            logger.warning("[WEB SEARCH] duckduckgo html POST failed for '%s': %s", query, exc)

        # 2. Try GET to html.duckduckgo.com/html/?q=...
        try:
            url = "https://html.duckduckgo.com/html/"
            response = await client.get(url, params={"q": query}, headers=headers)
            logger.info("[WEB SEARCH] duckduckgo html GET status=%d len=%d", response.status_code, len(response.text))
            if response.status_code == 200:
                hits = _hits_from_ddg_html(response.text)
                if hits:
                    logger.info("[WEB SEARCH] duckduckgo html GET parsed %d hits", len(hits))
                    return hits
        except Exception as exc:
            logger.warning("[WEB SEARCH] duckduckgo html GET failed for '%s': %s", query, exc)

        # 3. Try POST to lite.duckduckgo.com/lite/
        try:
            url = "https://lite.duckduckgo.com/lite/"
            response = await client.post(url, data={"q": query}, headers=headers)
            logger.info("[WEB SEARCH] duckduckgo lite POST status=%d len=%d", response.status_code, len(response.text))
            if response.status_code == 200:
                hits = _hits_from_ddg_html(response.text)
                if hits:
                    logger.info("[WEB SEARCH] duckduckgo lite POST parsed %d hits", len(hits))
                    return hits
        except Exception as exc:
            logger.warning("[WEB SEARCH] duckduckgo lite POST failed for '%s': %s", query, exc)

        return []

    async def _search_instant_answer(self, client: httpx.AsyncClient, query: str) -> list[WebSearchHit]:
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
        try:
            response = await client.get("https://api.duckduckgo.com/", params=params)
            logger.info("[WEB SEARCH] duckduckgo instant answer status=%d", response.status_code)
            if response.status_code == 200:
                return _hits_from_duckduckgo(response.json())
            return []
        except Exception as exc:
            logger.warning("[WEB SEARCH] duckduckgo instant answer error: %s", exc)
            return []

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds),
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None


def _hits_from_ddg_html(html_text: str) -> list[WebSearchHit]:
    import html
    import re
    import urllib.parse

    hits: list[WebSearchHit] = []
    if not html_text:
        return hits

    # Extract snippets from class containing snippet
    snippet_matches = list(
        re.finditer(
            r'<(?:a|td|div|span|p)\s+class="[^"]*snippet[^"]*"[^>]*>(.*?)</(?:a|td|div|span|p)>',
            html_text,
            re.IGNORECASE | re.DOTALL,
        )
    )
    snippets: list[str] = []
    for sm in snippet_matches:
        s_raw = sm.group(1)
        s_clean = re.sub(r'<[^>]+>', '', s_raw).strip()
        s_clean = html.unescape(s_clean)
        if s_clean and len(s_clean) > 5:
            snippets.append(s_clean)

    # Strategy 1: Look for uddg link redirects with titles
    uddg_matches = list(
        re.finditer(
            r'<a\s+[^>]*href="([^"]*uddg=[^"]*)"[^>]*>(.*?)</a>',
            html_text,
            re.IGNORECASE | re.DOTALL,
        )
    )

    candidate_links: list[tuple[str, str]] = []
    for m in uddg_matches:
        raw_href = html.unescape(m.group(1))
        title_raw = m.group(2)
        title = re.sub(r'<[^>]+>', '', title_raw).strip()
        title = html.unescape(title)

        if not title or len(title) < 3 or title.lower() in ["next", "previous", "images", "videos", "news", "map"]:
            continue

        url = raw_href
        if "uddg=" in raw_href:
            parsed = urllib.parse.urlparse(raw_href)
            qs = urllib.parse.parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                url = qs["uddg"][0]
        elif raw_href.startswith("//"):
            url = "https:" + raw_href

        candidate_links.append((title, url))

    seen_urls: set[str] = set()
    for idx, (title, url) in enumerate(candidate_links):
        if len(hits) >= 5:
            break
        if url in seen_urls:
            continue
        seen_urls.add(url)

        snippet = snippets[idx] if idx < len(snippets) else title
        hits.append(WebSearchHit(title=title, url=url, snippet=snippet))

    if hits:
        return hits

    # Strategy 2: General HTML links with result__a or result-link
    result_a_matches = list(
        re.finditer(
            r'<a\s+class="[^"]*(?:result__a|result-link|links_main)[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            html_text,
            re.IGNORECASE | re.DOTALL,
        )
    )

    for i, a_match in enumerate(result_a_matches):
        if len(hits) >= 5:
            break
        raw_href = html.unescape(a_match.group(1))
        title_raw = a_match.group(2)
        title = re.sub(r'<[^>]+>', '', title_raw).strip()
        title = html.unescape(title)

        snippet = snippets[i] if i < len(snippets) else title

        url = raw_href
        if "uddg=" in raw_href:
            parsed = urllib.parse.urlparse(raw_href)
            qs = urllib.parse.parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                url = qs["uddg"][0]
        elif raw_href.startswith("//"):
            url = "https:" + raw_href

        if title and url and url not in seen_urls:
            seen_urls.add(url)
            hits.append(WebSearchHit(title=title, url=url, snippet=snippet))

    return hits


def _hits_from_duckduckgo(payload: dict) -> list[WebSearchHit]:
    hits: list[WebSearchHit] = []

    abstract = (payload.get("AbstractText") or "").strip()
    abstract_url = (payload.get("AbstractURL") or "").strip()
    heading = (payload.get("Heading") or payload.get("AbstractSource") or "DuckDuckGo").strip()
    if abstract:
        hits.append(
            WebSearchHit(
                title=heading or "Result",
                url=abstract_url,
                snippet=abstract,
            )
        )

    answer = (payload.get("Answer") or "").strip()
    if answer and answer != abstract:
        hits.append(
            WebSearchHit(
                title=heading or "Answer",
                url=abstract_url,
                snippet=answer,
            )
        )

    for topic in payload.get("RelatedTopics") or []:
        if len(hits) >= 5:
            break
        if not isinstance(topic, dict):
            continue
        if "Topics" in topic:
            for nested in topic.get("Topics") or []:
                if len(hits) >= 5:
                    break
                hit = _hit_from_related(nested)
                if hit:
                    hits.append(hit)
            continue
        hit = _hit_from_related(topic)
        if hit:
            hits.append(hit)

    return hits


def _hit_from_related(topic: dict) -> WebSearchHit | None:
    if not isinstance(topic, dict):
        return None
    text = (topic.get("Text") or "").strip()
    url = (topic.get("FirstURL") or "").strip()
    if not text:
        return None
    title = text.split(" - ", 1)[0][:120]
    return WebSearchHit(title=title, url=url, snippet=text)


def get_web_search_provider(
    *,
    provider_name: str | None = None,
    timeout_seconds: float | None = None,
) -> WebSearchProvider:
    """Factory: ``duckduckgo`` (default) or ``stub``."""
    from app.core.config import get_settings

    settings = get_settings()
    name = (provider_name or settings.WEB_SEARCH_PROVIDER or "duckduckgo").strip().lower()
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else settings.WEB_SEARCH_TIMEOUT_SECONDS
    )

    if name == "stub":
        logger.info("[WEB SEARCH] using provider=stub")
        return StubWebSearchProvider()
    if name == "duckduckgo":
        logger.info("[WEB SEARCH] using provider=duckduckgo")
        return DuckDuckGoWebSearchProvider(timeout_seconds=timeout)

    logger.warning("[WEB SEARCH] unknown provider=%s; falling back to duckduckgo", name)
    return DuckDuckGoWebSearchProvider(timeout_seconds=timeout)
