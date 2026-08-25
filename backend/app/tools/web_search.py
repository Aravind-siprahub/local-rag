"""Web search tool providers for Agent Router v1."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx
from html.parser import HTMLParser

import time

logger = logging.getLogger(__name__)

class DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hits: list[WebSearchHit] = []
        self._current_title: list[str] = []
        self._current_snippet: list[str] = []
        self._current_url: str = ""
        self._in_title: bool = False
        self._in_snippet: bool = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        class_name = attrs_dict.get("class", "")
        classes = class_name.split() if class_name else []

        if tag == "a" and any(c in classes for c in ("result__a", "result__title", "large")):
            self._in_title = True
            href = attrs_dict.get("href", "")
            if href:
                self._current_url = href
        elif any(c in classes for c in ("result__snippet", "result__body")):
            self._in_snippet = True

    def handle_endtag(self, tag):
        if tag == "a" and self._in_title:
            self._in_title = False
        elif self._in_snippet:
            self._in_snippet = False
            title = "".join(self._current_title).strip()
            snippet = "".join(self._current_snippet).strip()
            if snippet:
                self.hits.append(
                    WebSearchHit(
                        title=title or "Search Result",
                        url=self._current_url,
                        snippet=snippet,
                    )
                )
            self._current_title = []
            self._current_snippet = []
            self._current_url = ""

    def handle_data(self, data):
        if self._in_title:
            self._current_title.append(data)
        elif self._in_snippet:
            self._current_snippet.append(data)

class BackupSnippetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.snippets = []
        self.in_snippet = False
        self.current_snippet = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        class_name = attrs_dict.get("class", "")
        classes = class_name.split() if class_name else []
        if tag in ("a", "td", "div") and "result__snippet" in classes:
            self.in_snippet = True
            self.current_snippet = []

    def handle_endtag(self, tag):
        if self.in_snippet:
            text = "".join(self.current_snippet).strip()
            if text:
                self.snippets.append(text)
            self.in_snippet = False

    def handle_data(self, data):
        if self.in_snippet:
            self.current_snippet.append(data)


import urllib.parse

class WebSearchError(Exception):
    """Raised when web search fails in a controlled way."""


@dataclass
class WebSearchHit:
    title: str
    url: str
    snippet: str
    source: str = "web"
    published_at: str | None = None
    content: str | None = None


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

    async def search(
        self,
        query: str,
        max_results: int = 5,
        recency_days: int | None = None,
        request_id: str | None = None,
    ) -> WebSearchResult: ...


class StubWebSearchProvider:
    """Deterministic provider for tests and offline use — no network calls."""

    async def search(
        self,
        query: str,
        max_results: int = 5,
        recency_days: int | None = None,
        request_id: str | None = None,
    ) -> WebSearchResult:
        q = (query or "").strip() or "empty"
        logger.info("[WEB SEARCH START] provider=stub query=%r request_id=%s", q, request_id or "N/A")
        result = WebSearchResult(
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
                    source="example.com",
                )
            ][: max_results if max_results > 0 else 5],
        )
        logger.info("[WEB SEARCH RESULT] provider=stub status=200 result_count=1 latency_ms=0 request_id=%s", request_id or "N/A")
        return result


class DuckDuckGoWebSearchProvider:
    """DuckDuckGo Instant Answer API + HTML Search fallback — no API key required."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 8.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self._client = client
        self._owns_client = client is None

    async def search(
        self,
        query: str,
        max_results: int = 5,
        recency_days: int | None = None,
        request_id: str | None = None,
    ) -> WebSearchResult:
        q = (query or "").strip().strip('"').strip("'").strip()
        if not q:
            raise WebSearchError("Search query must not be empty.")

        req_id = request_id or "N/A"
        start_mono = time.monotonic()
        logger.info("[WEB SEARCH START] provider=duckduckgo query=%r max_results=%d request_id=%s", q, max_results, req_id)

        hits: list[WebSearchHit] = []
        http_status: int | None = None

        # 1. Try Instant Answer API (fast, structured)
        try:
            client = await self._get_client()
            params = {
                "q": q,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }
            response = await client.get("https://api.duckduckgo.com/", params=params)
            http_status = response.status_code
            if response.status_code == 200:
                payload = response.json()
                hits = _hits_from_duckduckgo(payload)
        except httpx.TimeoutException:
            logger.warning("[WEB SEARCH] duckduckgo instant answer timeout")
        except httpx.HTTPError as exc:
            logger.warning("[WEB SEARCH] duckduckgo instant answer http error: %s", exc)
        except Exception as exc:
            logger.warning("[WEB SEARCH] duckduckgo instant answer unexpected error: %s", exc)

        logger.info("[WEB SEARCH API RESULT] request_id=%s api_hits=%d", req_id, len(hits))

        # 2. Fallback: HTML Scrape if Instant Answer returned 0 hits
        if not hits:
            logger.info("[WEB SEARCH FALLBACK START] request_id=%s url=https://html.duckduckgo.com/html/", req_id)
            try:
                client = await self._get_client()
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                }
                response = await client.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": q},
                    headers=headers,
                )
                http_status = response.status_code
                if response.status_code == 200:
                    parser = DuckDuckGoHTMLParser()
                    parser.feed(response.text)
                    hits = parser.hits
                    if not hits:
                        backup_parser = BackupSnippetParser()
                        backup_parser.feed(response.text)
                        hits = [
                            WebSearchHit(title="Search Result", url="", snippet=snip, source="duckduckgo")
                            for snip in backup_parser.snippets
                        ]
                logger.info("[WEB SEARCH FALLBACK RESULT] request_id=%s status=%s parsed_count=%d", req_id, http_status, len(hits))
            except httpx.TimeoutException as exc:
                latency_ms = int((time.monotonic() - start_mono) * 1000)
                logger.warning(
                    "[WEB SEARCH RESULT] provider=duckduckgo status=%s result_count=0 latency_ms=%d request_id=%s",
                    http_status or "timeout",
                    latency_ms,
                    req_id,
                )
                raise WebSearchError("Web search timed out. Please try again.") from exc
            except httpx.HTTPError as exc:
                latency_ms = int((time.monotonic() - start_mono) * 1000)
                logger.warning(
                    "[WEB SEARCH RESULT] provider=duckduckgo status=%s result_count=0 latency_ms=%d request_id=%s",
                    http_status or "http_error",
                    latency_ms,
                    req_id,
                )
                raise WebSearchError("Web search is temporarily unavailable.") from exc
            except Exception as exc:
                latency_ms = int((time.monotonic() - start_mono) * 1000)
                logger.exception(
                    "[WEB SEARCH RESULT] provider=duckduckgo status=%s result_count=0 latency_ms=%d request_id=%s",
                    http_status or "error",
                    latency_ms,
                    req_id,
                )
                raise WebSearchError("Web search failed.") from exc

        # Deduplicate hits by URL and extract source domains
        unique_hits: list[WebSearchHit] = []
        seen_urls: set[str] = set()
        for h in hits:
            clean_url = (h.url or "").strip().rstrip("/")
            if clean_url and clean_url in seen_urls:
                continue
            if clean_url:
                seen_urls.add(clean_url)

            source_name = h.source
            if (source_name == "web" or not source_name) and h.url:
                try:
                    netloc = urllib.parse.urlparse(h.url).netloc
                    if netloc:
                        source_name = netloc.replace("www.", "")
                except Exception:
                    pass

            unique_hits.append(
                WebSearchHit(
                    title=h.title or "Search Result",
                    url=h.url,
                    snippet=h.snippet,
                    source=source_name or "web",
                    published_at=h.published_at,
                    content=h.content,
                )
            )

        # Prioritize GitHub results if searching GitHub
        if "github" in q.lower():
            gh_hits = [h for h in unique_hits if "github.com" in (h.url or "").lower() or "github" in (h.source or "").lower()]
            other_hits = [h for h in unique_hits if h not in gh_hits]
            unique_hits = gh_hits + other_hits

        latency_ms = int((time.monotonic() - start_mono) * 1000)
        logger.info(
            "[WEB SEARCH RESULT] provider=duckduckgo status=%s result_count=%d latency_ms=%d request_id=%s",
            http_status if http_status is not None else 200,
            len(unique_hits),
            latency_ms,
            req_id,
        )

        if not unique_hits:
            raise WebSearchError("Web search yielded no results. Please try again.")

        limit = max_results if max_results and max_results > 0 else 5
        final_hits = unique_hits[:limit]

        logger.info("[WEB SEARCH] query=%r", q)
        logger.info("[WEB SEARCH RESULTS] provider=\"duckduckgo\" result_count=%d", len(final_hits))
        for hit in final_hits:
            logger.info("[WEB SEARCH RESULT] title=%r url=%r", hit.title, hit.url)
        logger.info("[WEB SEARCH CONTEXT] results_passed_to_llm=%d", len(final_hits))
        logger.info("[WEB SEARCH COMPLETE] success=true")

        return WebSearchResult(query=q, hits=final_hits, provider="duckduckgo")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds),
                headers={"User-Agent": "local-rag-agent-router/1.0"},
            )
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None


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
    name = (provider_name or getattr(settings, "WEB_SEARCH_PROVIDER", None) or "duckduckgo")
    name = str(name).strip().lower()
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else float(getattr(settings, "WEB_SEARCH_TIMEOUT_SECONDS", 8.0))
    )

    if name == "stub":
        logger.info("[WEB SEARCH] using provider=stub")
        return StubWebSearchProvider()
    if name == "duckduckgo":
        logger.info("[WEB SEARCH] using provider=duckduckgo")
        return DuckDuckGoWebSearchProvider(timeout_seconds=timeout)

    logger.warning("[WEB SEARCH] unknown provider=%s; falling back to duckduckgo", name)
    return DuckDuckGoWebSearchProvider(timeout_seconds=timeout)
