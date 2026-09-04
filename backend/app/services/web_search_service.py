"""Web Search Service orchestration layer.

Provides unified interface for open-source search engines (SearXNG, DuckDuckGo),
SSRF-safe page fetching, URL deduplication, and context construction for RAG.
"""
from __future__ import annotations

import logging
import time
import urllib.parse
from typing import Sequence

from app.core.config import get_settings
from app.tools.page_fetcher import PageFetcher, is_safe_url
from app.tools.web_search import (
    WebSearchError,
    WebSearchHit,
    WebSearchProvider,
    WebSearchResult,
    get_web_search_provider,
)

logger = logging.getLogger(__name__)


def is_article_url(url: str) -> bool:
    """Validate if a URL points to a specific article rather than a category, topic, tag, portal, or index page."""
    if not url:
        return False

    url_lower = url.lower().strip()
    parsed = urllib.parse.urlparse(url_lower)
    path = parsed.path.rstrip("/")

    # Root domain URLs without an article path are not specific articles
    if not path or path in ("", "/index.html", "/home", "/news", "/technology", "/technology/openai", "/latest/openai", "/technology/cybersecurity"):
        return False

    # Block category, topic, search, tag, newsroom, portal, and index pages
    forbidden_path_patterns = (
        "/topics/", "/topic/", "/tag/", "/tags/", "/search/", "/label/",
        "/newsroom", "/company-announcements", "/category/", "/categories/",
        "/archive/", "/feed/", "/labels/", "/rss/", "/all-news", "/verticals/",
        "/latest/openai", "/technology/openai"
    )
    if any(pattern in url_lower for pattern in forbidden_path_patterns):
        return False

    if "news.google.com/topics" in url_lower or "news.google.com/search" in url_lower:
        return False

    return True


class WebSearchService:
    """Service handling web search execution, page fetching, deduplication, and context construction."""

    def __init__(
        self,
        *,
        provider: WebSearchProvider | None = None,
        page_fetcher: PageFetcher | None = None,
    ) -> None:
        self.settings = get_settings()
        self.provider = provider or get_web_search_provider()
        self.page_fetcher = page_fetcher or PageFetcher(
            timeout_seconds=self.settings.WEB_SEARCH_TIMEOUT,
        )

    async def search_web(
        self,
        query: str,
        *,
        max_results: int | None = None,
        fetch_pages: bool = True,
        max_pages_to_fetch: int = 2,
        request_id: str | None = None,
    ) -> WebSearchResult:
        """Perform normalized web search with deduplication, security validation, and optional page text extraction."""
        req_id = request_id or "N/A"
        limit = max_results or self.settings.WEB_SEARCH_MAX_RESULTS
        start_time = time.monotonic()

        logger.info(
            "[WEB SEARCH SERVICE START] request_id=%s query=%r max_results=%d fetch_pages=%s",
            req_id,
            query,
            limit,
            fetch_pages,
        )

        if not getattr(self.settings, "WEB_SEARCH_ENABLED", True):
            logger.warning("[WEB SEARCH SERVICE] Web search is disabled by configuration (WEB_SEARCH_ENABLED=False).")
            raise WebSearchError("Web search is disabled in system configuration.")

        # 1. Execute search provider
        search_start = time.monotonic()
        try:
            raw_result = await self.provider.search(
                query,
                max_results=limit,
                request_id=req_id,
            )
        except WebSearchError:
            raise
        except Exception as exc:
            logger.exception("[WEB SEARCH SERVICE FAIL] request_id=%s error=%s", req_id, exc)
            raise WebSearchError(f"Web search execution failed: {exc}") from exc

        search_ms = int((time.monotonic() - search_start) * 1000)

        # 2. URL Deduplication & Security Validation
        unique_hits: list[WebSearchHit] = []
        seen_urls: set[str] = set()

        for hit in raw_result.hits:
            if not hit.url:
                continue
            clean_url = hit.url.strip().rstrip("/")
            if clean_url in seen_urls:
                continue

            # SSRF check
            is_safe, reason = is_safe_url(clean_url)
            if not is_safe:
                logger.warning(
                    "[WEB SEARCH SSRF REJECT] url=%r reason=%s request_id=%s",
                    clean_url,
                    reason,
                    req_id,
                )
                continue

            # Article URL validation (reject category, topic, tag, newsroom pages)
            if not is_article_url(clean_url):
                logger.info("[FILTERED_RESULTS REJECT] Non-article URL rejected: %s", clean_url)
                continue

            seen_urls.add(clean_url)
            domain = hit.source or urllib.parse.urlparse(clean_url).netloc.replace("www.", "")

            unique_hits.append(
                WebSearchHit(
                    title=hit.title or "Web Search Result",
                    url=clean_url,
                    snippet=hit.snippet or "",
                    source=domain or "web",
                    published_at=hit.published_at,
                    content=hit.content,
                )
            )

            if len(unique_hits) >= limit:
                break

        # Fallback: If strict article filtering excluded all results, include all SSRF-safe hits
        if not unique_hits:
            seen_urls.clear()
            for hit in raw_result.hits:
                if not hit.url:
                    continue
                clean_url = hit.url.strip().rstrip("/")
                if clean_url in seen_urls:
                    continue
                is_safe, _ = is_safe_url(clean_url)
                if not is_safe:
                    continue
                seen_urls.add(clean_url)
                domain = hit.source or urllib.parse.urlparse(clean_url).netloc.replace("www.", "")
                unique_hits.append(
                    WebSearchHit(
                        title=hit.title or "Web Search Result",
                        url=clean_url,
                        snippet=hit.snippet or "",
                        source=domain or "web",
                        published_at=hit.published_at,
                        content=hit.content,
                    )
                )
                if len(unique_hits) >= limit:
                    break

        # 3. Fast Parallel Deep Page Text Extraction (asyncio.gather)
        fetch_ms = 0
        pages_fetched_count = 0
        if fetch_pages and unique_hits:
            import asyncio
            fetch_start = time.monotonic()
            targets = unique_hits[:max_pages_to_fetch]

            async def _fetch_one(hit: WebSearchHit) -> None:
                if hit.content:
                    return
                try:
                    extracted_text, pub_date, headline = await self.page_fetcher.fetch_and_extract(
                        hit.url,
                        max_chars=2500,
                        request_id=req_id,
                    )
                    if pub_date and not hit.published_at:
                        hit.published_at = pub_date
                    if headline and (not hit.title or hit.title.lower().startswith(("http", "www.")) or "/" in hit.title):
                        hit.title = headline
                    if extracted_text:
                        hit.content = extracted_text
                except Exception as fetch_exc:
                    logger.debug("PAGE_FETCH_DEBUG: url=%s error=%s", hit.url, fetch_exc)

            try:
                # Concurrent fetch across all targets with 2.5s hard timeout
                await asyncio.wait_for(
                    asyncio.gather(*[_fetch_one(h) for h in targets], return_exceptions=True),
                    timeout=2.5,
                )
            except (asyncio.TimeoutError, TimeoutError):
                logger.info("[WEB SEARCH SERVICE] Page fetching 2.5s timeout reached. Continuing with gathered snippets.")
            except Exception as gather_exc:
                logger.warning("[WEB SEARCH SERVICE] Parallel page fetch error: %s", gather_exc)

            pages_fetched_count = sum(1 for h in targets if h.content)
            fetch_ms = int((time.monotonic() - fetch_start) * 1000)

        total_ms = int((time.monotonic() - start_time) * 1000)

        logger.info(
            "[WEB SEARCH SERVICE COMPLETE] request_id=%s provider=%s total_hits=%d pages_fetched=%d search_ms=%d fetch_ms=%d total_ms=%d",
            req_id,
            raw_result.provider,
            len(unique_hits),
            pages_fetched_count,
            search_ms,
            fetch_ms,
            total_ms,
        )

        return WebSearchResult(
            query=query,
            hits=unique_hits,
            provider=raw_result.provider,
        )


def format_web_context(hits: Sequence[WebSearchHit]) -> str:
    """Format web search hits into clean context string combining snippet and extracted page body for LLM prompting."""
    if not hits:
        return "No web search context available."

    sections: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        domain_str = f" [{hit.source}]" if hit.source else ""
        date_str = f" (Published: {hit.published_at})" if hit.published_at else ""
        snippet_part = hit.snippet.strip() if hit.snippet else ""
        content_part = hit.content.strip() if hit.content else ""

        if content_part and snippet_part and snippet_part not in content_part:
            combined_body = f"{snippet_part}\n{content_part}"
        else:
            combined_body = content_part or snippet_part or "No text content extracted."

        sections.append(
            f"Result [{idx}] {hit.title}{domain_str}{date_str}\n"
            f"URL: {hit.url}\n"
            f"Content Summary & Body:\n{combined_body}"
        )

    return "\n\n".join(sections)
