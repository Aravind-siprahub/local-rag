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

        # 3. Optional Deep Page Text Extraction
        fetch_ms = 0
        pages_fetched_count = 0
        if fetch_pages and unique_hits:
            fetch_start = time.monotonic()
            for idx, hit in enumerate(unique_hits[:max_pages_to_fetch]):
                if hit.content:
                    continue  # already has content
                try:
                    extracted_text = await self.page_fetcher.fetch_and_extract(
                        hit.url,
                        max_chars=2500,
                        request_id=req_id,
                    )
                    if extracted_text:
                        hit.content = extracted_text
                        pages_fetched_count += 1
                        logger.info("PAGE_FETCH_DEBUG: url=%s status=200 content_length=%d", hit.url, len(extracted_text))
                    else:
                        logger.info("PAGE_FETCH_DEBUG: url=%s status=empty content_length=0", hit.url)
                except Exception as fetch_exc:
                    logger.info("PAGE_FETCH_DEBUG: url=%s status=error error=%s", hit.url, fetch_exc)
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
    """Format web search hits into clean context string for LLM prompting."""
    if not hits:
        return "No web search context available."

    sections: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        domain_str = f" [{hit.source}]" if hit.source else ""
        date_str = f" (Published: {hit.published_at})" if hit.published_at else ""
        content_body = (hit.content or hit.snippet or "").strip()
        sections.append(
            f"Result [{idx}] {hit.title}{domain_str}{date_str}\n"
            f"URL: {hit.url}\n"
            f"Content: {content_body}"
        )

    return "\n\n".join(sections)
