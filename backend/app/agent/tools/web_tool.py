"""Web Search Tool for live web information retrieval, source quality rating, and evidence extraction."""
from __future__ import annotations

import logging
import time
from typing import Any

import asyncio
from app.agent.tools.base import Tool, ToolInput, ToolMetadata, ToolOutput
from app.tools.page_fetcher import PageFetcher
from app.tools.web_search import WebSearchError, WebSearchHit, get_web_search_provider

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    """Modular tool for real-time web search, SSRF-safe page fetching, and evidence extraction."""

    def __init__(self, web_search: Any = None, page_fetcher: PageFetcher | None = None) -> None:
        super().__init__(
            ToolMetadata(
                name="web_search",
                description="Performs live web searches, evaluates search result quality, and extracts web evidence.",
                version="1.1.0",
                requires_network=True,
            )
        )
        self.provider = web_search if web_search is not None else get_web_search_provider()
        self.page_fetcher = page_fetcher if page_fetcher is not None else PageFetcher()

    async def execute(self, tool_input: ToolInput) -> ToolOutput:
        start_mono = time.monotonic()
        query = tool_input.query.strip()
        request_id = tool_input.parameters.get("request_id")
        
        max_results_val = tool_input.parameters.get("max_results", 5)
        try:
            max_results = int(max_results_val)
        except (ValueError, TypeError):
            max_results = 5

        recency_days_val = tool_input.parameters.get("recency_days")
        recency_days: int | None = None
        if recency_days_val is not None:
            try:
                recency_days = int(recency_days_val)
            except (ValueError, TypeError):
                recency_days = None

        fetch_pages = bool(tool_input.parameters.get("fetch_pages", True))

        if not query:
            return ToolOutput(
                success=False,
                data={"hits": [], "evidence": [], "count": 0},
                error="Query must not be empty.",
                execution_time_ms=0,
            )

        try:
            search_result = await self.provider.search(
                query,
                max_results=max_results,
                recency_days=recency_days,
                request_id=request_id,
            )

            hits_list: list[WebSearchHit] = list(search_result.hits)

            # Optionally fetch page content for top N hits (up to top 3 links)
            fetched_count = 0
            if fetch_pages and hits_list:
                fetch_tasks = []
                target_hits = hits_list[:3]
                for hit in target_hits:
                    if hit.url and hit.url.startswith(("http://", "https://")):
                        fetch_tasks.append(self.page_fetcher.fetch_and_extract(hit.url, request_id=request_id))
                    else:
                        fetch_tasks.append(asyncio.sleep(0, result=None))

                page_contents = await asyncio.gather(*fetch_tasks, return_exceptions=True)

                updated_hits: list[WebSearchHit] = []
                for idx, hit in enumerate(hits_list):
                    fetched_content: str | None = None
                    if idx < len(page_contents):
                        res = page_contents[idx]
                        if isinstance(res, str) and res.strip():
                            fetched_content = res.strip()
                            fetched_count += 1

                    updated_hits.append(
                        WebSearchHit(
                            title=hit.title,
                            url=hit.url,
                            snippet=hit.snippet,
                            source=hit.source,
                            published_at=hit.published_at,
                            content=fetched_content or hit.content,
                        )
                    )
                hits_list = updated_hits

            evidence_items = []
            structured_hits = []
            for idx, hit in enumerate(hits_list, 1):
                snippet = (hit.snippet or hit.title).strip()
                content_text = hit.content or snippet
                evidence_items.append({
                    "source_type": "web",
                    "source_name": hit.source or hit.title or f"Web Source {idx}",
                    "url": hit.url,
                    "content": content_text[:1500],
                    "relevance_score": 0.85 - (idx * 0.05),
                })
                structured_hits.append({
                    "title": hit.title,
                    "url": hit.url,
                    "snippet": hit.snippet,
                    "source": hit.source,
                    "published_at": hit.published_at,
                    "content": hit.content,
                })

            duration_ms = int((time.monotonic() - start_mono) * 1000)
            logger.info(
                "[WEB TOOL SUCCESS] query=%r hits_count=%d fetched_count=%d duration_ms=%d request_id=%s",
                query, len(hits_list), fetched_count, duration_ms, request_id or "N/A"
            )

            return ToolOutput(
                success=True,
                data={
                    "query": search_result.query,
                    "hits": structured_hits,
                    "evidence": evidence_items,
                    "concise_text": search_result.concise_answer(),
                    "count": len(hits_list),
                    "provider": search_result.provider,
                },
                execution_time_ms=duration_ms,
            )

        except WebSearchError as werr:
            duration_ms = int((time.monotonic() - start_mono) * 1000)
            logger.warning("[WEB TOOL CONTROLLED FAILURE] query=%r error=%s", query, werr)
            return ToolOutput(
                success=False,
                data={"hits": [], "evidence": [], "count": 0},
                error=str(werr),
                execution_time_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - start_mono) * 1000)
            logger.exception("[WEB TOOL UNEXPECTED ERROR] query=%r error=%s", query, exc)
            return ToolOutput(
                success=False,
                data={"hits": [], "evidence": [], "count": 0},
                error=str(exc),
                execution_time_ms=duration_ms,
            )
