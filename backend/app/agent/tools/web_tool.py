"""Web Search Tool for live web information retrieval, source quality rating, and evidence extraction."""
from __future__ import annotations

import logging
import time
from typing import Any

from app.agent.tools.base import Tool, ToolInput, ToolMetadata, ToolOutput
from app.tools.web_search import WebSearchError, get_web_search_provider

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    """Modular tool for real-time web search and evidence extraction."""

    def __init__(self, web_search: Any = None) -> None:
        super().__init__(
            ToolMetadata(
                name="web_search",
                description="Performs live web searches, evaluates search result quality, and extracts web evidence.",
                version="1.0.0",
                requires_network=True,
            )
        )
        self.provider = web_search if web_search is not None else get_web_search_provider()

    async def execute(self, tool_input: ToolInput) -> ToolOutput:
        start_mono = time.monotonic()
        query = tool_input.query.strip()
        request_id = tool_input.parameters.get("request_id")

        if not query:
            return ToolOutput(
                success=False,
                data={"hits": [], "evidence": [], "count": 0},
                error="Query must not be empty.",
                execution_time_ms=0,
            )

        try:
            search_result = await self.provider.search(query, request_id=request_id)
            evidence_items = []
            for idx, hit in enumerate(search_result.hits, 1):
                snippet = (hit.snippet or hit.title).strip()
                evidence_items.append({
                    "source_type": "web",
                    "source_name": hit.title or f"Web Source {idx}",
                    "url": hit.url,
                    "content": snippet,
                    "relevance_score": 0.85 - (idx * 0.05),
                })

            duration_ms = int((time.monotonic() - start_mono) * 1000)
            logger.info(
                "[WEB TOOL SUCCESS] query=%r hits_count=%d duration_ms=%d",
                query, len(search_result.hits), duration_ms
            )

            return ToolOutput(
                success=True,
                data={
                    "query": search_result.query,
                    "hits": search_result.hits,
                    "evidence": evidence_items,
                    "concise_text": search_result.concise_answer(),
                    "count": len(search_result.hits),
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
