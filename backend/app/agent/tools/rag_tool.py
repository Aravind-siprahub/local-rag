"""Document RAG Tool for hybrid vector retrieval, reranking, and evidence extraction."""
from __future__ import annotations

import logging
import time
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tools.base import Tool, ToolInput, ToolMetadata, ToolOutput
from app.retrieval.retriever import Retriever
from app.retrieval.search import SearchFilters
from app.rag.service import _filter_relevant_chunks
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class DocumentRAGTool(Tool):
    """Modular tool for document RAG retrieval, reranking, and relevance gate validation."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(
            ToolMetadata(
                name="document_rag",
                description="Performs hybrid vector/fulltext retrieval, cross-encoder reranking, and evidence extraction over uploaded document chunks.",
                version="1.0.0",
                requires_gpu=False,
            )
        )
        self.session = session
        self.retriever = Retriever(session)

    async def execute(self, tool_input: ToolInput) -> ToolOutput:
        start_mono = time.monotonic()
        query = tool_input.query.strip()
        params = tool_input.parameters

        user_id = params.get("user_id")
        document_id = params.get("document_id")
        document_version_id = params.get("document_version_id")
        top_k = params.get("top_k", get_settings().TOP_K)
        similarity_threshold = params.get("similarity_threshold", get_settings().SIMILARITY_THRESHOLD)

        filters = SearchFilters(
            user_id=user_id,
            document_id=document_id,
            document_version_id=document_version_id,
        )

        try:
            # Stage 1: Hybrid Retrieval
            retrieved_chunks = await self.retriever.retrieve(
                query,
                filters=filters,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )

            # Fallback 1: Retrive without document_id restriction if 0 chunks hit
            if not retrieved_chunks and (document_id or document_version_id):
                logger.info("[RAG TOOL] 0 chunks with document filter. Retrying with global filters.")
                relaxed_filters = SearchFilters(user_id=user_id)
                retrieved_chunks = await self.retriever.retrieve(
                    query,
                    filters=relaxed_filters,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                )

            # Fallback 2: Global cross-user retry if 0 chunks hit for user_id
            if not retrieved_chunks and user_id is not None:
                logger.info("[RAG TOOL] 0 chunks for user_id=%s. Retrying globally.", user_id)
                unrestricted_filters = SearchFilters(user_id=None)
                retrieved_chunks = await self.retriever.retrieve(
                    query,
                    filters=unrestricted_filters,
                    top_k=top_k,
                    similarity_threshold=0.10,
                )

            # Stage 2: Relevance Gate Filtering
            relevant_chunks = _filter_relevant_chunks(query, retrieved_chunks)

            # Stage 3: Evidence Extraction
            evidence_items = []
            for c in relevant_chunks:
                evidence_items.append({
                    "chunk_id": str(c.chunk_id),
                    "document_id": str(c.document_id),
                    "document_title": getattr(c, "document_title", "Uploaded Document"),
                    "section_title": getattr(c, "section_title", "Document Section"),
                    "content": c.chunk_text,
                    "relevance_score": float(c.similarity_score),
                })

            duration_ms = int((time.monotonic() - start_mono) * 1000)
            logger.info(
                "[RAG TOOL SUCCESS] query=%r raw_hits=%d relevant_hits=%d duration_ms=%d",
                query, len(retrieved_chunks), len(relevant_chunks), duration_ms
            )

            return ToolOutput(
                success=True,
                data={
                    "chunks": relevant_chunks,
                    "evidence": evidence_items,
                    "count": len(relevant_chunks),
                    "raw_count": len(retrieved_chunks),
                },
                execution_time_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = int((time.monotonic() - start_mono) * 1000)
            logger.exception("[RAG TOOL FAILED] error=%s", exc)
            return ToolOutput(
                success=False,
                data={"chunks": [], "evidence": [], "count": 0},
                error=str(exc),
                execution_time_ms=duration_ms,
            )
