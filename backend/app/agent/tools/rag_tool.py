"""Document RAG Tool for hybrid vector retrieval, reranking, and evidence extraction."""
from __future__ import annotations

import logging
import time
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tools.base import Tool, ToolInput, ToolMetadata, ToolOutput
from app.retrieval.retriever import Retriever
from app.retrieval.ranking import RankedResult
from app.retrieval.search import SearchFilters
from app.rag.service import _filter_relevant_chunks
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class DocumentRAGTool(Tool):
    """Modular tool for document RAG retrieval, reranking, and relevance gate validation."""

    def __init__(self, session: AsyncSession | None = None, retriever: Any = None) -> None:
        super().__init__(
            ToolMetadata(
                name="document_rag",
                description="Performs hybrid vector/fulltext retrieval, cross-encoder reranking, and evidence extraction over uploaded document chunks.",
                version="1.0.0",
                requires_gpu=False,
            )
        )
        self.session = session
        self.retriever = retriever if retriever is not None else Retriever(session)

    async def execute(self, tool_input: ToolInput) -> ToolOutput:
        start_mono = time.monotonic()
        query = tool_input.query.strip()
        params = tool_input.parameters

        user_id = params.get("user_id")
        document_id = params.get("document_id")
        document_ids = params.get("document_ids")
        document_version_id = params.get("document_version_id")
        top_k = params.get("top_k", get_settings().TOP_K)
        similarity_threshold = params.get("similarity_threshold", get_settings().SIMILARITY_THRESHOLD)

        filters = SearchFilters(
            user_id=user_id,
            document_id=document_id,
            document_ids=document_ids,
            document_version_id=document_version_id,
        )

        route_str = params.get("route")
        from app.rag.intent_router import _is_document_summary, _is_document_detail, Route
        from app.rag.query_understanding import extract_query_intent, AttributeCategory
        intent = extract_query_intent(query)
        is_policy_overview = intent.category == AttributeCategory.POLICY_GENERAL
        is_summary_or_detail = (
            route_str in (Route.DOCUMENT_SUMMARY.value, Route.DOCUMENT_DETAIL.value, "DOCUMENT_SUMMARY", "DOCUMENT_DETAIL")
            or _is_document_summary(query.lower())
            or _is_document_detail(query.lower())
            or is_policy_overview
        )

        try:
            # Stage 1: Hybrid or Section-Aware Retrieval
            if is_summary_or_detail:
                logger.info("[RAG TOOL] Executing section-aware retrieval for summary/detail query=%r", query)
                retrieved_chunks = await self.retriever.retrieve_section_aware(
                    query,
                    filters=filters,
                    max_total_chunks=5,
                )
            else:
                retrieved_chunks = await self.retriever.retrieve(
                    query,
                    filters=filters,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                )

            has_doc_filter = bool(document_id or document_ids or document_version_id)
            # Fallback 1: If scoped to a document and 0 chunks hit, retry on the SAME document with relaxed threshold
            if not retrieved_chunks and has_doc_filter:
                logger.info("[RAG TOOL] 0 chunks with document filter. Retrying scoped retrieval with relaxed threshold.")
                retrieved_chunks = await self.retriever.retrieve(
                    query,
                    filters=filters,
                    top_k=top_k,
                    similarity_threshold=0.15,
                )

            # Fallback 2: Global cross-user retry ONLY if no document filter was requested
            if not retrieved_chunks and not has_doc_filter and user_id is not None:
                logger.info("[RAG TOOL] 0 chunks for user_id=%s. Retrying globally.", user_id)
                unrestricted_filters = SearchFilters(user_id=None)
                retrieved_chunks = await self.retriever.retrieve(
                    query,
                    filters=unrestricted_filters,
                    top_k=top_k,
                    similarity_threshold=0.15,
                )

            # Stage 2: Centralized Context Building (Filter, Dedup, Sort, Cap)
            # IMPORTANT: After cross-encoder reranking, similarity_score holds LOGIT values
            # (range: -10 to +10 for ms-marco cross-encoders), NOT cosine similarity [0,1].
            # We must NOT apply a cosine threshold here — pass 0.0 to bypass threshold filtering.
            # The retriever already applied the cosine threshold before reranking.
            from app.rag.context_builder import ContextBuilder
            ctx_builder = ContextBuilder(similarity_threshold=0.0, max_chunks=8, max_context_chars=12000)
            ctx_res = ctx_builder.build_context(retrieved_chunks, query=query)

            relevant_chunks: list[RankedResult] = []
            if ctx_res.has_context:
                for sel in ctx_res.selected_chunks:
                    relevant_chunks.append(
                        RankedResult(
                            chunk_id=sel.chunk_id,
                            chunk_text=sel.chunk_text,
                            document_id=sel.document_id,
                            document_version_id=sel.document_version_id,
                            similarity_score=sel.similarity_score,
                            rank=sel.rank,
                            document_title=sel.document_title or "",
                            section_title=sel.section_title,
                            page_number=sel.page_number,
                        )
                    )


            # Stage 3: Evidence Extraction
            evidence_items = []
            for c in relevant_chunks:
                evidence_items.append({
                    "chunk_id": str(c.chunk_id),
                    "document_id": str(c.document_id),
                    "document_title": getattr(c, "document_title", "Uploaded Document"),
                    "section_title": getattr(c, "section_title", "Document Section"),
                    "content": c.chunk_text,
                    "relevance_score": c.similarity_score,
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
