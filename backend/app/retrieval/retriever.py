"""End-to-end retrieval: embed question → search → rank."""
from __future__ import annotations

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.embeddings.client import EmbeddingClient, OllamaEmbeddingClient
from app.retrieval.ranking import RankedResult, rank_results
from app.retrieval.search import SearchFilters, search_similar

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Raised when retrieval input or configuration is invalid."""


class Retriever:
    """Retrieve relevant document chunks for a natural-language question.

    Embeds the question via the configured embedding client, runs pgvector
    cosine search, and returns ranked results. Independent of LLM chat
    generation and prompt building.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        client: EmbeddingClient | None = None,
        model_name: str | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self.session = session
        self.client = client or OllamaEmbeddingClient()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.top_k = top_k if top_k is not None else settings.TOP_K
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else settings.SIMILARITY_THRESHOLD
        )

    async def retrieve(
        self,
        question: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> list[RankedResult]:
        """Embed a user question, retrieve Top 20 candidates via hybrid search, and rerank to Top 5."""
        import time
        import asyncio
        from app.retrieval.ranking import rank_hybrid_rrf, rank_results, rerank_cross_encoder
        from app.retrieval.search import search_fulltext, search_similar

        if not question or not question.strip():
            raise RetrievalError("Question must not be empty.")

        settings = get_settings()
        candidate_top_k = top_k if top_k is not None else settings.TOP_K
        effective_threshold = (
            similarity_threshold if similarity_threshold is not None else self.similarity_threshold
        )
        final_top_k = getattr(settings, "FINAL_CONTEXT", 5)

        if candidate_top_k <= 0:
            raise RetrievalError("top_k must be greater than 0.")
        if not 0.0 <= effective_threshold <= 1.0:
            raise RetrievalError("similarity_threshold must be between 0.0 and 1.0.")

        start_time = time.monotonic()

        logger.info(
            "[RETRIEVER START] question=%r candidate_top_k=%d final_top_k=%d threshold=%.4f filters=%s",
            question.strip(),
            candidate_top_k,
            final_top_k,
            effective_threshold,
            filters,
        )

        filters_obj = filters or SearchFilters()
        mode = getattr(filters_obj, "search_mode", "hybrid")

        embed_start = time.monotonic()
        query_embedding = await self.client.embed(question.strip())
        embed_time_ms = int((time.monotonic() - embed_start) * 1000)

        retrieval_search_start = time.monotonic()

        if mode == "fulltext":
            ft_hits = await search_fulltext(self.session, question.strip(), top_k=candidate_top_k, filters=filters_obj)
            candidate_results = rank_results(ft_hits, 0.0)[:candidate_top_k]
            hits = ft_hits
        elif mode == "semantic":
            hits = await search_similar(
                self.session,
                query_embedding,
                model_name=self.model_name,
                top_k=candidate_top_k,
                filters=filters_obj,
            )
            candidate_results = rank_results(hits, effective_threshold)[:candidate_top_k]
        else:
            sem_task = search_similar(
                self.session,
                query_embedding,
                model_name=self.model_name,
                top_k=candidate_top_k,
                filters=filters_obj,
            )
            ft_task = search_fulltext(self.session, question.strip(), top_k=candidate_top_k, filters=filters_obj)
            sem_hits, ft_hits = await asyncio.gather(sem_task, ft_task, return_exceptions=True)

            clean_sem_hits: list[SearchHit] = sem_hits if isinstance(sem_hits, list) else []
            clean_ft_hits: list[SearchHit] = ft_hits if isinstance(ft_hits, list) else []

            hits = clean_sem_hits + clean_ft_hits
            candidate_results = rank_hybrid_rrf(clean_sem_hits, clean_ft_hits, similarity_threshold=effective_threshold)[:candidate_top_k]

        retrieval_time_ms = int((time.monotonic() - retrieval_search_start) * 1000)

        rerank_start = time.monotonic()
        results = rerank_cross_encoder(question.strip(), candidate_results, final_top_k=final_top_k)
        rerank_time_ms = int((time.monotonic() - rerank_start) * 1000)

        total_retrieval_ms = int((time.monotonic() - start_time) * 1000)

        logger.info(
            "[RETRIEVER METRICS] total_ms=%d embed_ms=%d search_ms=%d rerank_ms=%d candidates=%d selected=%d chunk_ids=%s",
            total_retrieval_ms,
            embed_time_ms,
            retrieval_time_ms,
            rerank_time_ms,
            len(candidate_results),
            len(results),
            [str(r.chunk_id) for r in results],
        )

        if not results:
            await self._log_empty_retrieval(
                filters=filters,
                hit_count=len(hits),
                top_k=candidate_top_k,
                threshold=effective_threshold,
            )

        return results



    async def _log_empty_retrieval(
        self,
        *,
        filters: SearchFilters | None,
        hit_count: int,
        top_k: int,
        threshold: float,
    ) -> None:
        """Explain zero results — common cause is user_id scoping past owned embeddings."""
        from sqlalchemy import func, select

        from app.models.document import Document
        from app.models.document_chunk import DocumentChunk
        from app.models.document_version import DocumentVersion
        from app.models.embedding import Embedding

        filters = filters or SearchFilters()
        scoped_count: int | None = None
        global_count: int | None = None

        try:
            stmt = (
                select(func.count())
                .select_from(Embedding)
                .join(DocumentChunk, Embedding.chunk_id == DocumentChunk.id)
                .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
                .join(Document, DocumentVersion.document_id == Document.id)
                .where(
                    (Embedding.model_name == self.model_name)
                    | (Embedding.model_name.ilike(f"{self.model_name.split(':')[0]}%"))
                )
                .where(Document.deleted_at.is_(None))
            )
            if filters.user_id is not None:
                stmt = stmt.where(Document.user_id == filters.user_id)
            if filters.document_id is not None:
                stmt = stmt.where(Document.id == filters.document_id)
            if filters.document_version_id is not None:
                stmt = stmt.where(DocumentVersion.id == filters.document_version_id)

            scoped_count = (await self.session.execute(stmt)).scalar_one()
            global_count = (
                await self.session.execute(
                    select(func.count())
                    .select_from(Embedding)
                    .where(
                        (Embedding.model_name == self.model_name)
                        | (Embedding.model_name.ilike(f"{self.model_name.split(':')[0]}%"))
                    )
                )
            ).scalar_one()
        except Exception as exc:
            logger.debug("Could not count embeddings for empty-retrieval diagnostics: %s", exc)

        logger.warning(
            "Retrieved 0 chunks (raw_hits=%d, top_k=%d, threshold=%.3f, filters=%s). "
            "Embeddings for model %r under these filters=%s; total embeddings for model=%s. "
            "If scoped count is 0 but global count > 0, the user_id/document filter is excluding all vectors.",
            hit_count,
            top_k,
            threshold,
            filters,
            self.model_name,
            scoped_count if scoped_count is not None else "unknown",
            global_count if global_count is not None else "unknown",
        )

    async def close(self) -> None:
        await self.client.close()
