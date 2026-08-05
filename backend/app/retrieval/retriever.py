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
        """Embed a user question and return ranked, threshold-filtered chunks."""
        if not question or not question.strip():
            raise RetrievalError("Question must not be empty.")

        effective_top_k = top_k if top_k is not None else self.top_k
        effective_threshold = (
            similarity_threshold if similarity_threshold is not None else self.similarity_threshold
        )

        if effective_top_k <= 0:
            raise RetrievalError("top_k must be greater than 0.")
        if not 0.0 <= effective_threshold <= 1.0:
            raise RetrievalError("similarity_threshold must be between 0.0 and 1.0.")

        query_embedding = await self.client.embed(question.strip())
        hits = await search_similar(
            self.session,
            query_embedding,
            model_name=self.model_name,
            top_k=effective_top_k,
            filters=filters,
        )

        results = rank_results(hits, effective_threshold)

        if not results:
            await self._log_empty_retrieval(
                filters=filters,
                hit_count=len(hits),
                top_k=effective_top_k,
                threshold=effective_threshold,
            )
        else:
            logger.info(
                "Retrieved %d chunks for question (top_k=%d, threshold=%.3f, filters=%s)",
                len(results),
                effective_top_k,
                effective_threshold,
                filters,
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
                .where(Embedding.model_name == self.model_name)
                .where(Document.deleted_at.is_(None))
            )
            if filters.user_id is not None:
                stmt = stmt.where(Document.user_id == filters.user_id)
            if filters.document_id is not None:
                stmt = stmt.where(Document.id == filters.document_id)
            if filters.document_version_id is not None:
                stmt = stmt.where(DocumentVersion.id == filters.document_version_id)

            scoped_count = int((await self.session.execute(stmt)).scalar_one())
            global_count = int(
                (
                    await self.session.execute(
                        select(func.count())
                        .select_from(Embedding)
                        .where(Embedding.model_name == self.model_name)
                    )
                ).scalar_one()
            )
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
