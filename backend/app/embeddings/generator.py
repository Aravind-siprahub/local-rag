"""Generate and persist embeddings for document chunks."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings.client import EmbeddingClient, EmbeddingClientError
from app.models.document_chunk import DocumentChunk
from app.models.enums import VectorMetric
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingGenerationResult:
    """Summary of an embedding run for one document version."""

    embedded_count: int
    skipped_count: int
    total_chunks: int


class EmbeddingGenerator:
    """Generates one embedding per chunk and stores via `EmbeddingService`.

  Skips chunks that already have an embedding for the configured model.
  """

    def __init__(
        self,
        session: AsyncSession,
        client: EmbeddingClient,
        *,
        model_name: str | None = None,
        dimensions: int | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        from app.core.config import get_settings

        settings = get_settings()
        self.client = client
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.dimensions = dimensions if dimensions is not None else settings.EMBEDDING_DIMENSIONS
        self.embeddings = embedding_service or EmbeddingService(session)

    async def embed_chunks(self, chunks: list[DocumentChunk]) -> EmbeddingGenerationResult:
        """Embed each chunk once, skipping duplicates for this model."""
        embedded_count = 0
        skipped_count = 0

        for chunk in chunks:
            existing = await self.embeddings.repository.get_by_chunk_and_model(chunk.id, self.model_name)
            if existing is not None:
                skipped_count += 1
                logger.debug("Skipping chunk %s — embedding already exists for %s", chunk.id, self.model_name)
                continue

            try:
                vector = await self.client.embed(chunk.content)
            except EmbeddingClientError as exc:
                raise EmbeddingClientError(
                    f"Failed to embed chunk {chunk.id!r} (index={chunk.chunk_index}): {exc}"
                ) from exc

            await self.embeddings.create_embedding(
                chunk_id=chunk.id,
                model_name=self.model_name,
                embedding=vector,
                dimensions=self.dimensions,
                metric=VectorMetric.COSINE,
            )
            embedded_count += 1

        return EmbeddingGenerationResult(
            embedded_count=embedded_count,
            skipped_count=skipped_count,
            total_chunks=len(chunks),
        )

    async def embed_chunk_ids(self, chunk_ids: list[uuid.UUID]) -> EmbeddingGenerationResult:
        """Load chunks by id and embed them."""
        from app.repositories.document_chunk_repository import DocumentChunkRepository

        repository = DocumentChunkRepository(self.embeddings.session)
        chunks: list[DocumentChunk] = []
        for chunk_id in chunk_ids:
            chunk = await repository.get(chunk_id)
            if chunk is not None:
                chunks.append(chunk)

        return await self.embed_chunks(chunks)
