"""Generate and persist embeddings for document chunks."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings.client import EmbeddingClient, EmbeddingClientError
from app.models.document_chunk import DocumentChunk
from app.models.enums import VectorMetric
from app.services.embedding import prepare_chunk_for_embedding
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
                embed_text = self._build_embed_text(chunk)
                vector = await self.client.embed(embed_text)
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

    def _build_embed_text(self, chunk: DocumentChunk) -> str:
        """Build embedding input with breadcrumb context when metadata is present."""
        meta = getattr(chunk, "metadata_", None) or {}
        if not meta and not getattr(chunk, "section_title", None):
            return chunk.content

        from app.services.metadata import Chunk, ContentType

        semantic = Chunk(
            id=str(chunk.id),
            document_id=chunk.document_version_id,
            document_name=meta.get("document_name", ""),
            page_number=meta.get("page_number", 0) or 0,
            section=meta.get("section", ""),
            subsection=meta.get("subsection", ""),
            breadcrumb=meta.get("breadcrumb", "") or (chunk.section_title or ""),
            chunk_index=chunk.chunk_index,
            content_type=ContentType(meta.get("content_type", "paragraph")),
            language=meta.get("language", "en"),
            text=chunk.content,
        )
        prepared = prepare_chunk_for_embedding(semantic)
        return prepared or chunk.content
