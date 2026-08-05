"""Vector similarity search over pgvector embeddings.

Retrieval-layer queries join embeddings → chunks → versions → documents so
filters can scope results by user, document, or version without modifying
the existing `EmbeddingRepository`.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_version import DocumentVersion
from app.models.embedding import EMBEDDING_DIM, Embedding


@dataclass(frozen=True)
class SearchFilters:
    """Optional scoping filters applied before vector ranking."""

    user_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    document_version_id: uuid.UUID | None = None


@dataclass(frozen=True)
class SearchHit:
    """One raw nearest-neighbor match before threshold filtering."""

    chunk_id: uuid.UUID
    chunk_text: str
    document_id: uuid.UUID
    document_version_id: uuid.UUID
    distance: float


async def search_similar(
    session: AsyncSession,
    query_embedding: list[float],
    *,
    model_name: str,
    top_k: int,
    filters: SearchFilters | None = None,
) -> list[SearchHit]:
    """Cosine-distance ANN search with optional ownership scoping.

    Returns hits ordered by ascending cosine distance (most similar first).
    """
    if len(query_embedding) != EMBEDDING_DIM:
        raise ValueError(f"query_embedding must have {EMBEDDING_DIM} dimensions.")

    filters = filters or SearchFilters()
    distance_expr = Embedding.embedding.cosine_distance(query_embedding)

    stmt = (
        select(
            Embedding.chunk_id,
            DocumentChunk.content,
            DocumentVersion.document_id,
            DocumentChunk.document_version_id,
            distance_expr.label("distance"),
        )
        .join(DocumentChunk, Embedding.chunk_id == DocumentChunk.id)
        .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        .where(Embedding.model_name == model_name)
        .where(Document.deleted_at.is_(None))
    )

    if filters.user_id is not None:
        stmt = stmt.where(Document.user_id == filters.user_id)
    if filters.document_id is not None:
        stmt = stmt.where(Document.id == filters.document_id)
    if filters.document_version_id is not None:
        stmt = stmt.where(DocumentVersion.id == filters.document_version_id)

    stmt = stmt.order_by(distance_expr).limit(top_k)

    result = await session.execute(stmt)
    hits: list[SearchHit] = []
    for row in result.all():
        hits.append(
            SearchHit(
                chunk_id=row.chunk_id,
                chunk_text=row.content,
                document_id=row.document_id,
                document_version_id=row.document_version_id,
                distance=float(row.distance),
            )
        )
    return hits
