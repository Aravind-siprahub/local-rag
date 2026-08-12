"""Vector similarity search over pgvector embeddings.

Retrieval-layer queries join embeddings → chunks → versions → documents so
filters can scope results by user, document, or version without modifying
the existing `EmbeddingRepository`.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_version import DocumentVersion
from app.models.embedding import EMBEDDING_DIM, Embedding
from app.models.enums import DocumentStatus


from datetime import datetime

@dataclass(frozen=True)
class SearchFilters:
    """Optional scoping and metadata filters applied before ranking."""

    user_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    document_version_id: uuid.UUID | None = None
    filename: str | None = None
    file_type: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    search_mode: str = "hybrid"  # "hybrid" | "semantic" | "fulltext"


@dataclass(frozen=True)
class SearchHit:
    """One raw nearest-neighbor or full-text match before threshold filtering."""

    chunk_id: uuid.UUID
    chunk_text: str
    document_id: uuid.UUID
    document_version_id: uuid.UUID
    document_title: str
    distance: float
    section_title: str | None = None
    page_number: int | None = None
    metadata_: dict | None = None


async def search_fulltext(
    session: AsyncSession,
    query_text: str,
    *,
    top_k: int = 20,
    filters: SearchFilters | None = None,
) -> list[SearchHit]:
    """Full-text search using PostgreSQL tsvector over document_chunks content."""
    from sqlalchemy import func

    filters = filters or SearchFilters()
    words = [w.strip() for w in query_text.split() if w.strip()]
    if not words:
        return []

    ts_query_str = " & ".join(words)
    ts_query = func.to_tsquery('english', ts_query_str)
    ts_vector = func.to_tsvector('english', DocumentChunk.content)
    rank_expr = func.ts_rank(ts_vector, ts_query)

    stmt = (
        select(
            DocumentChunk.id.label("chunk_id"),
            DocumentChunk.content,
            DocumentVersion.document_id,
            DocumentChunk.document_version_id,
            Document.title,
            DocumentChunk.section_title,
            DocumentChunk.page_number,
            DocumentChunk.metadata_,
            rank_expr.label("rank_score"),
        )
        .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        .where(Document.deleted_at.is_(None))
        .where(Document.status == DocumentStatus.READY)
        .where(ts_vector.op("@@")(ts_query))
    )

    if filters.user_id is not None:
        stmt = stmt.where(Document.user_id == filters.user_id)
    if filters.document_id is not None:
        stmt = stmt.where(Document.id == filters.document_id)
    if filters.filename is not None:
        stmt = stmt.where(Document.title.ilike(f"%{filters.filename}%"))
    if filters.date_from is not None:
        stmt = stmt.where(Document.created_at >= filters.date_from)
    if filters.date_to is not None:
        stmt = stmt.where(Document.created_at <= filters.date_to)

    stmt = stmt.order_by(rank_expr.desc()).limit(top_k)

    result = await session.execute(stmt)
    hits: list[SearchHit] = []
    for row in result.all():
        score = float(row.rank_score) if row.rank_score is not None else 0.001
        hits.append(
            SearchHit(
                chunk_id=row.chunk_id,
                chunk_text=row.content,
                document_id=row.document_id,
                document_version_id=row.document_version_id,
                document_title=row.title,
                distance=max(0.0, 1.0 - score),
                section_title=row.section_title,
                page_number=row.page_number,
                metadata_=row.metadata_ if isinstance(row.metadata_, dict) else {},
            )
        )
    return hits


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
            Document.title,
            DocumentChunk.section_title,
            DocumentChunk.page_number,
            DocumentChunk.metadata_,
            distance_expr.label("distance"),
        )
        .join(DocumentChunk, Embedding.chunk_id == DocumentChunk.id)
        .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        .where(
            (Embedding.model_name == model_name)
            | (Embedding.model_name.ilike(f"{model_name.split(':')[0]}%"))
        )
        .where(Document.deleted_at.is_(None))
        .where(Document.status == DocumentStatus.READY)
    )

    if filters.user_id is not None:
        stmt = stmt.where(Document.user_id == filters.user_id)
    if filters.document_id is not None:
        stmt = stmt.where(Document.id == filters.document_id)
    if filters.document_version_id is not None:
        stmt = stmt.where(DocumentVersion.id == filters.document_version_id)

    stmt = stmt.order_by(distance_expr).limit(top_k)

    logger.info(
        "[VECTOR QUERY EXECUTE] model=%s dim=%d top_k=%d filters=%s",
        model_name, len(query_embedding), top_k, filters
    )

    result = await session.execute(stmt)
    hits: list[SearchHit] = []
    for row in result.all():
        hits.append(
            SearchHit(
                chunk_id=row.chunk_id,
                chunk_text=row.content,
                document_id=row.document_id,
                document_version_id=row.document_version_id,
                document_title=row.title,
                distance=float(row.distance),
                section_title=row.section_title,
                page_number=row.page_number,
                metadata_=row.metadata_ if isinstance(row.metadata_, dict) else {},
            )
        )

    logger.info("[VECTOR QUERY RESULT] hits_found=%d", len(hits))
    for idx, hit in enumerate(hits, 1):
        logger.info(
            "  Hit #%d: chunk_id=%s doc_id=%s distance=%.4f sim=%.4f preview=%r",
            idx, hit.chunk_id, hit.document_id, hit.distance, 1.0 - hit.distance, hit.chunk_text[:80]
        )

    return hits

