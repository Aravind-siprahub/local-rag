"""Vector similarity search over pgvector embeddings.

Retrieval-layer queries join embeddings -> chunks -> versions -> documents so
filters can scope results by user, document, or version without modifying
the existing `EmbeddingRepository`.
"""
from __future__ import annotations

import logging
import re
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
    document_ids: tuple[uuid.UUID, ...] | None = None
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
    distance: float
    document_title: str = ""
    section_title: str | None = None
    page_number: int | None = None
    metadata_: dict | None = None


async def search_fulltext(
    session: AsyncSession | None,
    query_text: str,
    *,
    top_k: int = 20,
    filters: SearchFilters | None = None,
) -> list[SearchHit]:
    """Full-text search using PostgreSQL tsvector over document_chunks content."""
    from sqlalchemy import func

    if session is None or not query_text or not query_text.strip():
        return []

    filters = filters or SearchFilters()

    # Primary full-text query using websearch_to_tsquery over concatenated title, section, and content
    clean_query = re.sub(r"['’]s\b", "", query_text.strip(), flags=re.IGNORECASE)
    full_text_expr = (
        func.coalesce(Document.title, '') + ' ' +
        func.coalesce(DocumentChunk.section_title, '') + ' ' +
        DocumentChunk.content
    )
    ts_vector = func.to_tsvector('english', full_text_expr)
    ts_query = func.websearch_to_tsquery('english', clean_query)
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
    if filters.document_ids:
        stmt = stmt.where(Document.id.in_(filters.document_ids))
    elif filters.document_id is not None:
        stmt = stmt.where(Document.id == filters.document_id)
    if filters.document_version_id is not None:
        stmt = stmt.where(DocumentVersion.id == filters.document_version_id)
    else:
        stmt = stmt.where((Document.current_version_id.is_(None)) | (DocumentChunk.document_version_id == Document.current_version_id))
    if filters.filename is not None:
        stmt = stmt.where(Document.title.ilike(f"%{filters.filename}%"))
    if filters.date_from is not None:
        stmt = stmt.where(Document.created_at >= filters.date_from)
    if filters.date_to is not None:
        stmt = stmt.where(Document.created_at <= filters.date_to)

    stmt = stmt.order_by(rank_expr.desc()).limit(top_k)

    try:
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
    except Exception as ft_exc:
        logger.warning("[FULLTEXT SEARCH PRIMARY FAILED] query=%r error=%s", query_text, ft_exc)
        hits = []

    logger.info("[FULLTEXT SEARCH] query=%r hits=%d filters=%s", query_text, len(hits), filters)

    # Fallback to OR-based websearch tsquery if primary query returns 0 hits
    if not hits:
        _STOP_WORDS_FT = {"tell", "about", "explain", "give", "show", "details", "info", "information", "what", "how", "with", "from", "this", "that", "there", "the", "and", "or", "for", "in", "to", "of", "a", "an", "is", "are"}
        words = [w for w in re.findall(r"\w+", query_text) if len(w) >= 3 and w.lower() not in _STOP_WORDS_FT]
        if words:
            try:
                or_query_str = " OR ".join(words)
                ts_query_or = func.websearch_to_tsquery('english', or_query_str)
                rank_expr_or = func.ts_rank(ts_vector, ts_query_or)
                stmt_or = (
                    select(
                        DocumentChunk.id.label("chunk_id"),
                        DocumentChunk.content,
                        DocumentVersion.document_id,
                        DocumentChunk.document_version_id,
                        Document.title,
                        DocumentChunk.section_title,
                        DocumentChunk.page_number,
                        DocumentChunk.metadata_,
                        rank_expr_or.label("rank_score"),
                    )
                    .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
                    .join(Document, DocumentVersion.document_id == Document.id)
                    .where(Document.deleted_at.is_(None))
                    .where(Document.status == DocumentStatus.READY)
                    .where(ts_vector.op("@@")(ts_query_or))
                )
                if filters.user_id is not None:
                    stmt_or = stmt_or.where(Document.user_id == filters.user_id)
                if filters.document_ids:
                    stmt_or = stmt_or.where(Document.id.in_(filters.document_ids))
                elif filters.document_id is not None:
                    stmt_or = stmt_or.where(Document.id == filters.document_id)
                if filters.document_version_id is not None:
                    stmt_or = stmt_or.where(DocumentVersion.id == filters.document_version_id)
                else:
                    stmt_or = stmt_or.where((Document.current_version_id.is_(None)) | (DocumentChunk.document_version_id == Document.current_version_id))

                stmt_or = stmt_or.order_by(rank_expr_or.desc()).limit(top_k)
                res_or = await session.execute(stmt_or)
                for row in res_or.all():
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
                logger.info("[FULLTEXT SEARCH OR-FALLBACK] query=%r hits=%d", query_text, len(hits))
            except Exception as ft_or_exc:
                logger.warning("[FULLTEXT SEARCH OR-FALLBACK FAILED] query=%r error=%s", query_text, ft_or_exc)

    if not hits and filters.user_id is not None:
        logger.info("[FULLTEXT QUERY FALLBACK] 0 hits for user_id=%s. Searching system-wide ready documents.", filters.user_id)
        stmt_fb = (
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
        if filters.document_ids:
            stmt_fb = stmt_fb.where(Document.id.in_(filters.document_ids))
        elif filters.document_id is not None:
            stmt_fb = stmt_fb.where(Document.id == filters.document_id)
        if filters.document_version_id is not None:
            stmt_fb = stmt_fb.where(DocumentVersion.id == filters.document_version_id)
        else:
            stmt_fb = stmt_fb.where((Document.current_version_id.is_(None)) | (DocumentChunk.document_version_id == Document.current_version_id))

        stmt_fb = stmt_fb.order_by(rank_expr.desc()).limit(top_k)
        try:
            fb_res = await session.execute(stmt_fb)
            for row in fb_res.all():
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
            logger.info("[FULLTEXT SEARCH SYSTEM-WIDE FALLBACK] query=%r hits=%d", query_text, len(hits))
        except Exception as fb_exc:
            logger.warning("[FULLTEXT SEARCH SYSTEM-WIDE FALLBACK FAILED] %s", fb_exc)

    return hits


async def search_similar(
    session: AsyncSession | None,
    query_embedding: list[float],
    *,
    model_name: str,
    top_k: int,
    filters: SearchFilters | None = None,
) -> list[SearchHit]:
    """Cosine-distance ANN search with optional ownership scoping.

    Returns hits ordered by ascending cosine distance (most similar first).
    """
    if query_embedding is not None and len(query_embedding) != EMBEDDING_DIM:
        raise ValueError(f"query_embedding must have {EMBEDDING_DIM} dimensions.")

    if session is None or not query_embedding:
        return []

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
            | (Embedding.model_name.is_(None))
        )
        .where(Document.deleted_at.is_(None))
        .where(Document.status == DocumentStatus.READY)
    )

    if filters.user_id is not None:
        stmt = stmt.where(Document.user_id == filters.user_id)
    if filters.document_ids:
        stmt = stmt.where(Document.id.in_(filters.document_ids))
    elif filters.document_id is not None:
        stmt = stmt.where(Document.id == filters.document_id)
    if filters.document_version_id is not None:
        stmt = stmt.where(DocumentVersion.id == filters.document_version_id)
    else:
        stmt = stmt.where((Document.current_version_id.is_(None)) | (DocumentChunk.document_version_id == Document.current_version_id))

    stmt = stmt.order_by(distance_expr).limit(top_k)

    logger.info(
        "[VECTOR SEARCH EXECUTE] model=%s dim=%d top_k=%d filters=%s",
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

    logger.info("[VECTOR SEARCH RESULT] model=%s hits=%d top_sim=%.4f", model_name, len(hits), (1 - hits[0].distance) if hits else 0.0)

    # Fallback to system-wide documents if user_id filter produced 0 hits
    if not hits and filters.user_id is not None:
        logger.info("[VECTOR QUERY FALLBACK] 0 hits for user_id=%s. Searching system-wide ready documents.", filters.user_id)
        fallback_stmt = (
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
                | (Embedding.model_name.is_(None))
            )
            .where(Document.deleted_at.is_(None))
            .where(Document.status == DocumentStatus.READY)
        )
        if filters.document_ids:
            fallback_stmt = fallback_stmt.where(Document.id.in_(filters.document_ids))
        elif filters.document_id is not None:
            fallback_stmt = fallback_stmt.where(Document.id == filters.document_id)
        if filters.document_version_id is not None:
            fallback_stmt = fallback_stmt.where(DocumentVersion.id == filters.document_version_id)
        else:
            fallback_stmt = fallback_stmt.where((Document.current_version_id.is_(None)) | (DocumentChunk.document_version_id == Document.current_version_id))

        fallback_stmt = fallback_stmt.order_by(distance_expr).limit(top_k)
        fb_res = await session.execute(fallback_stmt)
        for row in fb_res.all():
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
        logger.info("[VECTOR QUERY FALLBACK RESULT] hits_found=%d", len(hits))

    for idx, hit in enumerate(hits, 1):
        logger.info(
            "  Hit #%d: chunk_id=%s doc_id=%s distance=%.4f sim=%.4f preview=%r",
            idx, hit.chunk_id, hit.document_id, hit.distance, 1.0 - hit.distance, hit.chunk_text[:80]
        )

    return hits


async def search_document_chunks_structured(
    session: AsyncSession | None,
    *,
    filters: SearchFilters | None = None,
    max_chunks: int = 150,
) -> list[SearchHit]:
    """Retrieve all chunks for a target document ordered by chunk_index to perform section-aware sampling."""
    if session is None:
        return []

    filters = filters or SearchFilters()

    # Safety resolution: if document_id is not specified, resolve the latest ready document for the user
    if filters.document_id is None and not filters.document_ids and filters.user_id is not None:
        target_doc_stmt = (
            select(Document.id)
            .where(Document.user_id == filters.user_id)
            .where(Document.deleted_at.is_(None))
            .where(Document.status == DocumentStatus.READY)
            .order_by(Document.created_at.desc())
            .limit(1)
        )
        try:
            resolved_id = (await session.execute(target_doc_stmt)).scalar_one_or_none()
            if resolved_id:
                filters = SearchFilters(
                    user_id=filters.user_id,
                    document_id=resolved_id,
                    document_version_id=filters.document_version_id,
                    search_mode=filters.search_mode,
                )
                logger.info("[STRUCTURED SEARCH TARGET RESOLVED] Scoped search to document_id=%s", resolved_id)
        except Exception as res_err:
            logger.warning("[STRUCTURED SEARCH TARGET RESOLUTION ERROR] %s", res_err)

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
        )
        .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
        .join(Document, DocumentVersion.document_id == Document.id)
        .where(Document.deleted_at.is_(None))
        .where(Document.status == DocumentStatus.READY)
    )

    if filters.user_id is not None:
        stmt = stmt.where(Document.user_id == filters.user_id)
    if filters.document_ids:
        stmt = stmt.where(Document.id.in_(filters.document_ids))
    elif filters.document_id is not None:
        stmt = stmt.where(Document.id == filters.document_id)
    if filters.document_version_id is not None:
        stmt = stmt.where(DocumentVersion.id == filters.document_version_id)
    else:
        stmt = stmt.where(
            (Document.current_version_id.is_(None))
            | (DocumentChunk.document_version_id == Document.current_version_id)
        )

    stmt = stmt.order_by(DocumentChunk.chunk_index.asc()).limit(max_chunks)

    try:
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
                    distance=0.0,
                    section_title=row.section_title,
                    page_number=row.page_number,
                    metadata_=row.metadata_ if isinstance(row.metadata_, dict) else {},
                )
            )
        logger.info("[STRUCTURED SEARCH] document_id=%s hits=%d", filters.document_id, len(hits))
        return hits
    except Exception as exc:
        logger.error("[STRUCTURED SEARCH FAILED] error=%s", exc)
        return []



