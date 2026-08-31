"""Debug and diagnostic endpoint for the RAG pipeline."""
from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db
from app.core.config import get_settings
from app.models.user import User
from app.rag.intent_router import classify
from app.rag.query_normalizer import normalize_query
from app.retrieval.retriever import Retriever
from app.retrieval.search import SearchFilters

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debug", tags=["Debug"])


class RAGDebugRequest(BaseModel):
    query: str = Field(..., description="Query to trace through the RAG pipeline.")
    document_id: uuid.UUID | None = Field(default=None, description="Optional document scoping filter.")
    top_k: int = Field(default=5, ge=1, le=20)


class RAGDebugResponse(BaseModel):
    query: str
    normalized_query: str
    intent_route: str
    search_filters: dict[str, Any]
    embedding_model: str
    embedding_dimension: int
    top_k: int
    retrieved_chunks: list[dict[str, Any]]
    final_context: str
    context_char_count: int


@router.post(
    "/rag",
    response_model=RAGDebugResponse,
    summary="Diagnostic trace of RAG query normalization, routing, vector search, and context construction",
)
async def debug_rag_query(
    payload: RAGDebugRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> RAGDebugResponse:
    query_text = payload.query.strip()
    if not query_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query text cannot be empty.")

    settings = get_settings()

    # 1. Normalization & Intent Routing
    _, norm_q, _ = normalize_query(query_text)
    route = classify(query_text, document_titles=["ActiveDocument.docx"] if payload.document_id else None)

    # 2. Filters
    filters = SearchFilters(
        user_id=current_user.id,
        document_id=payload.document_id,
    )

    # 3. Retrieval
    retriever = Retriever(session)
    results = await retriever.retrieve(
        query_text,
        filters=filters,
        top_k=payload.top_k,
        similarity_threshold=0.0,
    )

    # 4. Format Chunk Traces
    chunk_traces: list[dict[str, Any]] = []
    context_blocks: list[str] = []
    for r in results:
        chunk_traces.append({
            "chunk_id": str(r.chunk_id),
            "document_id": str(r.document_id),
            "document_title": r.document_title,
            "section_title": r.section_title,
            "page_number": r.page_number,
            "similarity_score": round(r.similarity_score, 4),
            "snippet": r.chunk_text[:180] + ("..." if len(r.chunk_text) > 180 else ""),
        })
        context_blocks.append(
            f"--- Document: {r.document_title} | Section: {r.section_title or 'General'} ---\n{r.chunk_text}"
        )

    final_context = "\n\n".join(context_blocks)

    return RAGDebugResponse(
        query=query_text,
        normalized_query=norm_q or query_text,
        intent_route=route.value,
        search_filters={
            "user_id": str(current_user.id),
            "document_id": str(payload.document_id) if payload.document_id else None,
        },
        embedding_model=settings.EMBEDDING_MODEL,
        embedding_dimension=768,
        top_k=payload.top_k,
        retrieved_chunks=chunk_traces,
        final_context=final_context,
        context_char_count=len(final_context),
    )
