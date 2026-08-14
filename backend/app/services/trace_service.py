"""TraceStore / RAGTraceService layer for durable request trace persistence."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.rag_trace import RAGTrace
from app.repositories.rag_trace_repository import RAGTraceRepository

logger = logging.getLogger(__name__)


class TraceStore:
    """Service layer for persisting and retrieving RAG request traces safely."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.repository = RAGTraceRepository(session)

    async def save_trace_safely(
        self,
        *,
        request_id: str,
        user_id: uuid.UUID | None = None,
        session_id: uuid.UUID | None = None,
        original_query: str,
        normalized_query: str | None = None,
        detected_intent: str | None = None,
        selected_route: str | None = None,
        retrieval_start: datetime | None = None,
        retrieval_end: datetime | None = None,
        retrieval_duration_ms: int = 0,
        retrieved_chunk_ids: list[str] | None = None,
        retrieved_document_ids: list[str] | None = None,
        document_version_ids: list[str] | None = None,
        similarity_scores: list[float] | None = None,
        document_metadata: dict[str, Any] | None = None,
        embedding_duration_ms: int = 0,
        llm_duration_ms: int = 0,
        total_duration_ms: int = 0,
        token_usage: dict[str, Any] | None = None,
        fallback_info: str | None = None,
        error_type: str | None = None,
        status: str = "SUCCESS",
    ) -> RAGTrace | None:
        """Persist a RAG execution trace. Catches all exceptions to protect the user's RAG response."""
        try:
            trace = RAGTrace(
                request_id=request_id,
                user_id=user_id,
                session_id=session_id,
                original_query=original_query,
                normalized_query=normalized_query,
                detected_intent=detected_intent,
                selected_route=selected_route,
                retrieval_start=retrieval_start,
                retrieval_end=retrieval_end,
                retrieval_duration_ms=retrieval_duration_ms,
                retrieved_chunk_ids=retrieved_chunk_ids or [],
                retrieved_document_ids=retrieved_document_ids or [],
                document_version_ids=document_version_ids or [],
                similarity_scores=similarity_scores or [],
                document_metadata=document_metadata or {},
                embedding_duration_ms=embedding_duration_ms,
                llm_duration_ms=llm_duration_ms,
                total_duration_ms=total_duration_ms,
                token_usage=token_usage,
                fallback_info=fallback_info,
                error_type=error_type,
                status=status,
                created_at=datetime.now(timezone.utc),
            )
            self.repository.session.add(trace)
            await self.repository.session.flush()
            logger.info("[TRACE STORE] Persisted trace request_id=%s id=%s", request_id, trace.id)
            return trace
        except Exception as exc:
            logger.warning("[TRACE STORE FAILURE] Failed to save trace for request_id=%s: %s", request_id, exc)
            return None

    async def get_by_request_id(self, request_id: str) -> RAGTrace | None:
        try:
            return await self.repository.get_by_request_id(request_id)
        except Exception as exc:
            logger.warning("[TRACE STORE FAILURE] Failed to retrieve trace request_id=%s: %s", request_id, exc)
            return None

    async def get_by_request_id_for_user(self, request_id: str, user_id: uuid.UUID) -> RAGTrace | None:
        try:
            trace = await self.repository.get_by_request_id(request_id)
            if trace and str(trace.user_id) == str(user_id):
                return trace
            return None
        except Exception as exc:
            logger.warning("[TRACE STORE FAILURE] Failed to retrieve user trace request_id=%s: %s", request_id, exc)
            return None

    async def list_by_user(self, user_id: uuid.UUID, *, limit: int = 50) -> list[RAGTrace]:
        try:
            return await self.repository.list_by_user(user_id, limit=limit)
        except Exception as exc:
            logger.warning("[TRACE STORE FAILURE] Failed to list traces for user_id=%s: %s", user_id, exc)
            return []
