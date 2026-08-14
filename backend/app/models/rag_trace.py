"""`rag_traces` table — maps to db/sql/008_rag_traces.sql."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import ForeignKey, Index, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class RAGTrace(Base):
    """Durable execution traces for RAG pipeline auditing, debugging, and evaluation."""

    __tablename__ = "rag_traces"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    request_id: Mapped[str] = mapped_column(nullable=False)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="SET NULL"), nullable=True
    )
    original_query: Mapped[str] = mapped_column(nullable=False)
    normalized_query: Mapped[str | None]
    detected_intent: Mapped[str | None]
    selected_route: Mapped[str | None]
    retrieval_start: Mapped[datetime | None]
    retrieval_end: Mapped[datetime | None]
    retrieval_duration_ms: Mapped[int] = mapped_column(default=0, nullable=False)
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"), nullable=False)
    retrieved_document_ids: Mapped[list[str]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"), nullable=False)
    document_version_ids: Mapped[list[str]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"), nullable=False)
    similarity_scores: Mapped[list[float]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"), nullable=False)
    document_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"), nullable=False)
    embedding_duration_ms: Mapped[int] = mapped_column(default=0, nullable=False)
    llm_duration_ms: Mapped[int] = mapped_column(default=0, nullable=False)
    total_duration_ms: Mapped[int] = mapped_column(default=0, nullable=False)
    token_usage: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    fallback_info: Mapped[str | None]
    error_type: Mapped[str | None]
    status: Mapped[str] = mapped_column(default="SUCCESS", nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=text("now()"), nullable=False)

    __table_args__ = (
        Index("ix_rag_traces_request_id", "request_id"),
        Index("ix_rag_traces_session_id", "session_id"),
        Index("ix_rag_traces_user_id", "user_id"),
        Index("ix_rag_traces_created_at", "created_at"),
    )
