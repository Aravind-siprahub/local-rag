"""`chat_messages` table — maps to db/sql/003_tables.sql."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Computed, ForeignKey, Index, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import MessageRole, pg_enum

if TYPE_CHECKING:
    from app.models.chat_session import ChatSession
    from app.models.citation import Citation


class ChatMessage(Base):
    """One row per conversation turn. Retrieval provenance lives in `citations`."""

    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[MessageRole] = mapped_column(pg_enum(MessageRole, name="message_role"), nullable=False)
    content: Mapped[str] = mapped_column(nullable=False)
    model_used: Mapped[str | None]
    prompt_tokens: Mapped[int | None]
    completion_tokens: Mapped[int | None]
    # Database-computed column (GENERATED ALWAYS AS ... STORED) — never set
    # this from Python; `Computed()` tells SQLAlchemy to exclude it from
    # INSERT/UPDATE statements entirely.
    total_tokens: Mapped[int | None] = mapped_column(
        Computed("COALESCE(prompt_tokens, 0) + COALESCE(completion_tokens, 0)", persisted=True)
    )
    latency_ms: Mapped[int | None]
    generation_time_ms: Mapped[int | None]
    error_message: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(server_default=text("now()"), nullable=False)

    session: Mapped[ChatSession] = relationship(back_populates="messages")
    citations: Mapped[list[Citation]] = relationship(
        back_populates="message", cascade="all, delete-orphan", order_by="Citation.rank"
    )

    __table_args__ = (
        CheckConstraint("btrim(content) <> ''", name="chat_messages_content_not_blank_chk"),
        Index("chat_messages_session_id_idx", "session_id"),
        Index("chat_messages_session_created_idx", "session_id", "created_at"),
    )
