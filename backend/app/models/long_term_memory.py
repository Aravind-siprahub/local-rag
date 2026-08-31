"""`long_term_memories` table — persistent cross-session user memories.

Maps to the migration `20260827_add_long_term_memory.py`.

Design notes:
- `is_active = False` is a soft-delete / superseded marker; rows are never
  physically removed unless the user explicitly purges all memories.
- `superseded_by` links to the newer memory that replaced this one, enabling
  an audit trail of how preferences changed over time.
- `metadata_` is a free-form JSONB bag for per-memory-type extra data (e.g.
  the model name for a technical_context memory).
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, TYPE_CHECKING

from sqlalchemy import Boolean, CheckConstraint, Float, ForeignKey, Index, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.chat_session import ChatSession


class LongTermMemory(Base):
    """One row per extracted long-term memory for a user."""

    __tablename__ = "long_term_memories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    memory_type: Mapped[str] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(nullable=False)
    importance: Mapped[float] = mapped_column(
        Float(), nullable=False, server_default=text("0.5")
    )
    confidence: Mapped[float] = mapped_column(
        Float(), nullable=False, server_default=text("0.5")
    )
    source_conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )
    superseded_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("long_term_memories.id", ondelete="SET NULL"),
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean(), nullable=False, server_default=text("true")
    )
    created_at: Mapped[datetime] = mapped_column(
        server_default=text("now()"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        server_default=text("now()"), nullable=False
    )
    last_accessed_at: Mapped[datetime | None]
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Relationships
    owner: Mapped[User] = relationship("User", back_populates="long_term_memories")
    source_session: Mapped[ChatSession | None] = relationship(
        "ChatSession", foreign_keys=[source_conversation_id]
    )

    __table_args__ = (
        CheckConstraint(
            "importance >= 0.0 AND importance <= 1.0", name="ltm_importance_range_chk"
        ),
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0", name="ltm_confidence_range_chk"
        ),
        CheckConstraint("btrim(content) <> ''", name="ltm_content_not_blank_chk"),
        Index(
            "long_term_memories_user_id_active_idx",
            "user_id",
            postgresql_where=text("is_active = true"),
        ),
        Index(
            "long_term_memories_user_type_idx",
            "user_id",
            "memory_type",
            postgresql_where=text("is_active = true"),
        ),
        Index(
            "long_term_memories_source_conv_idx",
            "source_conversation_id",
            postgresql_where=text("source_conversation_id IS NOT NULL"),
        ),
    )
