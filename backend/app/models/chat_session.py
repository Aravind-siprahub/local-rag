"""`chat_sessions` table — maps to db/sql/003_tables.sql."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from app.models.chat_message import ChatMessage
    from app.models.user import User


class ChatSession(TimestampMixin, Base):
    """A conversation thread belonging to exactly one user."""

    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(nullable=False, server_default="New chat")
    is_archived: Mapped[bool] = mapped_column(nullable=False, server_default=text("false"))
    last_message_at: Mapped[datetime | None]
    deleted_at: Mapped[datetime | None]

    owner: Mapped[User] = relationship(back_populates="chat_sessions")
    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )

    __table_args__ = (
        Index("chat_sessions_user_id_idx", "user_id", postgresql_where=text("deleted_at IS NULL")),
        Index(
            "chat_sessions_user_recent_idx",
            "user_id",
            text("last_message_at DESC"),
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )
