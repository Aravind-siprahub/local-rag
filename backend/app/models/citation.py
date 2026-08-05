"""`citations` table — maps to db/sql/003_tables.sql."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Double, ForeignKey, Index, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.chat_message import ChatMessage
    from app.models.document_chunk import DocumentChunk


class Citation(Base):
    """Retrieval provenance: which chunks were shown to the model for a
    given assistant message, in retrieval-rank order.
    """

    __tablename__ = "citations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False
    )
    similarity_score: Mapped[float | None] = mapped_column(Double)
    rank: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=text("now()"), nullable=False)

    message: Mapped[ChatMessage] = relationship(back_populates="citations")
    chunk: Mapped[DocumentChunk] = relationship()

    __table_args__ = (
        UniqueConstraint("message_id", "chunk_id", name="citations_message_chunk_unique"),
        CheckConstraint("rank > 0", name="citations_rank_positive_chk"),
        Index("citations_message_id_idx", "message_id"),
        Index("citations_chunk_id_idx", "chunk_id"),
    )
