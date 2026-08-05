"""`document_chunks` table — maps to db/sql/003_tables.sql."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, ForeignKey, Index, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from app.models.document_version import DocumentVersion
    from app.models.embedding import Embedding


class DocumentChunk(TimestampMixin, Base):
    """Immutable text span produced by the chunking stage."""

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    content: Mapped[str] = mapped_column(nullable=False)
    content_tokens: Mapped[int | None]
    page_number: Mapped[int | None]
    section_title: Mapped[str | None]
    char_start: Mapped[int | None]
    char_end: Mapped[int | None]
    # "metadata" is reserved on Declarative classes (Base.metadata), so the
    # Python attribute is `metadata_` while the actual column name stays
    # exactly "metadata" — the first mapped_column() argument.
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, server_default=text("'{}'"))

    document_version: Mapped[DocumentVersion] = relationship(back_populates="chunks")
    embeddings: Mapped[list[Embedding]] = relationship(back_populates="chunk", cascade="all, delete-orphan")

    # created_at only — chunks are immutable once produced, no updated_at.
    updated_at = None  # type: ignore[assignment]

    __table_args__ = (
        UniqueConstraint("document_version_id", "chunk_index", name="document_chunks_index_unique"),
        CheckConstraint("btrim(content) <> ''", name="document_chunks_content_not_blank_chk"),
        CheckConstraint(
            "char_end IS NULL OR char_start IS NULL OR char_end >= char_start",
            name="document_chunks_span_chk",
        ),
        Index("document_chunks_version_id_idx", "document_version_id"),
        Index("document_chunks_version_order_idx", "document_version_id", "chunk_index"),
        Index("document_chunks_content_trgm_idx", "content", postgresql_using="gin", postgresql_ops={"content": "gin_trgm_ops"}),
        Index("document_chunks_metadata_gin_idx", "metadata", postgresql_using="gin"),
    )
