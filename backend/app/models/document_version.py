"""`document_versions` table — maps to db/sql/003_tables.sql."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CHAR, BigInteger, CheckConstraint, ForeignKey, Index, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import DocumentVersionStatus, pg_enum
from app.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from app.models.document import Document
    from app.models.document_chunk import DocumentChunk
    from app.models.processing_job import ProcessingJob
    from app.models.user import User


class DocumentVersion(TimestampMixin, Base):
    """One immutable uploaded file per version; reprocessing lives here."""

    __tablename__ = "document_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    version_number: Mapped[int] = mapped_column(nullable=False)
    storage_key: Mapped[str] = mapped_column(nullable=False)
    original_filename: Mapped[str] = mapped_column(nullable=False)
    mime_type: Mapped[str] = mapped_column(nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    page_count: Mapped[int | None]
    status: Mapped[DocumentVersionStatus] = mapped_column(
        pg_enum(DocumentVersionStatus, name="document_version_status"),
        nullable=False,
        server_default=DocumentVersionStatus.UPLOADED.value,
    )
    error_message: Mapped[str | None]
    uploaded_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    parsed_at: Mapped[datetime | None]
    chunked_at: Mapped[datetime | None]
    embedded_at: Mapped[datetime | None]
    completed_at: Mapped[datetime | None]

    document: Mapped[Document] = relationship(back_populates="versions", foreign_keys=[document_id])
    uploaded_by_user: Mapped[User] = relationship(back_populates="document_versions_uploaded")
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document_version", cascade="all, delete-orphan"
    )
    jobs: Mapped[list[ProcessingJob]] = relationship(
        back_populates="document_version", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("document_id", "version_number", name="document_versions_doc_version_unique"),
        CheckConstraint("version_number > 0", name="document_versions_version_positive_chk"),
        CheckConstraint("file_size_bytes > 0", name="document_versions_size_positive_chk"),
        Index("document_versions_document_id_idx", "document_id"),
        Index("document_versions_status_idx", "status"),
        Index("document_versions_checksum_idx", "checksum_sha256"),
    )
