"""`processing_jobs` table — maps to db/sql/003_tables.sql."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, ForeignKey, Index, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import ProcessingJobStatus, ProcessingJobType, pg_enum
from app.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from app.models.document_version import DocumentVersion


class ProcessingJob(TimestampMixin, Base):
    """One row per attempt of a pipeline stage (parse/chunk/embed/index)."""

    __tablename__ = "processing_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False
    )
    job_type: Mapped[ProcessingJobType] = mapped_column(
        pg_enum(ProcessingJobType, name="processing_job_type"), nullable=False
    )
    status: Mapped[ProcessingJobStatus] = mapped_column(
        pg_enum(ProcessingJobStatus, name="processing_job_status"),
        nullable=False,
        server_default=ProcessingJobStatus.PENDING.value,
    )
    error_message: Mapped[str | None]
    retry_count: Mapped[int] = mapped_column(nullable=False, server_default=text("0"))
    started_at: Mapped[datetime | None]
    completed_at: Mapped[datetime | None]

    document_version: Mapped[DocumentVersion] = relationship(back_populates="jobs")

    __table_args__ = (
        CheckConstraint("retry_count >= 0", name="processing_jobs_retry_nonneg_chk"),
        Index("processing_jobs_version_id_idx", "document_version_id"),
        Index(
            "processing_jobs_active_status_idx",
            "status",
            "created_at",
            postgresql_where=text("status IN ('pending', 'running')"),
        ),
    )
