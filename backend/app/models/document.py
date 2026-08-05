"""`documents` table — maps to db/sql/003_tables.sql."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ARRAY, CheckConstraint, ForeignKey, Index, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import DocumentStatus, pg_enum
from app.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from app.models.document_version import DocumentVersion
    from app.models.user import User


class Document(TimestampMixin, Base):
    """Logical document owned by a user; wraps 1..N `document_versions`."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(nullable=False)
    description: Mapped[str | None]
    status: Mapped[DocumentStatus] = mapped_column(
        pg_enum(DocumentStatus, name="document_status"),
        nullable=False,
        server_default=DocumentStatus.UPLOADED.value,
    )
    # Circular reference with document_versions: this FK is added via
    # ALTER TABLE (use_alter=True) after document_versions exists, exactly
    # mirroring the deferred `ALTER TABLE ... ADD CONSTRAINT` in
    # 003_tables.sql. The explicit name must match that constraint's name.
    current_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "document_versions.id",
            ondelete="SET NULL",
            use_alter=True,
            name="documents_current_version_fk",
        ),
    )
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default=text("'{}'"))
    deleted_at: Mapped[datetime | None]

    owner: Mapped[User] = relationship(back_populates="documents")
    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="document",
        foreign_keys="DocumentVersion.document_id",
        cascade="all, delete-orphan",
    )
    current_version: Mapped[DocumentVersion | None] = relationship(
        foreign_keys=[current_version_id],
        post_update=True,  # breaks the INSERT-order cycle between documents <-> document_versions
    )

    __table_args__ = (
        CheckConstraint("btrim(title) <> ''", name="documents_title_not_blank_chk"),
        Index("documents_user_id_idx", "user_id", postgresql_where=text("deleted_at IS NULL")),
        Index("documents_user_status_idx", "user_id", "status", postgresql_where=text("deleted_at IS NULL")),
        Index("documents_tags_gin_idx", "tags", postgresql_using="gin"),
        Index("documents_title_trgm_idx", "title", postgresql_using="gin", postgresql_ops={"title": "gin_trgm_ops"}),
    )
