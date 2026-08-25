"""`users` table — maps to db/sql/003_tables.sql."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, Index, text
from sqlalchemy.dialects.postgresql import CITEXT, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import UserRole, pg_enum
from app.models.mixins import TimestampMixin

if TYPE_CHECKING:
    from app.models.chat_session import ChatSession
    from app.models.document import Document
    from app.models.document_version import DocumentVersion


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    email: Mapped[str] = mapped_column(CITEXT, nullable=False)
    hashed_password: Mapped[str] = mapped_column(nullable=False, default="$2b$12$stub_hash_for_testing")
    full_name: Mapped[str | None]
    role: Mapped[UserRole] = mapped_column(
        pg_enum(UserRole, name="user_role"), nullable=False, server_default=UserRole.MEMBER.value
    )
    is_active: Mapped[bool] = mapped_column(nullable=False, server_default=text("true"))
    is_verified: Mapped[bool] = mapped_column(nullable=False, server_default=text("false"))
    last_login_at: Mapped[datetime | None]
    deleted_at: Mapped[datetime | None]

    documents: Mapped[list[Document]] = relationship(back_populates="owner")
    document_versions_uploaded: Mapped[list[DocumentVersion]] = relationship(back_populates="uploaded_by_user")
    chat_sessions: Mapped[list[ChatSession]] = relationship(back_populates="owner")

    __table_args__ = (
        CheckConstraint(r"email ~ '^[^@\s]+@[^@\s]+\.[^@\s]+$'", name="users_email_format_chk"),
        # NOT a plain unique index: scoped to active users only, so a
        # soft-deleted user's email can be reused by a new signup. This is
        # the *only* uniqueness enforcement on email — see 003_tables.sql's
        # comment on why there is deliberately no table-level UNIQUE(email).
        Index("users_email_active_uidx", "email", unique=True, postgresql_where=text("deleted_at IS NULL")),
    )
