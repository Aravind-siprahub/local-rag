"""`system_settings` table — maps to db/sql/003_tables.sql."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.user import User


class SystemSetting(Base):
    """Runtime-configurable settings (default chunk size, active embedding
    model, retrieval top_k, etc.) without a redeploy.
    """

    __tablename__ = "system_settings"

    key: Mapped[str] = mapped_column(primary_key=True)
    value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    description: Mapped[str | None]
    updated_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    updated_at: Mapped[datetime] = mapped_column(server_default=text("now()"), nullable=False)

    updated_by_user: Mapped[User | None] = relationship()
