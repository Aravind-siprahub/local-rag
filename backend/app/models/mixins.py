"""Shared column mixins for ORM models."""
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.orm import Mapped, mapped_column


class TimestampMixin:
    """`created_at` / `updated_at`, matching every table created with
    `003_tables.sql`'s shared pattern.

    `updated_at` intentionally has no `onupdate=` here: the database trigger
    `set_updated_at()` (see `003_tables.sql`) already maintains it on every
    `UPDATE`, and it fires regardless of what value a client sends. Setting
    `onupdate` at the ORM level too would be redundant, not incorrect — but
    duplicating the trigger's responsibility in two places invites them
    drifting out of sync later.
    """

    created_at: Mapped[datetime] = mapped_column(server_default=text("now()"), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(server_default=text("now()"), nullable=False)
