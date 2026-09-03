"""Add 2FA and recovery code fields to users table.

Revision ID: 20260901_add_totp_2fa
Revises: 20260828_add_session_summary
Create Date: 2026-09-01

Adds is_2fa_enabled, totp_secret_encrypted, and recovery_codes_hash columns to public.users table.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260901_add_totp_2fa"
down_revision = "20260828_add_session_summary"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("is_2fa_enabled", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column(
        "users",
        sa.Column("totp_secret_encrypted", sa.Text(), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column("recovery_codes_hash", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("users", "recovery_codes_hash")
    op.drop_column("users", "totp_secret_encrypted")
    op.drop_column("users", "is_2fa_enabled")
