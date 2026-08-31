"""Add conversation_summary and summary_updated_at columns to chat_sessions table.

Revision ID: 20260828_add_session_summary
Revises: 20260827_add_long_term_memory
Create Date: 2026-08-28

This migration adds session-level conversation summary fields to public.chat_sessions.
It is purely additive and fully reversible.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260828_add_session_summary"
down_revision = "20260827_add_long_term_memory"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chat_sessions", sa.Column("conversation_summary", sa.Text(), nullable=True))
    op.add_column("chat_sessions", sa.Column("summary_updated_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("chat_sessions", "summary_updated_at")
    op.drop_column("chat_sessions", "conversation_summary")
