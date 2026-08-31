"""Add long_term_memories table for persistent cross-session memory.

Revision ID: 20260827_add_long_term_memory
Revises: (latest existing migration)
Create Date: 2026-08-27

This migration adds the long_term_memories table to support the Chat Memory
subsystem. It is purely additive — no existing tables or columns are modified.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers, used by Alembic.
revision = "20260827_add_long_term_memory"
down_revision = "20260813_add_message_attachments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "long_term_memories",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("memory_type", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "importance",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0.5"),
        ),
        sa.Column(
            "confidence",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0.5"),
        ),
        sa.Column(
            "source_conversation_id",
            UUID(as_uuid=True),
            sa.ForeignKey("chat_sessions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "superseded_by",
            UUID(as_uuid=True),
            sa.ForeignKey("long_term_memories.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata_", JSONB(), nullable=True),
        sa.CheckConstraint("importance >= 0.0 AND importance <= 1.0", name="ltm_importance_range_chk"),
        sa.CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="ltm_confidence_range_chk"),
        sa.CheckConstraint("btrim(content) <> ''", name="ltm_content_not_blank_chk"),
    )

    op.create_index(
        "long_term_memories_user_id_active_idx",
        "long_term_memories",
        ["user_id"],
        postgresql_where=sa.text("is_active = true"),
    )
    op.create_index(
        "long_term_memories_user_type_idx",
        "long_term_memories",
        ["user_id", "memory_type"],
        postgresql_where=sa.text("is_active = true"),
    )
    op.create_index(
        "long_term_memories_source_conv_idx",
        "long_term_memories",
        ["source_conversation_id"],
        postgresql_where=sa.text("source_conversation_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("long_term_memories_source_conv_idx", table_name="long_term_memories")
    op.drop_index("long_term_memories_user_type_idx", table_name="long_term_memories")
    op.drop_index("long_term_memories_user_id_active_idx", table_name="long_term_memories")
    op.drop_table("long_term_memories")
