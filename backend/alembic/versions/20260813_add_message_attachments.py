"""Add attachments column to chat_messages table.

Revision ID: 20260813_add_message_attachments
Revises: 20260806_add_storage_columns
Create Date: 2026-08-13
"""
from typing import Sequence, Union
from alembic import op

revision: str = '20260813_add_message_attachments'
down_revision: Union[str, None] = '20260806_add_storage_columns'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS attachments JSONB")


def downgrade() -> None:
    op.execute("ALTER TABLE chat_messages DROP COLUMN IF EXISTS attachments")
