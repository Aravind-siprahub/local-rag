"""Add storage columns to documents and document_versions tables.

These columns were added to the SQLAlchemy models but were never migrated
to the database, causing 500 errors on all document API endpoints.

Uses `ADD COLUMN IF NOT EXISTS` so the migration is safe to run even when
some or all columns already exist.

Revision ID: 20260806_add_storage_columns
Revises: 20260806_add_document_fields
Create Date: 2026-08-06
"""
from typing import Sequence, Union

from alembic import op


revision: str = '20260806_add_storage_columns'
down_revision: Union[str, None] = '20260806_add_document_fields'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Use raw SQL with IF NOT EXISTS so this is idempotent even if some
    # columns were already created directly in Supabase.
    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS storage_provider VARCHAR DEFAULT 'supabase'")
    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS bucket_name VARCHAR DEFAULT 'documents'")
    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS storage_path VARCHAR")
    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS last_error TEXT")
    op.execute("ALTER TABLE document_versions ADD COLUMN IF NOT EXISTS storage_path VARCHAR")


def downgrade() -> None:
    op.execute("ALTER TABLE document_versions DROP COLUMN IF EXISTS storage_path")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS last_error")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS storage_path")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS bucket_name")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS storage_provider")
