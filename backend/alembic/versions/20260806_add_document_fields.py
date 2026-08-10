"""Add missing columns to documents table for schema alignment.

Revision ID: 20260806_add_document_fields
Revises: 
Create Date: 2026-08-06
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '20260806_add_document_fields'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('documents', sa.Column('original_filename', sa.String(), nullable=True))
    op.add_column('documents', sa.Column('filename', sa.String(), nullable=True))
    op.add_column('documents', sa.Column('mime_type', sa.String(), nullable=True))
    op.add_column('documents', sa.Column('size', sa.BigInteger(), nullable=True))
    op.add_column('documents', sa.Column('file_size_bytes', sa.BigInteger(), nullable=True))


def downgrade() -> None:
    op.drop_column('documents', 'file_size_bytes')
    op.drop_column('documents', 'size')
    op.drop_column('documents', 'mime_type')
    op.drop_column('documents', 'filename')
    op.drop_column('documents', 'original_filename')
