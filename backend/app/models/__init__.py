"""ORM models package — maps to the already-deployed schema in `db/sql/`.

Do not create tables from these models via `Base.metadata.create_all()` or
a fresh Alembic migration — the schema already exists in Supabase. Generate
the first Alembic revision with `alembic revision --autogenerate` for
review/documentation, then apply it with `alembic stamp head` — NOT
`alembic upgrade head` — so Alembic marks the database as current without
re-running DDL against tables that already exist.

Every model module is imported here so `Base.metadata` is fully populated
for Alembic autogenerate — `alembic/env.py` imports this package as a whole
rather than each module individually.
"""
from app.models.chat_message import ChatMessage
from app.models.chat_session import ChatSession
from app.models.citation import Citation
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_version import DocumentVersion
from app.models.embedding import Embedding
from app.models.enums import (
    DocumentStatus,
    DocumentVersionStatus,
    MessageRole,
    ProcessingJobStatus,
    ProcessingJobType,
    UserRole,
    VectorMetric,
)
from app.models.processing_job import ProcessingJob
from app.models.rag_trace import RAGTrace
from app.models.system_setting import SystemSetting
from app.models.user import User

__all__ = [
    # tables
    "User",
    "Document",
    "DocumentVersion",
    "DocumentChunk",
    "Embedding",
    "ChatSession",
    "ChatMessage",
    "Citation",
    "ProcessingJob",
    "RAGTrace",
    "SystemSetting",
    # enums
    "UserRole",
    "DocumentStatus",
    "DocumentVersionStatus",
    "ProcessingJobType",
    "ProcessingJobStatus",
    "MessageRole",
    "VectorMetric",
]
