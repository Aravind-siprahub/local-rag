"""SQLAlchemy 2.x ORM models for Local RAG.

Mirrors db/sql/003_tables.sql exactly. This file is the source Alembic
autogenerate diffs against; the raw SQL files are the source of truth for
manual review, but both must stay in sync.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import CITEXT, JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 768  # nomic-embed-text via Ollama


class Base(DeclarativeBase):
    pass


class UserRole(str, enum.Enum):
    admin = "admin"
    member = "member"


class DocumentStatus(str, enum.Enum):
    uploaded = "uploaded"
    processing = "processing"
    ready = "ready"
    failed = "failed"
    archived = "archived"


class DocumentVersionStatus(str, enum.Enum):
    uploaded = "uploaded"
    parsing = "parsing"
    parsed = "parsed"
    chunking = "chunking"
    chunked = "chunked"
    embedding = "embedding"
    embedded = "embedded"
    indexing = "indexing"
    completed = "completed"
    failed = "failed"


class ProcessingJobType(str, enum.Enum):
    upload = "upload"
    parse = "parse"
    chunk = "chunk"
    embed = "embed"
    index = "index"


class ProcessingJobStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class MessageRole(str, enum.Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class VectorMetric(str, enum.Enum):
    cosine = "cosine"
    l2 = "l2"
    inner_product = "inner_product"


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=text("now()"),
        onupdate=text("now()"),
        nullable=False,
    )


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(CITEXT, unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(Text, nullable=False)
    full_name: Mapped[str | None] = mapped_column(Text)
    role: Mapped[UserRole] = mapped_column(
        String, nullable=False, server_default=UserRole.member.value
    )
    is_active: Mapped[bool] = mapped_column(nullable=False, server_default=text("true"))
    is_verified: Mapped[bool] = mapped_column(nullable=False, server_default=text("false"))
    last_login_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))

    documents: Mapped[list["Document"]] = relationship(back_populates="owner")
    chat_sessions: Mapped[list["ChatSession"]] = relationship(back_populates="owner")


class Document(TimestampMixin, Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[DocumentStatus] = mapped_column(
        String, nullable=False, server_default=DocumentStatus.uploaded.value
    )
    current_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_versions.id", ondelete="SET NULL", use_alter=True),
    )
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default=text("'{}'"))
    deleted_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))

    owner: Mapped[User] = relationship(back_populates="documents")
    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document",
        foreign_keys="DocumentVersion.document_id",
        cascade="all, delete-orphan",
    )
    current_version: Mapped["DocumentVersion | None"] = relationship(
        foreign_keys=[current_version_id], post_update=True
    )

    __table_args__ = (CheckConstraint("btrim(title) <> ''", name="documents_title_not_blank_chk"),)


class DocumentVersion(TimestampMixin, Base):
    __tablename__ = "document_versions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    original_filename: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[str] = mapped_column(Text, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[DocumentVersionStatus] = mapped_column(
        String, nullable=False, server_default=DocumentVersionStatus.uploaded.value
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    uploaded_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    parsed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    chunked_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    embedded_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))

    document: Mapped[Document] = relationship(back_populates="versions", foreign_keys=[document_id])
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document_version", cascade="all, delete-orphan"
    )
    jobs: Mapped[list["ProcessingJob"]] = relationship(
        back_populates="document_version", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("document_id", "version_number", name="document_versions_doc_version_unique"),
        CheckConstraint("version_number > 0", name="document_versions_version_positive_chk"),
        CheckConstraint("file_size_bytes > 0", name="document_versions_size_positive_chk"),
    )


class DocumentChunk(TimestampMixin, Base):
    __tablename__ = "document_chunks"
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_tokens: Mapped[int | None] = mapped_column(Integer)
    page_number: Mapped[int | None] = mapped_column(Integer)
    section_title: Mapped[str | None] = mapped_column(Text)
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, server_default=text("'{}'"))

    document_version: Mapped[DocumentVersion] = relationship(back_populates="chunks")
    embeddings: Mapped[list["Embedding"]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("document_version_id", "chunk_index", name="document_chunks_index_unique"),
        CheckConstraint("btrim(content) <> ''", name="document_chunks_content_not_blank_chk"),
    )

    # created_at only (no updated_at) — chunks are immutable once produced
    updated_at = None  # type: ignore[assignment]


class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[uuid.UUID] = _uuid_pk()
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False
    )
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[str | None] = mapped_column(Text)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    metric: Mapped[VectorMetric] = mapped_column(
        String, nullable=False, server_default=VectorMetric.cosine.value
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False
    )

    chunk: Mapped[DocumentChunk] = relationship(back_populates="embeddings")

    __table_args__ = (
        UniqueConstraint("chunk_id", "model_name", name="embeddings_chunk_model_unique"),
        CheckConstraint(f"dimensions = {EMBEDDING_DIM}", name="embeddings_dimensions_chk"),
    )


class ChatSession(TimestampMixin, Base):
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(Text, nullable=False, server_default="New chat")
    is_archived: Mapped[bool] = mapped_column(nullable=False, server_default=text("false"))
    last_message_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))

    owner: Mapped[User] = relationship(back_populates="chat_sessions")
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[MessageRole] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model_used: Mapped[str | None] = mapped_column(Text)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    # total_tokens is a DB-generated column (see 003_tables.sql); read-only here.
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    generation_time_ms: Mapped[int | None] = mapped_column(Integer)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False
    )

    session: Mapped[ChatSession] = relationship(back_populates="messages")
    citations: Mapped[list["Citation"]] = relationship(
        back_populates="message", cascade="all, delete-orphan", order_by="Citation.rank"
    )

    __table_args__ = (CheckConstraint("btrim(content) <> ''", name="chat_messages_content_not_blank_chk"),)


class Citation(Base):
    __tablename__ = "citations"

    id: Mapped[uuid.UUID] = _uuid_pk()
    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False
    )
    similarity_score: Mapped[float | None] = mapped_column()
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False
    )

    message: Mapped[ChatMessage] = relationship(back_populates="citations")
    chunk: Mapped[DocumentChunk] = relationship()

    __table_args__ = (
        UniqueConstraint("message_id", "chunk_id", name="citations_message_chunk_unique"),
        CheckConstraint("rank > 0", name="citations_rank_positive_chk"),
    )


class ProcessingJob(TimestampMixin, Base):
    __tablename__ = "processing_jobs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False
    )
    job_type: Mapped[ProcessingJobType] = mapped_column(String, nullable=False)
    status: Mapped[ProcessingJobStatus] = mapped_column(
        String, nullable=False, server_default=ProcessingJobStatus.pending.value
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    started_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))

    document_version: Mapped[DocumentVersion] = relationship(back_populates="jobs")

    __table_args__ = (CheckConstraint("retry_count >= 0", name="processing_jobs_retry_nonneg_chk"),)


class SystemSetting(Base):
    __tablename__ = "system_settings"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    updated_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()"), onupdate=text("now()"), nullable=False
    )
