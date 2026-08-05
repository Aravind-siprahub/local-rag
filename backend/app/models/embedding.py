"""`embeddings` table — maps to db/sql/003_tables.sql + 004_indexes.sql.

Dimension is fixed at 768 (nomic-embed-text via Ollama). See
`docs/DATABASE_DESIGN.md` "Future Extensibility" for how to add a second
embedding table if a different-dimension model is introduced later.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, ForeignKey, Index, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import VectorMetric, pg_enum

if TYPE_CHECKING:
    from app.models.document_chunk import DocumentChunk

EMBEDDING_DIM = 768


class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False
    )
    model_name: Mapped[str] = mapped_column(nullable=False)
    model_version: Mapped[str | None]
    dimensions: Mapped[int] = mapped_column(nullable=False)
    metric: Mapped[VectorMetric] = mapped_column(
        pg_enum(VectorMetric, name="vector_metric"), nullable=False, server_default=VectorMetric.COSINE.value
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=text("now()"), nullable=False)

    chunk: Mapped[DocumentChunk] = relationship(back_populates="embeddings")

    __table_args__ = (
        UniqueConstraint("chunk_id", "model_name", name="embeddings_chunk_model_unique"),
        CheckConstraint(f"dimensions = {EMBEDDING_DIM}", name="embeddings_dimensions_chk"),
        Index("embeddings_chunk_id_idx", "chunk_id"),
        Index("embeddings_model_name_idx", "model_name"),
        # HNSW ANN index for cosine distance. Alembic autogenerate does not
        # reliably diff `postgresql_with` HNSW options — always hand-review
        # any migration touching this index rather than trusting the diff.
        Index(
            "embeddings_embedding_hnsw_cosine_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
        ),
    )
