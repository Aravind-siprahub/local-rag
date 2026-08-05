"""Schemas for `app.models.embedding.Embedding`.

No `EmbeddingUpdate`: embeddings are insert-only — re-embedding with a new
model creates a new row (unique per `chunk_id` + `model_name`), it never
edits a vector in place.
"""
import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from app.models.enums import VectorMetric
from app.schemas.common import CreatedAtSchema, ORMModel, PaginatedResponse

EMBEDDING_DIM = 768


class EmbeddingBase(BaseModel):
    model_name: Annotated[str, Field(min_length=1, max_length=255)]
    model_version: str | None = None
    dimensions: Literal[768] = EMBEDDING_DIM
    metric: VectorMetric = VectorMetric.COSINE
    embedding: Annotated[list[float], Field(min_length=EMBEDDING_DIM, max_length=EMBEDDING_DIM)]


class EmbeddingCreate(EmbeddingBase):
    chunk_id: uuid.UUID


class EmbeddingResponse(EmbeddingBase, CreatedAtSchema, ORMModel):
    id: uuid.UUID
    chunk_id: uuid.UUID


EmbeddingListResponse = PaginatedResponse[EmbeddingResponse]
