"""Schemas for `app.models.citation.Citation`.

No `CitationUpdate`: a citation is an immutable record of what was retrieved
for a given message (no `updated_at` column) — if retrieval changes,
regenerate the message's citations, don't edit one in place.
"""
import uuid
from typing import Annotated

from pydantic import BaseModel, Field

from app.schemas.common import CreatedAtSchema, ORMModel, PaginatedResponse


class CitationBase(BaseModel):
    similarity_score: Annotated[float, Field(ge=-1.0, le=1.0)] | None = None
    rank: Annotated[int, Field(gt=0)]


class CitationCreate(CitationBase):
    message_id: uuid.UUID
    chunk_id: uuid.UUID


class CitationResponse(CitationBase, CreatedAtSchema, ORMModel):
    id: uuid.UUID
    message_id: uuid.UUID
    chunk_id: uuid.UUID


CitationListResponse = PaginatedResponse[CitationResponse]
