"""Schemas for `app.models.citation.Citation`.

No `CitationUpdate`: a citation is an immutable record of what was retrieved
for a given message (no `updated_at` column) — if retrieval changes,
regenerate the message's citations, don't edit one in place.
"""
import uuid
from typing import Annotated, Any

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
    chunk_text: str | None = None
    document_id: uuid.UUID | None = None
    document_version_id: uuid.UUID | None = None
    document_title: str | None = None
    section_title: str | None = None
    page_number: int | None = None

    @classmethod
    def model_validate(cls, obj: Any, *, strict: bool | None = None, from_attributes: bool | None = None, context: dict[str, Any] | None = None) -> "CitationResponse":
        if isinstance(obj, dict):
            return super().model_validate(obj, strict=strict, from_attributes=from_attributes, context=context)
        
        # If it's an ORM object, extract the fields from the nested relations
        data = {
            "id": obj.id,
            "message_id": obj.message_id,
            "chunk_id": obj.chunk_id,
            "similarity_score": obj.similarity_score,
            "rank": obj.rank,
            "created_at": obj.created_at,
        }
        
        if hasattr(obj, "chunk") and obj.chunk:
            data["chunk_text"] = obj.chunk.content
            data["document_version_id"] = obj.chunk.document_version_id
            data["section_title"] = obj.chunk.section_title
            data["page_number"] = obj.chunk.page_number
            if hasattr(obj.chunk, "document_version") and obj.chunk.document_version:
                dv = obj.chunk.document_version
                data["document_id"] = dv.document_id
                if hasattr(dv, "document") and dv.document:
                    data["document_title"] = dv.document.title
                    
        return super().model_validate(data, strict=strict, from_attributes=from_attributes, context=context)

CitationListResponse = PaginatedResponse[CitationResponse]
