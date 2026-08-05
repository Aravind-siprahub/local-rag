"""Schemas for `app.models.document_chunk.DocumentChunk`.

No `DocumentChunkUpdate`: chunks are immutable once produced (the ORM model
itself has no `updated_at` column) — re-chunking creates new chunk rows
rather than editing existing ones. See `document_chunk.py`'s model docstring.
"""
import uuid
from typing import Annotated, Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.common import CreatedAtSchema, ORMModel, PaginatedResponse


class DocumentChunkBase(BaseModel):
    chunk_index: Annotated[int, Field(ge=0)]
    content: Annotated[str, Field(min_length=1)]
    content_tokens: Annotated[int, Field(ge=0)] | None = None
    page_number: Annotated[int, Field(ge=0)] | None = None
    section_title: str | None = None
    char_start: Annotated[int, Field(ge=0)] | None = None
    char_end: Annotated[int, Field(ge=0)] | None = None
    # The ORM attribute is `metadata_` (see app/models/document_chunk.py —
    # `metadata` is reserved by SQLAlchemy's Base). AliasChoices accepts
    # either the JSON client key "metadata" or the Python attribute name
    # "metadata_"; serialization_alias puts "metadata" back in JSON output.
    #
    # ORDER MATTERS AND IS SECURITY/CORRECTNESS-CRITICAL HERE: "metadata_"
    # must come first. Every SQLAlchemy mapped instance also has an
    # unrelated `metadata` attribute (the `MetaData` registry inherited from
    # `Base`), which always exists and is never "missing" — verified live:
    # with "metadata" listed first, `from_attributes=True` silently read
    # that `MetaData` object instead of the real column value and raised
    # `Input should be a valid dictionary [input_type=MetaData]`, rather
    # than cleanly erroring on a missing attribute. Listing "metadata_"
    # first means the real column is always found before that shadow
    # attribute is ever consulted.
    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("metadata_", "metadata"),
        serialization_alias="metadata",
    )

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        # Mirrors the DB's `document_chunks_content_not_blank_chk`.
        if not value.strip():
            raise ValueError("content must not be blank.")
        return value

    @model_validator(mode="after")
    def char_end_not_before_char_start(self) -> "DocumentChunkBase":
        # Mirrors the DB's `document_chunks_span_chk`.
        if self.char_start is not None and self.char_end is not None and self.char_end < self.char_start:
            raise ValueError("char_end must be >= char_start.")
        return self


class DocumentChunkCreate(DocumentChunkBase):
    document_version_id: uuid.UUID


class DocumentChunkResponse(DocumentChunkBase, CreatedAtSchema, ORMModel):
    id: uuid.UUID
    document_version_id: uuid.UUID


DocumentChunkListResponse = PaginatedResponse[DocumentChunkResponse]
