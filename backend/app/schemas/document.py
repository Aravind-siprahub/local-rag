"""Schemas for `app.models.document.Document`."""
import uuid
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.enums import DocumentStatus
from app.schemas.common import ORMModel, PaginatedResponse, TimestampSchema
from app.schemas.openapi_examples import DOCUMENT_CREATE_OPENAPI_EXAMPLE


class DocumentBase(BaseModel):
    title: Annotated[str, Field(min_length=1, max_length=1000, examples=["My first document"])]
    description: Annotated[str | None, Field(default=None, examples=["Optional description"])]
    tags: list[str] = Field(default_factory=list, examples=[["demo"]])

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value: str) -> str:
        # Mirrors the DB's `documents_title_not_blank_chk` (btrim(title) <> '')
        # — Field(min_length=1) alone would still accept a whitespace-only string.
        if not value.strip():
            raise ValueError("title must not be blank.")
        return value


class DocumentCreate(DocumentBase):
    user_id: Annotated[
        uuid.UUID,
        Field(
            description="Owner user id. Call GET /users to list existing users, "
            "or POST /users to create one first.",
        ),
    ]

    model_config = ConfigDict(json_schema_extra={"example": DOCUMENT_CREATE_OPENAPI_EXAMPLE})


class DocumentUpdate(BaseModel):
    title: Annotated[str, Field(min_length=1, max_length=1000)] | None = None
    description: str | None = None
    tags: list[str] | None = None
    status: DocumentStatus | None = None
    current_version_id: uuid.UUID | None = None

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("title must not be blank.")
        return value


class DocumentResponse(DocumentBase, TimestampSchema, ORMModel):
    id: uuid.UUID
    user_id: uuid.UUID
    status: DocumentStatus
    current_version_id: uuid.UUID | None = None


DocumentListResponse = PaginatedResponse[DocumentResponse]
