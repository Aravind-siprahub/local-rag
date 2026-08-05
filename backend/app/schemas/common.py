"""Shared building blocks for every domain schema module.

Nothing here is domain-specific — it exists purely so `user.py`,
`document.py`, etc. don't each redeclare the same ORM-serialization config
or timestamp fields.
"""
import uuid
from datetime import datetime
from typing import Annotated, Any, Generic, TypeVar

from pydantic import BaseModel, BeforeValidator, ConfigDict

T = TypeVar("T")


def _empty_str_to_none(value: Any) -> Any:
    """Swagger Try-it-out often sends ``\"\"`` for cleared optional UUID fields.

    Pydantic rejects that as an invalid UUID; treat blank strings as omitted.
    """
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


OptionalUUID = Annotated[
    uuid.UUID | None,
    BeforeValidator(_empty_str_to_none),
]


class ORMModel(BaseModel):
    """Base for every *Response* schema.

    `from_attributes=True` lets `SomeResponse.model_validate(orm_instance)`
    read straight off SQLAlchemy model attributes instead of requiring a
    dict — this is what makes `response_model=...` work directly against
    ORM objects returned from a (future) repository/service layer.
    """

    model_config = ConfigDict(from_attributes=True)


class CreatedAtSchema(BaseModel):
    """For tables with `created_at` but no `updated_at` (immutable rows):
    `document_chunks`, `embeddings`, `chat_messages`, `citations`.
    """

    created_at: datetime


class TimestampSchema(CreatedAtSchema):
    """For tables with both `created_at` and `updated_at`: `users`,
    `documents`, `document_versions`, `chat_sessions`, `processing_jobs`.
    """

    updated_at: datetime


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic list envelope reused as `PaginatedResponse[SomeResponse]` —
    every domain's `*ListResponse` is a type alias of this, not a
    hand-written duplicate.
    """

    items: list[T]
    total: int
    limit: int
    offset: int
