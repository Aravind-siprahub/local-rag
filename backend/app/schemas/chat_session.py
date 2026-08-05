"""Schemas for `app.models.chat_session.ChatSession`."""
import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.common import ORMModel, PaginatedResponse, TimestampSchema
from app.schemas.openapi_examples import CHAT_SESSION_CREATE_OPENAPI_EXAMPLE


class ChatSessionBase(BaseModel):
    title: Annotated[
        str,
        Field(min_length=1, max_length=500, default="New chat", examples=["Research session"]),
    ]

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("title must not be blank.")
        return value


class ChatSessionCreate(ChatSessionBase):
    user_id: Annotated[
        uuid.UUID,
        Field(
            description=(
                "Owner user id. Call GET /users to list existing users, "
                "or POST /users to create one first. The returned session "
                "`id` is what POST /chat expects as `session_id`."
            ),
        ),
    ]

    model_config = ConfigDict(json_schema_extra={"example": CHAT_SESSION_CREATE_OPENAPI_EXAMPLE})


class ChatSessionUpdate(BaseModel):
    title: Annotated[str, Field(min_length=1, max_length=500)] | None = None
    is_archived: bool | None = None

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("title must not be blank.")
        return value


class ChatSessionResponse(ChatSessionBase, TimestampSchema, ORMModel):
    id: uuid.UUID
    user_id: uuid.UUID
    is_archived: bool
    last_message_at: datetime | None = None


ChatSessionListResponse = PaginatedResponse[ChatSessionResponse]
