"""Schemas for `app.models.chat_message.ChatMessage`.

No `ChatMessageUpdate`: messages are append-only (the ORM model has no
`updated_at` column) — correcting a message means posting a new one, not
editing history.
"""
import uuid
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

from app.models.enums import MessageRole
from app.schemas.citation import CitationResponse
from app.schemas.common import CreatedAtSchema, ORMModel, PaginatedResponse


class ChatMessageBase(BaseModel):
    role: MessageRole
    content: Annotated[str, Field(min_length=1)]
    model_used: str | None = None
    prompt_tokens: Annotated[int, Field(ge=0)] | None = None
    completion_tokens: Annotated[int, Field(ge=0)] | None = None
    latency_ms: Annotated[int, Field(ge=0)] | None = None
    generation_time_ms: Annotated[int, Field(ge=0)] | None = None

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        # Mirrors the DB's `chat_messages_content_not_blank_chk`.
        if not value.strip():
            raise ValueError("content must not be blank.")
        return value


class ChatMessageCreate(ChatMessageBase):
    session_id: uuid.UUID


class ChatMessageResponse(ChatMessageBase, CreatedAtSchema, ORMModel):
    id: uuid.UUID
    session_id: uuid.UUID
    # Database-computed (GENERATED ALWAYS AS ... STORED) — read-only, never
    # accepted on Create.
    total_tokens: int | None = None
    error_message: str | None = None
    citations: list[CitationResponse] = Field(default_factory=list)


ChatMessageListResponse = PaginatedResponse[ChatMessageResponse]
