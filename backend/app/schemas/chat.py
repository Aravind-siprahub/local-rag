"""Schemas for the unified chat / RAG API (`app/api/endpoints/chat.py`)."""
import uuid
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.swagger_constants import OPENAPI_PLACEHOLDER_UUID
from app.schemas.common import OptionalUUID
from app.schemas.openapi_examples import CHAT_REQUEST_OPENAPI_EXAMPLE


class ChatDocumentFilters(BaseModel):
    """Optional retrieval scoping for a chat question."""

    document_id: OptionalUUID = None
    document_version_id: OptionalUUID = None


class ChatAttachment(BaseModel):
    id: str | None = None
    filename: str
    mime_type: str
    size: int | None = None
    url: str | None = None
    storage_path: str | None = None
    document_id: OptionalUUID = None


class ChatRequest(BaseModel):
    """POST /chat body."""

    session_id: Annotated[
        OptionalUUID,
        Field(
            description=(
                "Chat session id from POST /chat-sessions. "
                "Omit (or leave blank) to auto-create a demo session for the "
                "first active user (convenient for Swagger Try-it-out)."
            ),
        ),
    ] = None
    question: Annotated[
        str,
        Field(
            min_length=1,
            examples=["What are the key findings in the uploaded report?"],
            json_schema_extra={
                "example": "What are the key findings in the uploaded report?",
            },
        ),
    ]
    document_id: OptionalUUID = None
    document_version_id: OptionalUUID = None
    top_k: Annotated[int, Field(ge=1, le=100)] | None = None
    similarity_threshold: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    attachments: list[ChatAttachment] | None = None
    provider: str | None = None
    model: str | None = None


    model_config = ConfigDict(json_schema_extra={"example": CHAT_REQUEST_OPENAPI_EXAMPLE})

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank.")
        return value.strip()

    @field_validator("document_id", "document_version_id", mode="after")
    @classmethod
    def drop_openapi_placeholder_filters(cls, value: uuid.UUID | None) -> uuid.UUID | None:
        # Swagger auto-fills optional UUID filters with the placeholder UUID,
        # which is not a real document — treat it as "no filter".
        if value == OPENAPI_PLACEHOLDER_UUID:
            return None
        return value


class ChatTokenUsageResponse(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ChatCitationResponse(BaseModel):
    chunk_id: uuid.UUID
    chunk_text: str
    document_id: uuid.UUID
    document_version_id: uuid.UUID | None = None
    document_title: str | None = None
    section_title: str | None = None
    page_number: int | None = None
    similarity_score: float
    rank: int
    url: str | None = None
    domain: str | None = None
    source_type: str = "local"


class ChatResponse(BaseModel):
    """POST /chat response."""

    answer: str
    citations: list[ChatCitationResponse]
    token_usage: ChatTokenUsageResponse | None = None
    model: str
    processing_time_ms: Annotated[int, Field(ge=0)]
    user_message_id: uuid.UUID
    assistant_message_id: uuid.UUID
    retrieval_mode: str | None = Field(default="local", description="Mode used: local, web, or hybrid")

