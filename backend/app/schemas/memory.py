"""Pydantic schemas for the Memory API."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.memory.types import MemoryType


class MemoryCreate(BaseModel):
    """Payload to manually create a long-term memory."""

    memory_type: MemoryType = Field(
        default=MemoryType.PREFERENCE,
        description="Category of memory (preference, goal, technical_context, etc.).",
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The facts or preferences to remember.",
    )
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance score from 0.0 (low) to 1.0 (critical).",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata bag for extra context.",
    )


class MemoryUpdate(BaseModel):
    """Payload to update an existing long-term memory."""

    content: str | None = Field(
        default=None,
        min_length=1,
        max_length=1000,
        description="Updated memory content.",
    )
    importance: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Updated importance score.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Updated confidence score.",
    )
    is_active: bool | None = Field(
        default=None,
        description="Set to false to deactivate (soft-delete).",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Updated metadata bag.",
    )


class MemoryResponse(BaseModel):
    """Representation of a long-term memory returned by the API."""

    id: uuid.UUID
    user_id: uuid.UUID
    memory_type: str
    content: str
    importance: float
    confidence: float
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime | None = None
    source_conversation_id: uuid.UUID | None = None
    superseded_by: uuid.UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        from_attributes = True


class MemoryListResponse(BaseModel):
    """Paginated list of long-term memories."""

    items: list[MemoryResponse]
    total: int


class MemoryPurgeResponse(BaseModel):
    """Response returned when purging all user memories."""

    deleted_count: int
    message: str
