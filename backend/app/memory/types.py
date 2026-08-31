"""Shared types for the Chat Memory subsystem.

Deliberately free of SQLAlchemy / Pydantic dependencies so they can be
imported in any layer without circular-import risk.
"""
from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


class MemoryType(str, enum.Enum):
    """Categories for long-term memories.

    Stored as plain strings in the DB so we can add new values without
    altering an enum type in PostgreSQL.
    """
    PREFERENCE = "preference"
    USER_PROFILE = "user_profile"
    PROJECT_CONTEXT = "project_context"
    REQUIREMENT = "requirement"
    DECISION = "decision"
    GOAL = "goal"
    TECHNICAL_CONTEXT = "technical_context"
    OTHER = "other"


@dataclass
class MemoryEntry:
    """In-memory representation of one long-term memory (not an ORM model).

    Used throughout the memory pipeline without touching SQLAlchemy session state.
    """
    id: uuid.UUID
    user_id: uuid.UUID
    memory_type: MemoryType
    content: str
    importance: float
    confidence: float
    created_at: datetime
    updated_at: datetime
    source_conversation_id: uuid.UUID | None = None
    last_accessed_at: datetime | None = None
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    # Set during retrieval — cosine similarity to the current query embedding
    similarity_score: float = 0.0

    @classmethod
    def from_orm(cls, orm_obj: Any) -> "MemoryEntry":
        """Construct from a `LongTermMemory` ORM instance."""
        return cls(
            id=orm_obj.id,
            user_id=orm_obj.user_id,
            memory_type=MemoryType(orm_obj.memory_type),
            content=orm_obj.content,
            importance=orm_obj.importance,
            confidence=orm_obj.confidence,
            created_at=orm_obj.created_at,
            updated_at=orm_obj.updated_at,
            source_conversation_id=orm_obj.source_conversation_id,
            last_accessed_at=orm_obj.last_accessed_at,
            is_active=orm_obj.is_active,
            metadata=orm_obj.metadata_ or {},
        )


@dataclass
class ExtractionCandidate:
    """One candidate memory proposed by the MemoryExtractor.

    Not yet persisted — the MemoryManager decides whether to create,
    update, or discard it after deduplication.
    """
    memory_type: MemoryType
    content: str
    importance: float
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    # Set by extractor when a conflict is detected
    conflicts_with: uuid.UUID | None = None
