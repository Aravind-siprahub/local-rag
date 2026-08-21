"""RAG orchestration response types."""
from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceCitation:
    """One source chunk cited in the generated answer."""

    chunk_id: uuid.UUID
    chunk_text: str
    document_id: uuid.UUID
    similarity_score: float
    rank: int
    document_version_id: uuid.UUID | None = None
    document_title: str | None = None
    section_title: str | None = None
    page_number: int | None = None


@dataclass(frozen=True)
class RAGTokenUsage:
    """Token usage from the LLM generation step."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class RAGResponse:
    """End-to-end RAG result returned to API or worker callers."""

    answer: str
    sources: list[SourceCitation]
    token_usage: RAGTokenUsage | None
    model: str
    processing_time_ms: int
    user_message_id: uuid.UUID
    assistant_message_id: uuid.UUID
