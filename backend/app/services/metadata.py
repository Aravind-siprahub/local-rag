"""Pydantic models for structured document parsing and semantic chunk metadata."""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ContentType(str, Enum):
    """Semantic content category for a chunk."""

    PARAGRAPH = "paragraph"
    TABLE = "table"
    CODE = "code"
    FAQ = "faq"
    LIST = "list"
    IMAGE_CAPTION = "image_caption"


class BlockType(str, Enum):
    """Structural block type extracted during parsing."""

    HEADING = "heading"
    SUBHEADING = "subheading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    CODE = "code"
    FAQ = "faq"
    IMAGE_CAPTION = "image_caption"


class DocumentBlock(BaseModel):
    """One structural unit within a parsed document."""

    block_type: BlockType
    text: str
    level: int = 0
    page_number: int | None = None
    language: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, value: str) -> str:
        return value.strip() if value else value


class HierarchyContext(BaseModel):
    """Current section hierarchy while walking document blocks."""

    section: str = ""
    subsection: str = ""
    breadcrumb_parts: list[str] = Field(default_factory=list)

    @property
    def breadcrumb(self) -> str:
        return " → ".join(self.breadcrumb_parts) if self.breadcrumb_parts else ""

    def with_heading(self, text: str, level: int) -> HierarchyContext:
        """Return a new context after applying a heading at the given level."""
        parts = list(self.breadcrumb_parts)
        if level <= 1:
            parts = [text]
            return HierarchyContext(section=text, subsection="", breadcrumb_parts=parts)
        if level == 2:
            section = parts[0] if parts else text
            return HierarchyContext(section=section, subsection=text, breadcrumb_parts=[section, text])
        # Deeper headings extend the breadcrumb trail.
        trimmed = parts[: level - 1]
        trimmed.append(text)
        section = trimmed[0] if trimmed else text
        subsection = trimmed[1] if len(trimmed) > 1 else ""
        return HierarchyContext(section=section, subsection=subsection, breadcrumb_parts=trimmed)


class ParsedDocument(BaseModel):
    """Structured representation of a parsed document with hierarchy."""

    document_id: uuid.UUID
    document_name: str
    language: str = "en"
    page_count: int = 0
    blocks: list[DocumentBlock] = Field(default_factory=list)
    source_format: str = ""
    parser_used: str = ""


class Chunk(BaseModel):
    """Semantic chunk ready for embedding and vector storage."""

    id: str
    document_id: uuid.UUID
    document_name: str
    page_number: int = 0
    section: str = ""
    subsection: str = ""
    breadcrumb: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    token_count: int = 0
    content_type: ContentType = ContentType.PARAGRAPH
    keywords: list[str] = Field(default_factory=list)
    language: str = "en"
    text: str = ""
    embedding: list[float] | None = None
    char_start: int | None = None
    char_end: int | None = None

    def to_metadata_dict(self) -> dict[str, Any]:
        """Serialize chunk fields for JSONB storage (excluding embedding vector)."""
        return {
            "id": self.id,
            "document_id": str(self.document_id),
            "document_name": self.document_name,
            "page_number": self.page_number,
            "section": self.section,
            "subsection": self.subsection,
            "breadcrumb": self.breadcrumb,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "token_count": self.token_count,
            "content_type": self.content_type.value,
            "keywords": self.keywords,
            "language": self.language,
        }

    def to_pgvector_record(self) -> dict[str, Any]:
        """Output compatible with pgvector ingestion."""
        return {
            "chunk_id": self.id,
            "document_id": str(self.document_id),
            "embedding": self.embedding,
            "metadata": self.to_metadata_dict(),
            "text": self.text,
        }


class ChunkingConfig(BaseModel):
    """Token-based semantic chunking parameters."""

    min_tokens: int = 400
    max_tokens: int = 700
    overlap_min: int = 50
    overlap_max: int = 100
    min_meaningful_chars: int = 50

    @field_validator("max_tokens")
    @classmethod
    def max_gte_min(cls, value: int, info: Any) -> int:
        min_tokens = info.data.get("min_tokens", 400)
        if value < min_tokens:
            raise ValueError("max_tokens must be >= min_tokens.")
        return value
