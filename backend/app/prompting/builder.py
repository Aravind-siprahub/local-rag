"""Build LLM-ready prompts from retrieved document chunks."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.core.config import get_settings
from app.prompting.templates import format_chunk, format_user_prompt
from app.retrieval.ranking import RankedResult


class PromptBuilderError(Exception):
    """Raised when prompt input is invalid."""


@dataclass(frozen=True)
class RetrievedChunkContext:
    """Chunk metadata preserved for citations and downstream provenance."""

    chunk_id: uuid.UUID
    chunk_text: str
    document_id: uuid.UUID
    document_version_id: uuid.UUID
    similarity_score: float
    rank: int
    context_index: int


@dataclass(frozen=True)
class Prompt:
    """Structured prompt ready for an LLM call (no generation performed here)."""

    system_prompt: str
    user_prompt: str
    retrieved_chunks: list[RetrievedChunkContext]


class PromptBuilder:
    """Assemble system and user prompts from retrieval results.

    Independent of Ollama and any LLM provider — only formats text and
    enforces context size limits.
    """

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        max_context_chars: int | None = None,
    ) -> None:
        settings = get_settings()
        self.system_prompt = system_prompt if system_prompt is not None else settings.SYSTEM_PROMPT
        self.max_context_chars = max_context_chars if max_context_chars is not None else settings.MAX_CONTEXT_CHARS

    def build(self, question: str, retrieved_chunks: list[RankedResult]) -> Prompt:
        """Build a prompt from a user question and ranked retrieval results."""
        if not question or not question.strip():
            raise PromptBuilderError("Question must not be empty.")
        if self.max_context_chars <= 0:
            raise PromptBuilderError("max_context_chars must be greater than 0.")

        included_chunks, context_text = _build_context(retrieved_chunks, self.max_context_chars)
        user_prompt = format_user_prompt(context_text, question.strip())

        return Prompt(
            system_prompt=self.system_prompt.strip(),
            user_prompt=user_prompt,
            retrieved_chunks=included_chunks,
        )


def _build_context(
    retrieved_chunks: list[RankedResult],
    max_context_chars: int,
) -> tuple[list[RetrievedChunkContext], str]:
    """Select chunks that fit within the context budget and format them."""
    if not retrieved_chunks:
        return [], ""

    included: list[RetrievedChunkContext] = []
    formatted_parts: list[str] = []
    used_chars = 0

    for result in retrieved_chunks:
        context_index = len(included) + 1
        chunk_block = format_chunk(context_index, result.chunk_text)
        separator_len = 2 if formatted_parts else 0  # "\n\n" between blocks

        if used_chars + separator_len + len(chunk_block) <= max_context_chars:
            included.append(
                RetrievedChunkContext(
                    chunk_id=result.chunk_id,
                    chunk_text=result.chunk_text,
                    document_id=result.document_id,
                    document_version_id=result.document_version_id,
                    similarity_score=result.similarity_score,
                    rank=result.rank,
                    context_index=context_index,
                )
            )
            formatted_parts.append(chunk_block)
            used_chars += separator_len + len(chunk_block)
            continue

        remaining = max_context_chars - used_chars - separator_len
        if remaining <= len(f"[Chunk {context_index}]\n"):
            break

        truncated_text = _truncate_text(result.chunk_text, remaining - len(f"[Chunk {context_index}]\n"))
        if not truncated_text.strip():
            break

        truncated_block = format_chunk(context_index, truncated_text)
        included.append(
            RetrievedChunkContext(
                chunk_id=result.chunk_id,
                chunk_text=truncated_text,
                document_id=result.document_id,
                document_version_id=result.document_version_id,
                similarity_score=result.similarity_score,
                rank=result.rank,
                context_index=context_index,
            )
        )
        formatted_parts.append(truncated_block)
        break

    context_text = "\n\n".join(formatted_parts)
    return included, context_text


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."
