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
    document_title: str | None = None
    section_title: str | None = None
    page_number: int | None = None


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

    def build(
        self,
        question: str,
        retrieved_chunks: list[RankedResult],
        chat_history: list[dict[str, str]] | None = None,
    ) -> Prompt:
        """Build a prompt from a user question, chat history, and ranked retrieval results."""
        if not question or not question.strip():
            raise PromptBuilderError("Question must not be empty.")
        if self.max_context_chars <= 0:
            raise PromptBuilderError("max_context_chars must be greater than 0.")

        included_chunks, context_text = _build_context(retrieved_chunks, self.max_context_chars, question=question.strip())
        user_prompt = format_user_prompt(context_text, question.strip(), chat_history=chat_history)

        return Prompt(
            system_prompt=self.system_prompt.strip(),
            user_prompt=user_prompt,
            retrieved_chunks=included_chunks,
        )


def _trim_passage_around_query(text: str, question: str, max_chars: int = 3500) -> str:
    """Trim oversized chunks to a passage window around matched query sentences."""
    text = text.strip()
    if len(text) <= max_chars:
        return text

    import re
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return text[:max_chars].rstrip() + "..."

    q_tokens = set(w.lower() for w in re.findall(r"\w+", question) if len(w) >= 3)
    matched_indices = []
    if q_tokens:
        for idx, sentence in enumerate(sentences):
            s_tokens = set(w.lower() for w in re.findall(r"\w+", sentence))
            if q_tokens & s_tokens:
                matched_indices.append(idx)

    if matched_indices:
        first_idx = matched_indices[0]
        last_idx = matched_indices[-1]
        start = max(0, first_idx - 2)
        end = min(len(sentences), last_idx + 3)
        trimmed = " ".join(sentences[start:end])
        if len(trimmed) <= max_chars:
            return f"... {trimmed} ..." if start > 0 or end < len(sentences) else trimmed

    return text[:max_chars].rstrip() + "..."


def _build_context(
    retrieved_chunks: list[RankedResult],
    max_context_chars: int,
    question: str = "",
) -> tuple[list[RetrievedChunkContext], str]:
    """Select chunks that fit within the context budget, apply passage trimming, and format them."""
    if not retrieved_chunks:
        return [], ""

    included: list[RetrievedChunkContext] = []
    formatted_parts: list[str] = []
    used_chars = 0

    for result in retrieved_chunks:
        context_index = len(included) + 1
        title = getattr(result, "document_title", "Unknown Document")
        section = getattr(result, "section_title", "General") or "General"
        page = str(getattr(result, "page_number", 1) or 1)
        chunk_id = str(result.chunk_id)

        # Apply simple truncation only — do NOT use passage trimming around keywords.
        # The trim function removes sentences that don't contain query words, which silently
        # strips answer content when it's in a different sentence from the matched keyword.
        effective_text = result.chunk_text if len(result.chunk_text) <= 3500 else result.chunk_text[:3500].rstrip() + "..."

        chunk_block = format_chunk(
            context_index,
            effective_text,
            title=title,
            section=section,
            page=page,
            chunk_id=chunk_id,
        )
        separator_len = 2 if formatted_parts else 0  # "\n\n" between blocks

        if used_chars + separator_len + len(chunk_block) <= max_context_chars:
            included.append(
                RetrievedChunkContext(
                    chunk_id=result.chunk_id,
                    chunk_text=effective_text,
                    document_id=result.document_id,
                    document_version_id=result.document_version_id,
                    similarity_score=result.similarity_score,
                    rank=result.rank,
                    context_index=context_index,
                    document_title=getattr(result, "document_title", None),
                    section_title=getattr(result, "section_title", None),
                    page_number=getattr(result, "page_number", None),
                )
            )
            formatted_parts.append(chunk_block)
            used_chars += separator_len + len(chunk_block)
            continue

        remaining = max_context_chars - used_chars - separator_len
        empty_block = format_chunk(
            context_index,
            "",
            title=title,
            section=section,
            page=page,
            chunk_id=chunk_id,
        )
        template_len = len(empty_block)
        is_single_chunk = len(retrieved_chunks) == 1

        if remaining <= template_len:
            if not is_single_chunk or used_chars > 0:
                break

        if is_single_chunk and remaining <= template_len:
            allowed_text_chars = max(10, remaining - 10)
        else:
            allowed_text_chars = remaining - template_len

        truncated_text = _truncate_text(effective_text, allowed_text_chars)
        if not truncated_text.strip():
            if not is_single_chunk:
                break
            truncated_text = "..."

        truncated_block = format_chunk(
            context_index,
            truncated_text,
            title=title,
            section=section,
            page=page,
            chunk_id=chunk_id,
        )

        included.append(
            RetrievedChunkContext(
                chunk_id=result.chunk_id,
                chunk_text=truncated_text,
                document_id=result.document_id,
                document_version_id=result.document_version_id,
                similarity_score=result.similarity_score,
                rank=result.rank,
                context_index=context_index,
                document_title=getattr(result, "document_title", None),
                section_title=getattr(result, "section_title", None),
                page_number=getattr(result, "page_number", None),
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
