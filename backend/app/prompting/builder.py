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
    similarity_score: float
    rank: int
    context_index: int
    document_version_id: uuid.UUID | None = None
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
        self._custom_system_prompt = system_prompt is not None
        self.system_prompt = system_prompt if system_prompt is not None else settings.SYSTEM_PROMPT
        self.max_context_chars = max_context_chars if max_context_chars is not None else settings.MAX_CONTEXT_CHARS

    def build(
        self,
        question: str,
        retrieved_chunks: list[RankedResult],
        chat_history: list[dict[str, str]] | None = None,
        working_memory_summary: str | None = None,
        *,
        is_vision: bool = False,
        long_term_memory_context: str | None = None,
    ) -> Prompt:
        """Build a prompt from a user question, chat history, and ranked retrieval results.

        Args:
            long_term_memory_context: Pre-formatted memory section from MemoryContextBuilder.
                Injected into the user prompt as a clearly-labelled DATA block.
                Never modifies the system prompt.
        """
        if not question or not question.strip():
            raise PromptBuilderError("Question must not be empty.")
        if self.max_context_chars <= 0:
            raise PromptBuilderError("max_context_chars must be greater than 0.")

        included_chunks, context_text = _build_context(retrieved_chunks, self.max_context_chars, question=question.strip())
        user_prompt = format_user_prompt(
            context_text,
            question.strip(),
            chat_history=chat_history,
            working_memory_summary=working_memory_summary,
            long_term_memory_context=long_term_memory_context,
        )

        from datetime import datetime, timezone
        now_str = datetime.now(timezone.utc).strftime("%A, %B %d, %Y (%Y-%m-%d)")
        date_context = f"\n\nCurrent Temporal Context:\nToday's Date: {now_str}\nIf the user asks for today's date, current time, or day of the week, answer directly using this temporal context."

        system_prompt = self.system_prompt
        if is_vision and not self._custom_system_prompt:
            settings = get_settings()
            if included_chunks:
                system_prompt = settings.VISION_RAG_SYSTEM_PROMPT
            else:
                system_prompt = settings.VISION_SYSTEM_PROMPT
            return Prompt(
                system_prompt=system_prompt.strip(),
                user_prompt=user_prompt,
                retrieved_chunks=included_chunks,
            )

        if self._custom_system_prompt:
            full_system_prompt = system_prompt.strip()
        else:
            full_system_prompt = (system_prompt.strip() + date_context).strip()

        return Prompt(
            system_prompt=full_system_prompt,
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
    """Select chunks that fit within the context budget, apply passage trimming, and format them.

    NOTE: Chunks passed here are already cross-encoder reranked. Their similarity_score
    holds cross-encoder logit values (range -10 to +10), NOT cosine similarity [0,1].
    We bypass cosine threshold filtering (threshold=0.0) to prevent silent chunk drops.
    """
    if not retrieved_chunks:
        return [], ""

    from app.rag.context_builder import ContextBuilder

    # Use 0.0 threshold — cosine filtering was already applied by the retriever before reranking.
    # Using default settings.SIMILARITY_THRESHOLD here incorrectly filters by logit scores.
    builder = ContextBuilder(similarity_threshold=0.0, max_context_chars=max_context_chars)
    result = builder.build_context(retrieved_chunks, query=question)
    return result.selected_chunks, result.formatted_context


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."
