"""Centralized RAG context builder.

Filters, deduplicates, ranks, bounds, and formats retrieved document snippets
before they are injected into the LLM prompt. Prevents full-document exposure,
hallucinations, token overflow, and excessive latency.
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.core.config import get_settings
from app.prompting.builder import RetrievedChunkContext
from app.retrieval.ranking import RankedResult

logger = logging.getLogger(__name__)

# Standard fallback message when no relevant context is available
STANDARDIZED_UNANSWERABLE_MESSAGE = (
    "I couldn't find enough information in the available documents to answer this question."
)


@dataclass(frozen=True)
class ContextResult:
    """Outcome of centralized context building."""

    formatted_context: str
    selected_chunks: list[RetrievedChunkContext]
    total_retrieved: int
    total_filtered: int
    top_similarity_score: float
    selected_chunk_ids: list[str]
    total_chars: int
    has_context: bool


class ContextBuilder:
    """Centralized builder for bounding and formatting RAG context."""

    def __init__(
        self,
        *,
        similarity_threshold: float | None = None,
        max_chunks: int | None = None,
        max_context_chars: int | None = None,
        max_chunk_chars: int = 1200,
    ) -> None:
        settings = get_settings()
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else getattr(settings, "SIMILARITY_THRESHOLD", 0.30)
        )
        self.max_chunks = (
            max_chunks
            if max_chunks is not None
            else getattr(settings, "FINAL_CONTEXT", 5)
        )
        self.max_context_chars = (
            max_context_chars
            if max_context_chars is not None
            else min(getattr(settings, "MAX_CONTEXT_CHARS", 6000), 6000)
        )
        self.max_chunk_chars = max_chunk_chars

    def build_context(
        self,
        retrieved_chunks: Sequence[RankedResult | Any],
        query: str = "",
        *,
        custom_threshold: float | None = None,
        custom_max_chunks: int | None = None,
    ) -> ContextResult:
        """Filter, deduplicate, sort, cap, and format retrieved chunks into bounded context."""
        total_retrieved = len(retrieved_chunks)
        threshold = custom_threshold if custom_threshold is not None else self.similarity_threshold
        max_chunks_limit = custom_max_chunks if custom_max_chunks is not None else self.max_chunks

        if not retrieved_chunks:
            logger.info(
                "[CONTEXT BUILDER] query=%r retrieved=0 filtered=0 top_score=0.0000 selected_ids=[] chars=0 has_context=False",
                query[:80],
            )
            return ContextResult(
                formatted_context="",
                selected_chunks=[],
                total_retrieved=0,
                total_filtered=0,
                top_similarity_score=0.0,
                selected_chunk_ids=[],
                total_chars=0,
                has_context=False,
            )

        # 1. Filter out chunks below similarity threshold
        filtered_by_score: list[RankedResult] = []
        for r in retrieved_chunks:
            score = float(getattr(r, "similarity_score", 0.0))
            if score >= threshold:
                filtered_by_score.append(r)

        # 2. Deduplicate chunks by chunk_id and near-duplicate text content
        seen_ids: set[str] = set()
        seen_texts: list[str] = []
        deduplicated: list[RankedResult] = []

        for r in filtered_by_score:
            cid = str(getattr(r, "chunk_id", ""))
            if cid and cid in seen_ids:
                continue

            raw_text = getattr(r, "chunk_text", "") or ""
            norm_text = re.sub(r"\s+", " ", raw_text).strip().lower()
            if not norm_text:
                continue

            # Check for near-identical duplicate text (>85% token overlap or identical prefix)
            prefix = norm_text[:120]
            is_dup = False
            for existing in seen_texts:
                if prefix and prefix == existing[:120]:
                    is_dup = True
                    break
            if is_dup:
                continue

            if cid:
                seen_ids.add(cid)
            seen_texts.append(norm_text)
            deduplicated.append(r)

        # 3. Sort strictly by similarity score descending
        deduplicated.sort(
            key=lambda x: float(getattr(x, "similarity_score", 0.0)),
            reverse=True,
        )

        # 4. Cap max chunks
        capped_chunks = deduplicated[:max_chunks_limit]

        # 5. Format chunks preserving metadata and enforcing max_context_chars
        selected_contexts: list[RetrievedChunkContext] = []
        formatted_blocks: list[str] = []
        used_chars = 0
        top_score = float(getattr(capped_chunks[0], "similarity_score", 0.0)) if capped_chunks else 0.0

        for idx, item in enumerate(capped_chunks, start=1):
            raw_text = getattr(item, "chunk_text", "") or ""
            # Enforce per-chunk maximum characters
            trimmed_text = raw_text.strip()
            if len(trimmed_text) > self.max_chunk_chars:
                trimmed_text = trimmed_text[: self.max_chunk_chars].rstrip() + "..."

            title = getattr(item, "document_title", None) or "Document"
            page = getattr(item, "page_number", None)
            section = getattr(item, "section_title", None)
            cid = getattr(item, "chunk_id", None)
            doc_id = getattr(item, "document_id", None) or uuid.uuid4()
            score = float(getattr(item, "similarity_score", 0.0))

            header_parts = [f"Document: {title}"]
            if page is not None and page > 0:
                header_parts.append(f"Page: {page}")
            if section and section.strip():
                header_parts.append(f"Section: {section.strip()}")
            header = " | ".join(header_parts)

            chunk_block = f"[Chunk {idx}] ({header})\n{trimmed_text}"
            block_len = len(chunk_block) + (2 if formatted_blocks else 0)

            if used_chars + block_len <= self.max_context_chars:
                formatted_blocks.append(chunk_block)
                used_chars += block_len
                selected_contexts.append(
                    RetrievedChunkContext(
                        chunk_id=cid if isinstance(cid, uuid.UUID) else uuid.UUID(str(cid)),
                        chunk_text=trimmed_text,
                        document_id=doc_id if isinstance(doc_id, uuid.UUID) else uuid.UUID(str(doc_id)),
                        document_version_id=getattr(item, "document_version_id", None),
                        similarity_score=score,
                        rank=idx,
                        context_index=idx,
                        document_title=title,
                        section_title=section,
                        page_number=page,
                    )
                )
            elif len(capped_chunks) == 1 and not selected_contexts:
                # Single chunk with tight max_context_chars limit: truncate text
                avail_chars = max(4, self.max_context_chars - 10)
                truncated_text = trimmed_text[: avail_chars - 3].rstrip() + "..."
                single_block = f"[Chunk {idx}] ({header})\n{truncated_text}"
                formatted_blocks.append(single_block)
                selected_contexts.append(
                    RetrievedChunkContext(
                        chunk_id=cid if isinstance(cid, uuid.UUID) else uuid.UUID(str(cid)),
                        chunk_text=truncated_text,
                        document_id=doc_id if isinstance(doc_id, uuid.UUID) else uuid.UUID(str(doc_id)),
                        document_version_id=getattr(item, "document_version_id", None),
                        similarity_score=score,
                        rank=idx,
                        context_index=idx,
                        document_title=title,
                        section_title=section,
                        page_number=page,
                    )
                )
                break
            else:
                # If budget exceeded, stop adding chunks
                break

        formatted_context = "\n\n".join(formatted_blocks)
        selected_ids = [str(c.chunk_id) for c in selected_contexts]
        has_context = len(selected_contexts) > 0

        logger.info(
            "[CONTEXT BUILDER] query=%r retrieved=%d filtered=%d top_score=%.4f selected_ids=%s chars=%d has_context=%s",
            query[:80],
            total_retrieved,
            len(selected_contexts),
            top_score,
            selected_ids,
            len(formatted_context),
            has_context,
        )

        return ContextResult(
            formatted_context=formatted_context,
            selected_chunks=selected_contexts,
            total_retrieved=total_retrieved,
            total_filtered=len(selected_contexts),
            top_similarity_score=top_score,
            selected_chunk_ids=selected_ids,
            total_chars=len(formatted_context),
            has_context=has_context,
        )
