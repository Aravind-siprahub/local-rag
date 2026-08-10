"""Text normalization and embedding preparation for semantic chunks."""
from __future__ import annotations

import re

from app.services.metadata import Chunk

_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_TRAILING_WS_LINE_RE = re.compile(r"[ \t]+\n")


def normalize_text_for_embedding(text: str) -> str:
    """Normalize chunk text while preserving semantic structure.

    - Collapses horizontal whitespace runs.
    - Preserves paragraph breaks (double newlines).
    - Preserves markdown tables and fenced code blocks.
    - Does NOT lowercase or strip punctuation.
    """
    if not text:
        return ""

    # Protect fenced code blocks and markdown tables from aggressive normalization.
    protected: list[str] = []

    def _protect(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"\x00PROT{len(protected) - 1}\x00"

    # Fenced code blocks.
    working = re.sub(r"```[\s\S]*?```", _protect, text)
    # Markdown table blocks (consecutive lines starting with |).
    working = re.sub(r"(?:^\|.+\|\s*$\n?)+", _protect, working, flags=re.MULTILINE)

    working = working.replace("\r\n", "\n").replace("\r", "\n")
    working = _TRAILING_WS_LINE_RE.sub("\n", working)
    working = _MULTI_NEWLINE_RE.sub("\n\n", working)

    lines = working.split("\n")
    normalized_lines = [_MULTI_SPACE_RE.sub(" ", line) for line in lines]
    result = "\n".join(normalized_lines).strip()

    # Restore protected regions.
    for idx, original in enumerate(protected):
        result = result.replace(f"\x00PROT{idx}\x00", original)

    return result


def prepare_chunk_for_embedding(chunk: Chunk) -> str:
    """Build the final text payload sent to the embedding model."""
    parts: list[str] = []
    if chunk.breadcrumb:
        parts.append(f"[{chunk.breadcrumb}]")
    body = normalize_text_for_embedding(chunk.text)
    if body:
        parts.append(body)
    return "\n\n".join(parts) if parts else ""


def chunks_to_pgvector_records(chunks: list[Chunk]) -> list[dict]:
    """Convert chunks to pgvector-compatible records."""
    return [chunk.to_pgvector_record() for chunk in chunks]
