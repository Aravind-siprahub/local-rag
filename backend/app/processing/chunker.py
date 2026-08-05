"""Split cleaned text into ordered chunks with character-span metadata."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    """One contiguous span of cleaned text with positional metadata."""

    content: str
    chunk_index: int
    char_start: int
    char_end: int


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[TextChunk]:
    """Split text into overlapping chunks while preserving document order.

    Args:
        text: Cleaned text to chunk.
        chunk_size: Maximum characters per chunk (must be > 0).
        overlap: Characters shared with the previous chunk (must be >= 0 and < chunk_size).

    Returns:
        Ordered list of non-empty chunks with char_start/char_end offsets
        into the original `text`.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size.")

    if not text or not text.strip():
        return []

    chunks: list[TextChunk] = []
    start = 0
    text_len = len(text)
    chunk_index = 0

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            end = _prefer_paragraph_break(text, start, end)

        segment = text[start:end].strip()
        if segment:
            # char_start/char_end reflect the span in the original text (trimmed bounds).
            leading = len(text[start:end]) - len(text[start:end].lstrip())
            trailing = len(text[start:end]) - len(text[start:end].rstrip())
            char_start = start + leading
            char_end = end - trailing
            chunks.append(
                TextChunk(
                    content=segment,
                    chunk_index=chunk_index,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            chunk_index += 1

        if end >= text_len:
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def _prefer_paragraph_break(text: str, start: int, target_end: int) -> int:
    """Prefer breaking at a paragraph or line boundary near target_end."""
    search_start = max(start + 1, target_end - 200)
    window = text[search_start:target_end]

    para_idx = window.rfind("\n\n")
    if para_idx != -1:
        return search_start + para_idx + 2

    line_idx = window.rfind("\n")
    if line_idx != -1:
        return search_start + line_idx + 1

    return target_end
