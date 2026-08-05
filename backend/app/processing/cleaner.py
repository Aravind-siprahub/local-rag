"""Normalize and clean extracted document text."""
import re

_WHITESPACE_RE = re.compile(r"[ \t]+")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


def clean_text(raw: str) -> str:
    """Normalize whitespace, remove empty lines, and preserve paragraph boundaries.

    Paragraphs are separated by blank lines in the source. Within each
    paragraph, runs of spaces/tabs collapse to a single space. Empty
    paragraphs are dropped. The result joins paragraphs with a double newline.
    """
    if not raw:
        return ""

    paragraphs = _PARAGRAPH_SPLIT_RE.split(raw.strip())
    cleaned_paragraphs: list[str] = []

    for paragraph in paragraphs:
        lines = paragraph.splitlines()
        normalized_lines = [_WHITESPACE_RE.sub(" ", line.strip()) for line in lines]
        normalized_lines = [line for line in normalized_lines if line]
        if normalized_lines:
            cleaned_paragraphs.append("\n".join(normalized_lines))

    return "\n\n".join(cleaned_paragraphs)
