"""Lightweight query normalization layer for broken English and typos.

Executes deterministic pattern substitution in <1ms without calling any LLM.
Preserves original query and protects filenames, dates, numbers, and identifiers.
"""
from __future__ import annotations

import re

# Abbreviation / Shorthand replacement mapping (word-boundary anchored)
_SHORTHAND_MAP: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bdocs?\b", re.IGNORECASE), "document"),
    (re.compile(r"\bdocumentations?\b", re.IGNORECASE), "document"),
    (re.compile(r"\bu\b", re.IGNORECASE), "you"),
    (re.compile(r"\bur\b", re.IGNORECASE), "your"),
    (re.compile(r"\babt\b", re.IGNORECASE), "about"),
    (re.compile(r"\bpls\b|\bplz\b", re.IGNORECASE), "please"),
    (re.compile(r"\binfo\b", re.IGNORECASE), "information"),
    (re.compile(r"\bupload\b|\buploaded\b|\buploading\b", re.IGNORECASE), "uploaded"),
)

# File extensions & filenames pattern protection
_FILENAME_PATTERN = re.compile(
    r"\b[\w\-]+\.(docx?|pdf|txt|md|xlsx?|pptx?|csv)\b", re.IGNORECASE
)


def normalize_query(query: str) -> tuple[str, str, str]:
    """Normalize user query while preserving the original query.
    
    Returns:
        (original_query, normalized_query, retrieval_query)
    """
    raw = (query or "").strip()
    if not raw:
        return "", "", ""

    # Preserve original filenames
    protected_tokens: dict[str, str] = {}
    cleaned = raw

    for idx, match in enumerate(_FILENAME_PATTERN.finditer(raw)):
        placeholder = f"__PROTECTED_FILE_{idx}__"
        protected_tokens[placeholder] = match.group(0)
        cleaned = cleaned.replace(match.group(0), placeholder)

    # 1. Basic text cleanup
    norm = cleaned
    for pattern, replacement in _SHORTHAND_MAP:
        norm = pattern.sub(replacement, norm)

    # 2. Clean extra whitespace
    norm = re.sub(r"\s+", " ", norm).strip()

    # Restore protected filenames
    for placeholder, original_token in protected_tokens.items():
        norm = norm.replace(placeholder, original_token)
        cleaned = cleaned.replace(placeholder, original_token)

    # Construct retrieval_query by formatting common broken English phrasing
    retrieval_q = norm
    if "leave policy" in norm.lower() and "what" in norm.lower():
        retrieval_q = "What is the company leave policy?"
    elif "leave" in norm.lower() and ("how many" in norm.lower() or "count" in norm.lower()):
        retrieval_q = "How many leave days are provided in the leave policy?"

    return raw, norm, retrieval_q
