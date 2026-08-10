"""Lightweight keyword extraction for semantic chunks (5–15 keywords)."""
from __future__ import annotations

import math
import re
from collections import Counter

# Common English stopwords — kept inline to avoid heavy NLP dependencies.
_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "would", "you",
    "your", "yours", "yourself", "yourselves",
})

_WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b")
_PHRASE_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9\-]*(?:\s+[a-zA-Z][a-zA-Z0-9\-]*){1,3}\b")
_CAMEL_SNAKE_RE = re.compile(r"[A-Z][a-z]+(?:[A-Z][a-z]+)+|[a-z]+_[a-z_]+")


def extract_keywords(
    text: str,
    *,
    min_keywords: int = 5,
    max_keywords: int = 15,
    language: str = "en",
) -> list[str]:
    """Extract 5–15 ranked keywords/phrases from chunk text.

    Uses frequency scoring with stopword filtering and bigram/trigram phrases.
    No external NLP models required.
    """
    if not text or not text.strip():
        return []

    normalized = text.strip()
    candidates: Counter[str] = Counter()

    # Single tokens (length >= 3, not stopwords).
    for match in _WORD_RE.finditer(normalized):
        word = match.group(0).lower()
        if len(word) >= 3 and word not in _STOPWORDS:
            candidates[word] += 1

    # Multi-word phrases (2–4 words).
    for match in _PHRASE_RE.finditer(normalized):
        phrase = match.group(0).strip()
        words = phrase.lower().split()
        if len(words) < 2:
            continue
        if all(w in _STOPWORDS for w in words):
            continue
        if words[0] in _STOPWORDS and words[-1] in _STOPWORDS:
            continue
        candidates[phrase.lower()] += 2  # Boost phrases.

    # Technical identifiers (camelCase, snake_case).
    for match in _CAMEL_SNAKE_RE.finditer(normalized):
        term = match.group(0)
        candidates[term.lower()] += 3

    if not candidates:
        return _fallback_keywords(normalized, max_keywords)

    # TF-style scoring with length bonus for multi-word terms.
    scored: list[tuple[str, float]] = []
    for term, count in candidates.items():
        tf = 1.0 + math.log1p(count)
        length_bonus = min(len(term.split()), 3) * 0.5
        scored.append((term, tf + length_bonus))

    scored.sort(key=lambda item: (-item[1], -len(item[0]), item[0]))
    keywords = [term for term, _ in scored[:max_keywords]]

    # Pad with single high-value tokens if below minimum.
    if len(keywords) < min_keywords:
        extras = _fallback_keywords(normalized, min_keywords)
        for extra in extras:
            if extra not in keywords:
                keywords.append(extra)
            if len(keywords) >= min_keywords:
                break

    return keywords[:max_keywords]


def _fallback_keywords(text: str, limit: int) -> list[str]:
    """Last-resort keyword list from longest non-stopword tokens."""
    tokens = [
        w.lower()
        for w in _WORD_RE.findall(text)
        if len(w) >= 4 and w.lower() not in _STOPWORDS
    ]
    seen: set[str] = set()
    result: list[str] = []
    for token in sorted(tokens, key=len, reverse=True):
        if token not in seen:
            seen.add(token)
            result.append(token)
        if len(result) >= limit:
            break
    return result
