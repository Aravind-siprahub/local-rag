"""Deterministic question routing for Agent Router v1.

Classification is rule-based (no LLM) so routing stays faster than the
pipelines it optimises.
"""
from __future__ import annotations

import logging
import re
from enum import Enum

logger = logging.getLogger(__name__)


class Route(str, Enum):
    DOCUMENT_QA = "DOCUMENT_QA"
    RAG = "DOCUMENT_QA"  # Alias for backward compatibility
    DOCUMENT_LIST = "DOCUMENT_LIST"
    DOCUMENT_METADATA = "DOCUMENT_METADATA"
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
    GENERIC_CHAT = "GENERIC_CHAT"
    CALCULATOR = "CALCULATOR"
    WEB = "WEB"
    DIRECT = "DIRECT"


# Greetings / Conversational patterns
_GREETING_PATTERNS = re.compile(
    r"^(hi+|hello+|hey+|good\s+(morning|afternoon|evening|day|night)|greetings|howdy|what'?s\s+up|sup)\b",
    re.IGNORECASE,
)
_CHAT_PHRASES = (
    "who are you",
    "what are you",
    "how are you",
    "how do you do",
    "nice to meet you",
    "thank you",
    "thanks",
    "bye",
    "goodbye",
)

# Arithmetic / percent-of patterns
_CALC_PERCENT_OF = re.compile(
    r"\b\d+(?:\.\d+)?\s*%\s*of\s*-?\d+(?:\.\d+)?\b",
    re.IGNORECASE,
)
_CALC_EXPRESSION = re.compile(
    r"(?i)^\s*(?:what\s+is|calculate|compute|evaluate)?\s*"
    r"[\d\.\s\+\-\*\/\(\)%]+\s*\??\s*$"
)
_CALC_KEYWORDS = re.compile(
    r"(?i)\b(calculate|compute|evaluate)\b.*[\d]"
)

# Document extension cue
_DOC_EXTENSION = re.compile(
    r"(?i)\b[\w\-]+\.(docx?|pdf|txt|md|xlsx?|pptx?|csv)\b"
)

# Document QA cues
_DOC_QA_PHRASES = (
    "according to my document",
    "according to my documents",
    "according to the document",
    "according to the documents",
    "according to my uploaded",
    "what does the document say",
    "what do the documents say",
    "what does my document say",
    "what does my uploaded document say",
    "summarise my uploaded",
    "summarize my uploaded",
    "summarise my document",
    "summarize my document",
    "uploaded file",
    "uploaded files",
    "uploaded document",
    "uploaded documents",
    "knowledge base",
    "in my document",
    "in my documents",
    "in the document",
    "in the documents",
    "from my document",
    "from my documents",
    "from the document",
    "from the documents",
    "my documents",
    "my uploaded",
    "leave policy",
    "leave policy what say",
    "policy in my document",
    "policy in document",
    "in my file",
    "in the file",
    "my doc say",
    "my document say",
    "what does my document",
    "what does the document",
)

# Document list cues
_DOC_LIST_KEYWORDS = (
    "list of document",
    "list of documents",
    "list my document",
    "list my documents",
    "list document",
    "list documents",
    "list out document",
    "list out documents",
    "what document",
    "what documents",
    "which document",
    "which documents",
    "show my file",
    "show my files",
    "show my document",
    "show my documents",
    "list uploaded",
    "show uploaded",
    "files uploaded",
    "documents uploaded",
    "documents do i have",
    "files do i have",
    "available documents",
    "available files",
    "documents are available",
    "files are available",
    "list files",
    "show files",
    "what doc u have",
    "what docs u have",
    "what file u have",
    "what doc i have",
    "what docs i have",
)

_DOC_LIST_REGEX = re.compile(
    r"\b(?:list|show|get|display|count)\s+(?:out\s+)?(?:all\s+)?(?:my\s+)?(?:uploaded\s+)?(?:documents?|files?|docs?)\b|"
    r"\bwhat\s+(?:documents?|files?|docs?)\s+(?:do\s+)?(?:u|you)\s+have\b|"
    r"\bwhich\s+(?:documents?|files?|docs?)\s+(?:do\s+)?(?:u|you)\s+have\b|"
    r"\bwhat\s+(?:documents?|files?|docs?)\s+(?:are\s+)?(?:available|uploaded)\b",
    re.IGNORECASE,
)

# Document metadata cues
_DOC_METADATA_KEYWORDS = (
    "when was",
    "when this file",
    "when document",
    "upload date",
    "date of upload",
    "when uploaded",
    "file size",
    "size of document",
    "who uploaded",
    "version of",
    "when was file",
    "when was document",
)

# Current / external information cues
_WEB_KEYWORDS = (
    "weather today",
    "exchange rate",
    "exchange rates",
    "stock price",
    "stock prices",
    "current price",
    "current prices",
    "latest news",
    "news today",
    "what time is it",
    "today's date",
    "public holiday",
    "public holidays",
)


def _is_generic_chat(lower: str) -> bool:
    if _GREETING_PATTERNS.match(lower):
        return True
    return any(lower == phrase or lower.startswith(phrase + " ") for phrase in _CHAT_PHRASES)


def _is_document_list(lower: str) -> bool:
    if any(w in lower for w in ["about", "inside", "content", "summary", "summarize", "summarise", "detail", "explain", "policy"]):
        return False
    if _DOC_LIST_REGEX.search(lower):
        return True
    return any(kw in lower for kw in _DOC_LIST_KEYWORDS)


def _is_document_metadata(lower: str) -> bool:
    return any(kw in lower for kw in _DOC_METADATA_KEYWORDS)


def _is_document_qa(text: str, lower: str) -> bool:
    if _DOC_EXTENSION.search(text) and not _is_document_metadata(lower):
        return True
    if any(phrase in lower for phrase in _DOC_QA_PHRASES):
        return True
    if ("document" in lower or "file" in lower or "policy" in lower or "doc" in lower) and any(kw in lower for kw in ["say", "state", "mention", "contain", "in", "according", "what", "how", "tell", "explain"]):
        return True
    return False


def _is_calculator(text: str, lower: str) -> bool:
    if _CALC_PERCENT_OF.search(text):
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*percent\s+of\s+-?\d+", lower):
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*[\+\-\*\/]\s*-?\d+(?:\.\d+)?\b", text):
        return True
    if _CALC_EXPRESSION.match(text) and re.search(r"[\d]", text) and re.search(r"[\+\-\*\/\%]", text):
        return True
    if _CALC_KEYWORDS.search(text):
        return True
    return False


def _is_web(lower: str) -> bool:
    if any(kw in lower for kw in _WEB_KEYWORDS):
        return True
    return False


def classify(question: str) -> Route:
    """Return the route for ``question`` using lightweight deterministic rules and normalization."""
    text = (question or "").strip()
    if not text:
        return Route.GENERIC_CHAT

    from app.rag.query_normalizer import normalize_query
    _, norm, _ = normalize_query(text)
    lower = norm.lower() if norm else text.lower()

    if _is_generic_chat(lower):
        route = Route.GENERIC_CHAT
    elif _is_document_list(lower):
        route = Route.DOCUMENT_LIST
    elif _is_document_metadata(lower):
        route = Route.DOCUMENT_METADATA
    elif _is_document_qa(text, lower):
        route = Route.DOCUMENT_QA
    elif _is_calculator(text, lower):
        route = Route.CALCULATOR
    elif _is_web(lower):
        route = Route.WEB
    else:
        route = Route.GENERAL_KNOWLEDGE

    logger.info('[AI ROUTER] question="%s" norm="%s" route=%s', text[:200], lower[:200], route.value)
    return route

