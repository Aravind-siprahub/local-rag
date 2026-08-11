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
    RAG = "RAG"
    WEB = "WEB"
    CALCULATOR = "CALCULATOR"
    DIRECT = "DIRECT"
    DOCUMENT_LIST = "DOCUMENT_LIST"


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

# Document / knowledge-base cues (checked before WEB)
_DOC_EXTENSION = re.compile(
    r"(?i)\b[\w\-]+\.(docx?|pdf|txt|md|xlsx?|pptx?|csv)\b"
)
_DOC_PHRASES = (
    "according to my documents",
    "according to my document",
    "according to the document",
    "according to the documents",
    "according to my uploaded",
    "what does the document say",
    "what do the documents say",
    "what does my document say",
    "summarise my uploaded",
    "summarize my uploaded",
    "summarise my document",
    "summarize my document",
    "uploaded file",
    "uploaded files",
    "uploaded document",
    "uploaded documents",
    "knowledge base",
    "in my documents",
    "in the documents",
    "from my documents",
    "from the documents",
    "my documents",
    "my uploaded",
    "list documents",
    "list all documents",
    "show documents",
    "show all documents",
    "documents are available",
    "what documents",
    "which documents",
)

_DOC_LIST_KEYWORDS = (
    "list of document",
    "list of documents",
    "list my document",
    "list my documents",
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
)

# Current / external information cues
_WEB_KEYWORDS = (
    "today",
    "current",
    "latest",
    "recent",
    "weather",
    "exchange rate",
    "exchange rates",
    "stock price",
    "stock prices",
    "current price",
    "current prices",
    "news",
    "headline",
    "headlines",
    "what time is it",
    "current time",
    "current date",
    "today's date",
    "good friday",
    "public holiday",
    "public holidays",
    "when is ",
)


def _is_document_list(lower: str) -> bool:
    if any(w in lower for w in ["about", "inside", "content", "summary", "summarize", "summarise", "detail", "explain", "policy", "say"]):
        return False
    return any(kw in lower for kw in _DOC_LIST_KEYWORDS)


def classify(question: str) -> Route:
    """Return the route for ``question`` using lightweight deterministic rules and normalization."""
    text = (question or "").strip()
    if not text:
        return Route.DIRECT

    from app.rag.query_normalizer import normalize_query
    _, norm, _ = normalize_query(text)
    lower = norm.lower() if norm else text.lower()

    if _is_document_list(lower):
        route = Route.DOCUMENT_LIST
    elif _is_calculator(text, lower):
        route = Route.CALCULATOR
    elif _is_web(lower):
        route = Route.WEB
    else:
        route = Route.RAG

    logger.info('[AI ROUTER] question="%s" norm="%s" route=%s', text[:200], lower[:200], route.value)
    return route


def _is_calculator(text: str, lower: str) -> bool:
    if _CALC_PERCENT_OF.search(text):
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*percent\s+of\s+-?\d+", lower):
        return True
    # Binary arithmetic expression somewhere in the question (e.g. "10 + 5")
    if re.search(
        r"\b\d+(?:\.\d+)?\s*[\+\-\*\/]\s*-?\d+(?:\.\d+)?\b",
        text,
    ):
        return True
    if _CALC_EXPRESSION.match(text) and re.search(r"[\d]", text) and re.search(
        r"[\+\-\*\/\%]", text
    ):
        return True
    if _CALC_KEYWORDS.search(text):
        return True
    return False


def _is_rag(text: str, lower: str) -> bool:
    if _DOC_EXTENSION.search(text):
        return True
    if any(phrase in lower for phrase in _DOC_PHRASES):
        return True
    if "document say" in lower or "documents say" in lower:
        return True
    return False


def _is_web(lower: str) -> bool:
    # If the user explicitly asks about their documents, uploaded files, or specific document names, it is RAG not WEB
    if _is_rag(lower, lower):
        return False
    if any(kw in lower for kw in _WEB_KEYWORDS):
        return True
    if re.search(r"\b(price|prices)\b", lower) and re.search(
        r"\b(current|today|latest|now)\b", lower
    ):
        return True
    return False
