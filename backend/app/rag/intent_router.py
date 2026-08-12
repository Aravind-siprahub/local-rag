"""Deterministic question routing for Agent Router v1.

Classification is rule-based (no LLM) so routing stays faster than the
pipelines it optimises.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
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
    "deployment guide",
    "in deployment guide",
)

_DOC_QA_CUE_WORDS = (
    "document", "documents", "file", "files", "policy", "policies",
    "doc", "docs", "guide", "guides", "manual", "manuals",
    "handbook", "handbooks", "sheet", "sheets", "prd", "specification",
)

_DOC_QA_ACTION_WORDS = (
    "say", "state", "mention", "contain", "in", "according",
    "what", "how", "tell", "explain", "summarize", "summarise", "describe",
    "show", "find", "search", "details", "about", "abt", "ssl", "nginx", "setup",
)

# Project / corpus-aware document questions (tech stack, architecture, etc.)
_PROJECT_INFO_CUES = (
    "tech stack",
    "technology stack",
    "techstack",
    "frontend",
    "backend",
    "front end",
    "back end",
    "architecture",
    "were using",
    "we're using",
    "we using",
    "what using",
    "what we use",
    "technologies",
    "technology",
    "database",
    "framework",
    "built with",
    "built on",
    "stack",
)

_GENERIC_DEFINITION = re.compile(
    r"^\s*what\s+is\s+[\w\-]+[?]?\s*$",
    re.IGNORECASE,
)

_TITLE_NOISE = {
    "prd",
    "guide",
    "summary",
    "staging",
    "deployment",
    "document",
    "doc",
    "docs",
    "final",
    "draft",
    "v1",
    "v2",
    "v3",
    "v4",
}

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


def _is_generic_chat(lower: str) -> bool:
    if _GREETING_PATTERNS.match(lower):
        return True
    return any(lower == phrase or lower.startswith(phrase + " ") for phrase in _CHAT_PHRASES)


def _is_document_list(lower: str) -> bool:
    if any(w in lower for w in ["about", "inside", "content", "summary", "summarize", "summarise", "detail", "explain", "policy", "tell"]):
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
    if any(noun in lower for noun in _DOC_QA_CUE_WORDS) and any(action in lower for action in _DOC_QA_ACTION_WORDS):
        return True
    return False


def _has_project_info_cues(lower: str) -> bool:
    return any(cue in lower for cue in _PROJECT_INFO_CUES)


def _aliases_from_title(title: str) -> set[str]:
    """Build searchable aliases from an uploaded document title."""
    stem = title.rsplit(".", 1)[0] if "." in title else title
    clean = re.sub(r"[_\-]+", " ", stem).strip().lower()
    clean = re.sub(r"\s+", " ", clean)
    aliases: set[str] = set()
    if clean:
        aliases.add(clean)

    tokens = [t for t in clean.split() if t]
    core_tokens = [t for t in tokens if t not in _TITLE_NOISE and not t.isdigit()]
    core = " ".join(core_tokens).strip()
    if len(core) >= 3:
        aliases.add(core)

    # Significant single tokens (e.g. "airis") — avoid short/noisy tokens.
    for token in core_tokens:
        if len(token) >= 5:
            aliases.add(token)

    return {a for a in aliases if a}


def _matches_document_entity(haystack: str, document_titles: Sequence[str]) -> bool:
    lower = haystack.lower()
    for title in document_titles:
        for alias in _aliases_from_title(title):
            if len(alias) >= 5 and alias in lower:
                return True
            # Multi-word aliases (e.g. "talk to my data")
            if " " in alias and alias in lower:
                return True
    return False


def _is_corpus_document_qa(
    lower: str,
    *,
    document_titles: Sequence[str] | None,
    context_texts: Sequence[str] | None,
) -> bool:
    """Route project questions to DOCUMENT_QA when they reference the user's corpus.

    Does NOT force every entity mention into RAG:
    - "what is AIRIS?" stays GENERAL_KNOWLEDGE
    - "AIRIS what tech stack were using" becomes DOCUMENT_QA when AIRIS docs exist
    """
    if not document_titles:
        return False
    if not _has_project_info_cues(lower):
        return False
    if _GENERIC_DEFINITION.match(lower):
        return False

    if _matches_document_entity(lower, document_titles):
        return True

    # Anaphoric follow-up: entity lives in recent conversation, cues in current turn.
    if context_texts:
        context_blob = " ".join(t.lower() for t in context_texts if t)
        if context_blob and _matches_document_entity(context_blob, document_titles):
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


def classify(
    question: str,
    *,
    document_titles: Sequence[str] | None = None,
    context_texts: Sequence[str] | None = None,
    request_id: str | None = None,
) -> Route:
    """Return the route for ``question`` using lightweight deterministic rules.

    Optional ``document_titles`` / ``context_texts`` enable corpus-aware routing
    for project questions without hard-coding brand names into GENERAL->RAG.
    """
    text = (question or "").strip()
    req_id = request_id or "N/A"
    if not text:
        logger.info('[AI ROUTER] request_id="%s" question="" selected_intent="GENERIC_CHAT" selected_route="generic_chat"', req_id)
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
    elif _is_corpus_document_qa(
        lower,
        document_titles=document_titles,
        context_texts=context_texts,
    ):
        route = Route.DOCUMENT_QA
    elif _is_calculator(text, lower):
        route = Route.CALCULATOR
    else:
        # General questions (dates, current events, facts) -> web search
        route = Route.GENERAL_KNOWLEDGE

    logger.info(
        '[AI ROUTER] request_id="%s" query="%s" norm="%s" selected_intent="%s" selected_route="%s"',
        req_id,
        text[:200],
        lower[:200],
        route.name,
        route.value,
    )
    return route
