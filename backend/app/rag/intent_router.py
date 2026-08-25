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
    WEB = "WEB"


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
    "in my project",
    "in the project",
    "in project",
    "my project",
)

_DOC_QA_CUE_WORDS = (
    "document", "documents", "documentation", "file", "files", "policy", "policies",
    "doc", "docs", "guide", "guides", "manual", "manuals",
    "handbook", "handbooks", "sheet", "sheets", "prd", "specification",
    "problem", "statement", "architecture", "requirements",
)

_DOC_QA_ACTION_WORDS = (
    "say", "state", "mention", "contain", "in", "according",
    "what", "how", "tell", "explain", "summarize", "summarise", "describe",
    "show", "find", "search", "details", "about", "abt", "ssl", "nginx", "setup",
    "compare", "comparison", "match", "check", "contrast",
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
    "problem statement",
    "problem",
    "statement",
    "prd",
    "requirements",
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
    "deployment",
    "deploy",
    "deployed",
    "deploying",
    "setup",
    "vm",
    "virtual machine",
    "server",
    "process",
    "process manager",
    "reverse proxy",
    "proxy",
    "port",
    "ports",
    "pm2",
    "nginx",
    "install",
    "installation",
    "python",
    "version",
    "required",
    "leave",
    "policy",
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
    "project",
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
    "what documents u have",
    "what files u have",
    "what are documents u have",
    "what are the documents u have",
    "what are files u have",
    "what are the files u have",
    "documents u have list it",
    "documents u have list",
    "files u have list",
    "what doc you have",
    "what docs you have",
    "what file you have",
    "what documents you have",
    "what files you have",
    "what are documents you have",
    "what are the documents you have",
    "what are files you have",
    "what are the files you have",
    "documents you have list it",
    "documents you have list",
    "files you have list",
)

_DOC_LIST_REGEX = re.compile(
    r"\b(?:list|show|get|display|count)\s+(?:out\s+)?(?:all\s+)?(?:my\s+)?(?:uploaded\s+)?(?:documents?|files?|docs?)\b|"
    r"\bwhat\s+(?:are\s+)?(?:the\s+)?(?:documents?|files?|docs?)\s+(?:do\s+)?(?:u|you)?\s*(?:have|uploaded|available)?\b|"
    r"\bwhich\s+(?:documents?|files?|docs?)\s+(?:do\s+)?(?:u|you)?\s*(?:have|uploaded|available)?\b|"
    r"\bwhat\s+(?:documents?|files?|docs?)\s+(?:are\s+)?(?:available|uploaded)\b",
    re.IGNORECASE,
)

# Document metadata cues
_DOC_METADATA_KEYWORDS = (
    "when was file",
    "when was document",
    "when this file",
    "when document",
    "upload date",
    "date of upload",
    "when uploaded",
    "file size",
    "size of document",
    "who uploaded",
    "version of file",
    "version of document",
    "version of the document",
    "version of the file",
    "document version",
    "file version",
)


# Creative / generative task verbs — these are NEVER document lookups regardless of context.
# Queries starting with these verbs should always route to GENERAL_KNOWLEDGE or DIRECT.
_CREATIVE_GENERATION_VERBS = re.compile(
    r"^(?:"
    r"write\b|create\b|generate\b|make\b|build\b|code\b|draft\b|"
    r"design\b|implement\b|program\b|develop\b|produce\b|"
    r"give me\b|help me\b|show me how\b|how to\b|how do i\b|"
    r"write a\b|write me\b|create a\b|generate a\b|make a\b|build a\b|"
    r"draw\b|sketch\b|plan\b|explain how to\b|teach me\b|tell me how\b"
    r")",
    re.IGNORECASE,
)


def _is_generic_chat(lower: str) -> bool:
    if _GREETING_PATTERNS.match(lower):
        return True
    return any(lower == phrase or lower.startswith(phrase + " ") for phrase in _CHAT_PHRASES)


def _is_document_metadata(lower: str) -> bool:
    if any(kw in lower for kw in _DOC_METADATA_KEYWORDS):
        return True
    if ("when" in lower or "date" in lower or "time" in lower or "who" in lower or "size" in lower) and ("uploaded" in lower or "upload" in lower or "created" in lower or "modified" in lower or "added" in lower):
        return True
    return False


def _is_document_list(lower: str) -> bool:
    if _is_document_metadata(lower):
        return False
    if any(w in lower for w in ["about", "inside", "content", "summary", "summarize", "summarise", "detail", "explain", "policy"]):
        return False
    if _DOC_LIST_REGEX.search(lower):
        return True
    if any(kw in lower for kw in _DOC_LIST_KEYWORDS):
        return True
    doc_target = any(term in lower for term in ["document", "documents", "file", "files", "doc", "docs"])
    list_action = any(term in lower for term in ["list", "show", "have", "available", "uploaded"])
    if doc_target and list_action and not any(w in lower for w in ["what is", "how to", "why", "where is", "according to", "when"]):
        return True
    return False


def _is_document_qa(text: str, lower: str) -> bool:
    if _DOC_EXTENSION.search(text) and not _is_document_metadata(lower):
        return True
    if any(phrase in lower for phrase in _DOC_QA_PHRASES):
        return True
    word_tokens = set(re.findall(r"\b\w+\b", lower))
    if any(noun in word_tokens for noun in _DOC_QA_CUE_WORDS) and any(action in word_tokens for action in _DOC_QA_ACTION_WORDS):
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
    - "write a login page" stays GENERAL_KNOWLEDGE even with corpus context
    """
    if not document_titles:
        return False
    if _GENERIC_DEFINITION.match(lower):
        return False
    # Creative/generative tasks are never document lookups even in a doc session
    if _CREATIVE_GENERATION_VERBS.match(lower.strip()):
        return False

    has_entity = _matches_document_entity(lower, document_titles)

    # Anaphoric follow-up: entity lives in recent conversation, cues in current turn.
    if not has_entity and context_texts:
        context_blob = " ".join(t.lower() for t in context_texts if t)
        if context_blob and _matches_document_entity(context_blob, document_titles):
            # The query itself doesn't mention the entity, but the history does.
            # We should only set has_entity = True if the query has pronouns/anaphora
            # or direct technical project cues referencing the system.
            anaphora_cues = [" it ", " this ", " that ", " the system ", " the project ", " the tool ", " the app ", " the codebase ", " the document ", " the file ", " the code ", " they "]
            has_anaphora = any(cue in f" {lower} " for cue in anaphora_cues) or _has_project_info_cues(lower)
            if has_anaphora:
                has_entity = True

    if not has_entity:
        return False

    # Route if the query contains project info cues or standard question action words
    if _has_project_info_cues(lower):
        return True
    if any(action in lower for action in _DOC_QA_ACTION_WORDS):
        return True
    if any(req in lower for req in ["tell me", "explain", "about", "describe", "what is"]):
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


def _is_datetime_query(lower: str) -> bool:
    datetime_phrases = (
        "today date", "today's date", "todays date", "date today", "the date today",
        "what is the date", "what date is it", "current date", "what time is it",
        "current time", "what day is today", "what day is it today", "time right now",
        "today's time", "current day"
    )
    return any(p in lower for p in datetime_phrases)


def _is_web_query(text: str, lower: str) -> bool:
    if _is_datetime_query(lower):
        return False

    # Explicit web search intent phrases
    web_phrases = (
        "search the web", "search web", "web search", "search online",
        "find online", "look up online", "search for", "google", "browse",
        "latest news", "current version", "what is the latest", "who is the current",
        "latest python", "latest react", "latest version", "current react"
    )
    if any(phrase in lower for phrase in web_phrases):
        return True

    # Look for keywords indicating real-time info or search queries
    web_keywords = {
        "weather", "tomorrow", "yesterday",
        "news", "stock", "price", "good friday",
        "forecast", "temperature", "temp", "latest", "recent", "who won"
    }
    if any(kw in lower for kw in web_keywords):
        return True
    # If the question contains a specific 4-digit year like 2024, 2025, 2026
    if re.search(r"\b20\d{2}\b", lower) and not any(k in lower for k in ("date", "time")):
        return True
    # If London/cities or similar real-time queries are present, or "weather in ..."
    if "london" in lower:
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
    text = (question or "").strip().strip('"\'`')
    req_id = request_id or "N/A"
    if not text:
        logger.info('[AI ROUTER] request_id="%s" question="" selected_intent="GENERIC_CHAT" selected_route="generic_chat"', req_id)
        return Route.GENERIC_CHAT

    from app.rag.query_normalizer import normalize_query
    _, norm, _ = normalize_query(text)
    lower = norm.lower() if norm else text.lower()

    if _is_datetime_query(lower):
        route = Route.WEB
    elif _is_generic_chat(lower):
        route = Route.GENERIC_CHAT
    elif len(lower.split()) == 1:
        route = Route.DIRECT
    elif _is_document_metadata(lower):
        route = Route.DOCUMENT_METADATA
    elif _is_document_list(lower):
        route = Route.DOCUMENT_LIST
    elif _is_calculator(text, lower):
        route = Route.CALCULATOR
    elif _is_document_qa(text, lower):
        route = Route.DOCUMENT_QA
    elif _is_corpus_document_qa(
        lower,
        document_titles=document_titles,
        context_texts=context_texts,
    ):
        route = Route.DOCUMENT_QA
    elif _is_web_query(text, lower):
        route = Route.WEB
    else:
        # Default fallback for general questions without document cues
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


# ---------------------------------------------------------------------------
# OpenJarvis Query Complexity & Dynamic Token Allocation
# ---------------------------------------------------------------------------

_COMPLEXITY_CODE_PATTERNS = re.compile(
    r"```|`[^`]+`|\bdef\s|\bclass\s|\bimport\s|\bfunction\s|\bconst\s|\bvar\s|\blet\s|"
    r"\bif\s*\(|->|=>|\{\s*\}|\bfor\s+\w+\s+in\s|#include|System\.out",
    re.IGNORECASE,
)
_COMPLEXITY_MATH_PATTERNS = re.compile(
    r"\bsolve\b|\bintegral\b|\bequation\b|\bproof\b|\bderivative\b|\bmatrix\b|"
    r"\btheorem\b|\bcalculate\b|\bcompute\b|\bsigma\b|\bsum\b|\blimit\b|\bprobability\b",
    re.IGNORECASE,
)
_COMPLEXITY_REASONING_PATTERNS = re.compile(
    r"\bexplain\b|\banalyze\b|\bcompare\b|\bwhy\b"
    r"|\bstep[- ]by[- ]step\b|\breason\b|\bthink\b"
    r"|\bpros\s+and\s+cons\b|\btrade-?\s*offs?\b|\bevaluate\b",
    re.IGNORECASE,
)
_COMPLEXITY_MULTI_STEP_PATTERNS = re.compile(
    r"\bthen\b.*\bthen\b|\bfirst\b.*\bnext\b|\bstep\s*\d"
    r"|\b(?:and\s+also|additionally|furthermore)\b"
    r"|\b\d+\.\s",
    re.IGNORECASE | re.DOTALL,
)
_THINKING_MODEL_PATTERNS = re.compile(
    r"qwen3|qwq|deepseek-r1|o1-|o3-|o4-", re.IGNORECASE
)

_TOKEN_TIERS = {
    "trivial": 256,
    "simple": 512,
    "moderate": 768,
    "complex": 1024,
    "very_complex": 2048,
}

# Alias for backward compatibility with regression test suites
route_question = classify


def analyze_complexity(query: str, model_name: str | None = None) -> int:
    """Analyze query complexity and return the recommended max_tokens generation budget.
    
    Ported from OpenJarvis learning/routing/complexity.py.
    """
    text = (query or "").strip()
    if not text or len(text.split()) <= 3:
        tier = "trivial"
    elif _COMPLEXITY_CODE_PATTERNS.search(text) or _COMPLEXITY_MULTI_STEP_PATTERNS.search(text):
        tier = "complex"
    elif _COMPLEXITY_MATH_PATTERNS.search(text) or _COMPLEXITY_REASONING_PATTERNS.search(text):
        tier = "moderate"
    elif len(text.split()) > 30:
        tier = "moderate"
    else:
        tier = "simple"

    base_tokens = _TOKEN_TIERS[tier]
    logger.info("[QUERY COMPLEXITY] query_len=%d tier=%s max_tokens=%d model=%s", len(text), tier, base_tokens, model_name or "default")
    return base_tokens

