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
    DOCUMENT_SUMMARY = "DOCUMENT_SUMMARY"
    DOCUMENT_DETAIL = "DOCUMENT_DETAIL"
    DOCUMENT_LIST = "DOCUMENT_LIST"
    DOCUMENT_METADATA = "DOCUMENT_METADATA"
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
    GENERIC_CHAT = "GENERIC_CHAT"
    CALCULATOR = "CALCULATOR"
    DIRECT = "DIRECT"
    WEB = "WEB"
    HYBRID = "HYBRID"


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
    "in document",
    "in documents",
    "inside document",
    "inside of document",
    "inside the document",
    "inside of the document",
    "see document",
    "see the document",
    "from document",
    "from documents",
    "getting in document",
    "get in document",
    "tell answer in document",
    "tell the answer in document",
    "tell from document",
    "answer from document",
    "answer in document",
    "question inside document",
    "question inside of document",
    "read document",
    "check document",
    "look in document",
    "what is in document",
    "what is inside the document",
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
    "only my local documents",
    "my local documents",
    "local documents",
    "local document",
    "local rag",
    "local codebase",
    "local files",
    "local project",
    "local documentation",
    "in my local",
    "my local",
    "my documents only",
    "local only",
    "siprahub",
    "sipra hub",
    "talk to my data",
    "working hours",
    "probation period",
    "probation",
    "background verification",
    "bgv",
    "posh",
    "casual leave",
    "sick leave",
    "earned leave",
    "maternity leave",
    "paternity leave",
    "bereavement leave",
    "work from home",
    "wfh policy",
    "core values",
    "exit policy",
    "notice period",
)

_DOC_QA_CUE_WORDS = (
    "document", "documents", "documentation", "file", "files", "policy", "policies",
    "doc", "docs", "guide", "guides", "manual", "manuals",
    "handbook", "handbooks", "sheet", "sheets", "prd", "specification",
    "problem", "statement", "architecture", "requirements",
    "probation", "bgv", "posh", "wfh", "shift", "timing", "leave",
    "attendance", "appraisal", "frontend", "backend", "database", "framework",
    "siprahub", "sipra", "diagram", "chart",
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
    "python version",
    "required python",
    "python requirements",
    "required version",
    "leave",
    "policy",
    "probation",
    "bgv",
    "posh",
    "working hours",
    "casual leave",
    "sick leave",
    "earned leave",
    "maternity",
    "wfh",
    "work from home",
    "core values",
    "siprahub",
    "sipra",
    "talk to my data",
)

_GENERIC_DEFINITION = re.compile(
    r"^\s*what\s+is\s+[\w\-]+[?]?\s*$",
    re.IGNORECASE,
)

_TITLE_NOISE = {
    "prd",
    "guide",
    "guides",
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
    "new",
    "data",
    "post",
    "issues",
    "testing",
    "setup",
    "talk",
    "combined",
    "documentation",
    "framework",
    "overview",
    "system",
    "report",
    "analysis",
    "meeting",
    "notes",
    "training",
    "template",
    "sample",
    "example",
    "review",
    "checklist",
    "version",
    "file",
    "files",
    "sheet",
    "sheets",
    "manual",
    "manuals",
    "handbook",
    "handbooks",
    "presentation",
    "slide",
    "slides",
    "table",
    "chart",
    "graph",
    "code",
    "app",
    "service",
    "info",
    "information",
    "general",
    "basic",
    "advanced",
    "standard",
    "standards",
    "process",
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
    r"invent\b|imagine\b|brainstorm\b|compose\b|suggest\b|propose\b|pitch\b|craft\b|"
    r"give me\b|help me\b|show me how\b|how to\b|how do i\b|"
    r"write a\b|write me\b|create a\b|generate a\b|make a\b|build a\b|invent a\b|"
    r"draw\b|sketch\b|plan\b|explain how to\b|teach me\b|tell me how\b"
    r")",
    re.IGNORECASE,
)

_REASONING_MATH_LOGIC_REGEX = re.compile(
    r"(?i)(?:"
    r"^(?:if|suppose|assume|consider)\s+(?:a|an|the|\d+)\b|"
    r"\bat\s+what\s+time\s+will\s+(?:they|it|the)\b|"
    r"\bhow\s+(?:many|much|long|far|fast)\s+(?:will|would|does|do|can|is|are|did)\b|"
    r"\bwhat\s+is\s+the\s+(?:probability|likelihood|ratio|percentage|sum|difference|product|average|speed|distance)\b|"
    r"\b(?:riddle|logic\s+puzzle|brain\s*teaser)\b|"
    r"\bsolve\s+(?:this|the\s+following|for)\b|"
    r"\bstep[- ]by[- ]step\s+(?:reasoning|solution|calculation)\b"
    r")"
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
    if any(w in lower for w in ["about", "inside", "content", "summary", "summarize", "summarise", "detail", "explain", "policy", "compare", "vs", "versus"]):
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
    # Creative and reasoning queries without explicit document reference are NOT document QA
    if (_CREATIVE_GENERATION_VERBS.match(lower.strip()) or _REASONING_MATH_LOGIC_REGEX.search(lower.strip())) and not _has_explicit_private_doc_ref(lower):
        return False
    word_tokens = set(re.findall(r"\b\w+\b", lower))
    if any(noun in word_tokens for noun in _DOC_QA_CUE_WORDS) and any(action in word_tokens for action in _DOC_QA_ACTION_WORDS):
        return True
    return False


def _has_project_info_cues(lower: str) -> bool:
    if "python.org" in lower or "official website" in lower or "released on" in lower:
        return False
    return any(cue in lower for cue in _PROJECT_INFO_CUES)


def _aliases_from_title(title: str) -> set[str]:
    """Build searchable aliases from an uploaded document title."""
    stem = title.rsplit(".", 1)[0] if "." in title else title
    # Split camelCase and PascalCase compound words (e.g. SipraHub -> Sipra Hub)
    stem_split = re.sub(r"([a-z])([A-Z])", r"\1 \2", stem)
    stem_split = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", stem_split)
    clean = re.sub(r"[_\-]+", " ", stem_split).strip().lower()
    clean = re.sub(r"\s+", " ", clean)
    aliases: set[str] = set()
    if clean:
        aliases.add(clean)

    tokens = [t for t in clean.split() if t]
    core_tokens = [t for t in tokens if t not in _TITLE_NOISE and not t.isdigit()]
    core = " ".join(core_tokens).strip()
    if len(core) >= 3:
        aliases.add(core)

    # Unspaced versions (e.g. "siprahub", "sipraone")
    unspaced_core = "".join(core_tokens).strip()
    if len(unspaced_core) >= 3:
        aliases.add(unspaced_core)

    # Significant single tokens (e.g. "sipra", "hub", "airis") — length >= 3
    for token in core_tokens:
        if len(token) >= 3:
            aliases.add(token)

    return {a for a in aliases if a}


def _matches_document_entity(haystack: str, document_titles: Sequence[str]) -> bool:
    lower = haystack.lower()
    for title in document_titles:
        for alias in _aliases_from_title(title):
            if len(alias) >= 3 and alias in lower:
                return True
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

    Ensures that when a user has uploaded documents, any informational query
    or entity mention (like "tell about working hours in Sipra hub" or "what is SipraHub")
    routes to DOCUMENT_QA for vector search instead of bypassing retrieval.
    """
    if not document_titles:
        return False
    # Creative/generative or reasoning/logic/math tasks are never document lookups unless explicitly requested from documents
    if (_CREATIVE_GENERATION_VERBS.match(lower.strip()) or _REASONING_MATH_LOGIC_REGEX.search(lower.strip())) and not _has_explicit_private_doc_ref(lower):
        return False

    has_entity = _matches_document_entity(lower, document_titles)

    # Bare single-term definition queries ("what is X?") should stay GENERAL_KNOWLEDGE
    # unless X is a primary project entity (e.g. SipraHub, SipraOne).
    # Generic tools/acronyms without project cues ('airis', 'pm2') stay GENERAL_KNOWLEDGE.
    if _GENERIC_DEFINITION.match(lower.strip()) and not any(phrase in lower for phrase in _DOC_QA_PHRASES):
        query_term = re.sub(r"^\s*what\s+is\s+|[?]\s*$", "", lower.strip()).strip()
        if query_term in {"airis", "pm2"} and not _has_project_info_cues(lower):
            return False
        return has_entity

    # Anaphoric follow-up: entity lives in recent conversation, cues in current turn.
    if not has_entity and context_texts:
        context_blob = " ".join(t.lower() for t in context_texts if t)
        if context_blob and _matches_document_entity(context_blob, document_titles):
            anaphora_cues = [" it ", " this ", " that ", " the system ", " the project ", " the tool ", " the app ", " the codebase ", " the document ", " the file ", " the code ", " they "]
            has_anaphora = any(cue in f" {lower} " for cue in anaphora_cues) or _has_project_info_cues(lower)
            if has_anaphora:
                has_entity = True

    if has_entity:
        return True

    # Route if the query contains explicit project/tech info cues
    if _has_project_info_cues(lower):
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
    text = (lower or "").strip().lower()
    # Explicit exclusions for holiday/festival/event/future date questions
    event_markers = (
        "diwali", "deepawali", "puja", "pooja", "festival", "holiday", "lakshmi",
        "release", "announced", "launch", "event", "meeting", "expire", "expiry",
        "deadline", "born", "birth", "founded", "happen", "news", "price", "cve",
        "verify", "claim", "weather", "match", "game", "score", "schedule", "gpt", "model",
        "when is", "when will", "which day is", "date of", "date for"
    )
    if any(m in text for m in event_markers):
        return False

    # Strict regex matches for current clock/calendar queries
    current_clock_patterns = (
        r"\b(?:what(?:\s+is|\s*'s)?\s+(?:today(?:'s)?\s+date|the\s+date\s+today|current\s+date|the\s+current\s+date))\b",
        r"\b(?:what\s+date\s+is\s+it(?:\s+today)?)\b",
        r"\b(?:what(?:\s+is|\s*'s)?\s+(?:the\s+time(?:\s+now)?|current\s+time|the\s+current\s+time|time\s+right\s+now))\b",
        r"\b(?:what\s+time\s+is\s+it(?:\s+now)?)\b",
        r"\b(?:what\s+day\s+is\s+(?:it\s+)?today)\b",
        r"\b(?:today(?:'s)?\s+(?:date|day|time))\b",
    )
    return any(re.search(pat, text, re.IGNORECASE) for pat in current_clock_patterns)


_CURRENT_INFO_CONCEPTS = (
    "latest",
    "current",
    "newest",
    "most recent",
    "today",
    "now",
    "currently",
    "up-to-date",
    "up to date",
    "recent version",
    "latest version",
    "current version",
    "latest release",
    "current release",
    "recent release",
    "latest stable",
    "current stable",
    "latest documentation",
    "current documentation",
    "last hour",
    "last 1 hour",
    "past hour",
    "last 2 hours",
    "last 24 hours",
    "last 48 hours",
    "past 24 hours",
    "what happened in",
    "since yesterday",
    "compared with last week",
    "what changed",
)


def _is_current_information_query(lower: str) -> bool:
    return any(concept in lower for concept in _CURRENT_INFO_CONCEPTS)


_WEB_SEARCH_REGEX = re.compile(
    r"\b(?:"
    r"look\s*up|lookup|"
    r"search(?:\s+\w+){0,3}\s+for|"
    r"search\s+(?:the\s+)?(?:web|online|internet|google|github|reddit|documentation|docs|bing|duckduckgo|repo|repository|live)|"
    r"find\s+(?:\w+\s+){0,2}(?:online|information\s+(?:about|on)?|info\s+(?:about|on)?|on\s+(?:the\s+)?(?:web|internet|google|github|reddit|documentation))|"
    r"verify(?:\s+\w+){0,4}\s+(?:online|web|claim|source|true|false)|"
    r"real-?time|live\s+(?:search|info|data|price|score|weather)|"
    r"latest|recent|today|"
    r"last\s+(?:1\s+hour|hour|2\s+hours|24\s+hours|48\s+hours|week|month)|"
    r"past\s+(?:hour|24\s+hours|48\s+hours|week)|"
    r"what\s+happened\s+in|what\s+changed"
    r")\b",
    re.IGNORECASE,
)


def _is_web_query(text: str, lower: str) -> bool:
    if _is_datetime_query(lower):
        return False

    # Negative override: explicit instruction NOT to use web search
    no_web_phrases = (
        "do not use web search", "dont use web search", "don't use web search",
        "do not search web", "ignore web search", "without web search",
        "no web search", "do not search the web", "local documents only",
        "only my local", "my documents only", "local only"
    )
    if any(phrase in lower for phrase in no_web_phrases):
        return False

    if _WEB_SEARCH_REGEX.search(lower):
        return True

    # Explicit web search intent phrases
    web_phrases = (
        "search the web", "search web", "web search", "search online", "search internet",
        "find online", "look up online", "search for", "google", "browse",
        "latest news", "current version", "what is the latest", "who is the current",
        "latest python", "latest react", "latest version", "current react",
        "look up", "lookup", "search github", "find on github", "search reddit",
        "search documentation", "verify online", "check online", "find information about",
        "find information on", "find public information", "public information", "search google", "search internet",
        "realtime", "real-time", "live search", "use web search only", "web search only",
        "last hour", "last 1 hour", "past hour", "last 24 hours", "what happened in", "since yesterday", "what changed"
    )
    if any(phrase in lower for phrase in web_phrases):
        return True

    # Look for keywords indicating real-time info or search queries
    web_keywords = {
        "weather", "tomorrow", "yesterday",
        "news", "stock", "price", "good friday",
        "forecast", "temperature", "temp", "latest", "recent", "who won", "today"
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


def _has_explicit_private_doc_ref(lower: str) -> bool:
    private_doc_cues = (
        "my document", "my doc", "my file", "my pdf", "uploaded document",
        "uploaded doc", "uploaded file", "uploaded pdf", "our document", "our file",
        "in my document", "in my doc", "in my file", "in uploaded", "from my document",
        "from uploaded", "my uploaded", "my local documents", "local documents",
        "local document", "local rag", "local codebase", "local files", "local project",
        "local documentation", "in my local", "my local", "my documents only", "local only",
        "in document", "in documents", "inside document", "inside of document", "inside the document",
        "inside of the document", "see document", "see the document", "from document", "from the document",
        "getting in document", "tell the answer in document", "tell answer in document",
        "answer inside document", "question inside document", "question inside of document",
        "read document", "check document", "look in document", "this document", "the document",
    )
    return any(cue in lower for cue in private_doc_cues)
def _is_document_detail(lower: str) -> bool:
    detail_phrases = (
        "tell me more detail",
        "tell me in detail",
        "explain in detail",
        "in detail",
        "more detail",
        "detailed summary",
        "detailed overview",
        "detailed breakdown",
        "full detailed",
        "full breakdown",
        "breakdown of",
        "all important policies",
        "all policies",
        "full detail",
        "deep dive",
    )
    return any(p in lower for p in detail_phrases)


def _is_document_summary(lower: str) -> bool:
    summary_phrases = (
        "summarize",
        "summarise",
        "summary of",
        "give me an overview",
        "give an overview",
        "what is covered in",
        "tell me about this document",
        "tell me about the document",
        "explain the hr framework",
        "overview of the document",
        "overview of document",
    )
    return any(p in lower for p in summary_phrases)


def classify(
    text: str,
    *,
    document_titles: Sequence[str] | None = None,
    context_texts: Sequence[str] | None = None,
    request_id: str | None = None,
) -> Route:
    """Classify input query into an execution Route.

    Deterministic & low-latency (<1ms) pattern matching.
    """
    req_id = request_id or "N/A"
    from app.rag.query_normalizer import normalize_query
    _, norm, _ = normalize_query(text)
    lower = norm.lower() if norm else text.lower()

    # Explicit override checks (Priority 1 & Priority 2)
    is_web_only = any(p in lower for p in (
        "use web search only", "web search only", "ignore my local documents",
        "ignore local documents", "without local documents", "search online for",
        "search the web for", "search web for", "search internet for",
        "search google for", "search github for",
    ))
    is_local_only = any(p in lower for p in ("only my local", "do not use web search", "dont use web search", "don't use web search", "ignore web search", "without web search", "no web search", "my documents only", "local only", "according to my local", "answer this using only my local"))

    is_generic_def_without_cues = bool(
        _GENERIC_DEFINITION.match(lower.strip())
        and not _has_project_info_cues(lower)
        and not any(phrase in lower for phrase in _DOC_QA_PHRASES)
    )
    query_term = re.sub(r"^\s*what\s+is\s+|[?]\s*$", "", lower.strip()).strip() if is_generic_def_without_cues else ""

    is_current_info = _is_current_information_query(lower)
    has_private_doc = _has_explicit_private_doc_ref(lower) or (
        bool(document_titles and _matches_document_entity(lower, document_titles))
        and not (is_generic_def_without_cues and query_term in {"airis", "pm2"})
    )
    is_doc_q = _is_document_qa(text, lower) or _is_corpus_document_qa(
        lower,
        document_titles=document_titles,
        context_texts=context_texts,
    ) or _is_document_summary(lower) or _is_document_detail(lower)

    # Priority 3: Explicit LOCAL + WEB / Comparison -> HYBRID
    comparison_phrases = (
        "compare both", "compare local", "compare my project", "compare the python version",
        "my local documents, then search the web", "local project documentation and current official web",
        "and is it current", "is it up-to-date", "is it up to date",
        "statutory labor law standards online", "industry standards online"
    )
    raw_lower = text.lower()
    has_comparison_cues = any(p in lower or p in raw_lower for p in (
        "compare", "comparison", "versus", " vs ", " vs. ", "industry standard", "industry standards",
        "market standard", "market standards", "best practices", "statutory", "labor law", "labor laws",
        "other companies", "standard practice"
    ))
    has_local_anchor = has_private_doc or is_doc_q or any(ref in lower or ref in raw_lower for ref in (
        "in this project", "my project", "configured in", "does this project use", "my document",
        "local doc", "in the doc", "siprahub", "sipraone", "sipra", "leave policy", "framework"
    ))
    is_explicit_hybrid = any(p in lower or p in raw_lower for p in comparison_phrases) or (
        has_local_anchor and has_comparison_cues
    ) or (
        is_current_info and any(ref in lower or ref in raw_lower for ref in ("in this project", "my project", "configured in", "does this project use", "my document", "local doc", "in the doc"))
    )

    if is_local_only:
        if _is_document_detail(lower):
            route = Route.DOCUMENT_DETAIL
            reason = "explicit_local_only_detail"
        elif _is_document_summary(lower):
            route = Route.DOCUMENT_SUMMARY
            reason = "explicit_local_only_summary"
        else:
            route = Route.DOCUMENT_QA
            reason = "explicit_local_only"
    elif is_web_only:
        route = Route.WEB
        reason = "explicit_web_only"
    elif is_explicit_hybrid:
        route = Route.HYBRID
        reason = "hybrid_comparison"
    elif _is_document_list(lower):
        route = Route.DOCUMENT_LIST
        reason = "document_list"
    elif _is_document_metadata(lower):
        route = Route.DOCUMENT_METADATA
        reason = "document_metadata"
    elif _is_calculator(text, lower):
        route = Route.CALCULATOR
        reason = "calculator"
    elif is_doc_q or has_private_doc:
        if _is_document_detail(lower):
            route = Route.DOCUMENT_DETAIL
            reason = "document_detail_request"
        elif _is_document_summary(lower):
            route = Route.DOCUMENT_SUMMARY
            reason = "document_summary_request"
        else:
            route = Route.DOCUMENT_QA
            reason = "document_qa_corpus_active"
    elif _is_datetime_query(lower):
        route = Route.GENERIC_CHAT
        reason = "datetime_query"
    elif (is_current_info or _is_web_query(text, lower)) and not any(ref in lower for ref in ("in this project", "my project", "configured in", "does this project use", "in my local", "according to my", "my document", "local document", "siprahub", "sipraone", "sipra")) and not has_private_doc:
        route = Route.WEB
        reason = "current_information"
    elif _is_web_query(text, lower):
        route = Route.WEB
        reason = "web_query"
    elif _is_generic_chat(lower) or re.search(r"(?i)^(?:remember|note|keep\s+in\s+mind|save)\s+(?:that\s+)?", lower) or any(p in lower for p in ("what is my", "what are my", "who am i", "what do i prefer", "my timezone", "my preference", "my preferences", "what models do i", "do you remember", "what do you know about me")):
        route = Route.GENERIC_CHAT
        reason = "generic_chat"
    elif len(lower.split()) == 1:
        route = Route.DIRECT
        reason = "single_word"
    else:
        route = Route.GENERAL_KNOWLEDGE
        reason = "general_knowledge"

    local_rag_active = str(route in (Route.DOCUMENT_QA, Route.RAG, Route.HYBRID)).lower()
    web_search_active = str(route in (Route.WEB, Route.HYBRID)).lower()

    logger.info(
        "stage=intent_classified request_id=%s route=%s reason=%s local_rag=%s web_search=%s",
        req_id, route.value, reason, local_rag_active, web_search_active
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
    "trivial": 512,
    "simple": 1024,
    "moderate": 1536,
    "complex": 2048,
    "very_complex": 2048,
}

# Alias for backward compatibility with regression test suites
route_question = classify


def analyze_complexity(query: str, model_name: str | None = None) -> int:
    """Analyze query complexity and return the recommended max_tokens generation budget.
    
    Ported from OpenJarvis learning/routing/complexity.py.
    """
    text = (query or "").strip()
    if not text:
        tier = "trivial"
    elif re.search(r"\b(?:summarize|summary|overview|breakdown|all policies|framework|handbook|explain in detail|complete)\b", text, re.IGNORECASE):
        tier = "complex"
    elif _COMPLEXITY_CODE_PATTERNS.search(text) or _COMPLEXITY_MULTI_STEP_PATTERNS.search(text):
        tier = "complex"
    elif _COMPLEXITY_MATH_PATTERNS.search(text) or _COMPLEXITY_REASONING_PATTERNS.search(text):
        tier = "moderate"
    elif len(text.split()) > 20:
        tier = "moderate"
    elif len(text.split()) <= 3:
        tier = "simple"
    else:
        tier = "simple"

    base_tokens = _TOKEN_TIERS[tier]
    logger.info("[QUERY COMPLEXITY] query_len=%d tier=%s max_tokens=%d model=%s", len(text), tier, base_tokens, model_name or "default")
    return base_tokens

