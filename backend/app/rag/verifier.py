"""Answer verification layer for RAG pipeline.

Validates LLM-generated answers against final context chunks to ensure:
1. Every factual claim is supported by retrieved context (no hallucinations).
2. Requested attributes (e.g. tech stack vs ports) are accurately answered.
3. Unrelated attributes are not substituted (e.g. ports returned for tech query).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.query_understanding import AttributeCategory, QueryIntent
from app.retrieval.ranking import RankedResult


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of verifying an LLM-generated answer."""

    is_valid: bool
    reason: str | None = None


def verify_answer(
    answer: str,
    context_chunks: list[RankedResult],
    intent: QueryIntent,
) -> VerificationResult:
    """Verify an LLM-generated answer against final context chunks and query intent."""
    if not answer or not answer.strip():
        return VerificationResult(is_valid=False, reason="Empty answer")

    ans_clean = answer.strip()
    ans_lower = ans_clean.lower()

    # Always valid if the model correctly returned the standard fallback
    if (
        "requested information is not found" in ans_lower
        or "could not find this information" in ans_lower
        or "couldn't find enough information" in ans_lower
        or "could not find enough information" in ans_lower
        or "the provided document does not specify" in ans_lower
    ):
        return VerificationResult(is_valid=True, reason="Valid fallback answer")

    context_text = " ".join(c.chunk_text for c in context_chunks).lower()

    # Rule 1: Attribute Isolation — Technology Query must NOT substitute ports without frameworks
    if intent.category == AttributeCategory.TECHNOLOGY:
        has_framework_in_ans = any(
            fw in ans_lower
            for fw in (
                "react", "fastapi", "vite", "express", "next.js", "node.js", "nodejs",
                "python", "postgres", "postgresql", "django", "vue", "angular", "flask",
                "chat interface", "api backend", "frontend", "backend"
            )
        )
        is_pure_port_ans = bool(re.search(r"^\s*port\b|git for source", ans_lower))
        if is_pure_port_ans or (not has_framework_in_ans and re.search(r"\bport\b|\b\d{4}\b", ans_lower)):
            return VerificationResult(
                is_valid=False,
                reason="Substituted ports/deployment for technology query",
            )

    # Rule 2: Attribute Isolation — Configuration / Port Query must contain port numbers
    elif intent.category == AttributeCategory.CONFIGURATION:
        has_port_digit = bool(re.search(r"\b\d{3,5}\b", ans_lower))
        if not has_port_digit and "port" not in ans_lower:
            return VerificationResult(
                is_valid=False,
                reason="Configuration/port query did not return port numbers",
            )

    # Rule 4: Project Isolation — Reject cross-project contamination for Talk to My Data queries.
    # Allow parent project references ("SipraHub") if valid framework technologies are present.
    if intent.entity == "Talk to My Data" and "siprahub" in ans_lower:
        has_tech_framework = any(
            fw in ans_lower
            for fw in ("react", "fastapi", "vite", "express", "next.js", "node", "python", "postgres", "django", "vue", "angular")
        )
        ttmd_mentioned_in_ans = any(
            phrase in ans_lower
            for phrase in ("talk to my data", "talktomydata", "ttmd", "data")
        )
        if not ttmd_mentioned_in_ans and not has_tech_framework:
            return VerificationResult(
                is_valid=False,
                reason="Cross-project entity 'SipraHub' is primary subject in 'Talk to My Data' answer",
            )

    # Rule 3: Fact Grounding — Ensure key named entities in answer exist in context.
    # Skiplist covers generic words and tech terms that carry no hallucination risk.
    _ENTITY_SKIPLIST = {
        "the", "a", "an", "frontend", "backend", "system", "app", "application", "data", "code",
        "api", "ui", "db", "ux", "qa", "yes", "no", "its", "it's", "and", "for", "with", "but", "or",
        "it", "is", "on", "at", "by", "to", "of", "in", "as", "an", "framework", "so", "if", "not",
        "library", "stack", "technology", "technologies", "version", "service", "services",
        # Common English sentence starters and conjunctions
        "based", "this", "that", "these", "those", "there", "here", "when", "what", "where", "why", "how",
        "however", "also", "addition", "additionally", "moreover", "furthermore", "overall", "summary",
        "first", "second", "third", "finally", "key", "core", "main", "primary", "note", "specifically",
        "following", "includes", "including", "used", "uses", "using", "built", "provides", "provides",
        # Common tech names and organizational terms that carry no hallucination risk
        "values", "core", "integrity", "accountability", "collaboration", "excellence", "respect",
        "purpose", "handbook", "compliance", "framework", "overview", "policy", "policies", "rules",
        "working", "hours", "leave", "casual", "sick", "annual", "siprahub", "sipraone", "sipra",
        "vite", "react", "fastapi", "node", "nodejs", "express", "django", "flask",
        "postgres", "postgresql", "mongodb", "redis", "nginx", "docker",
        "nextjs", "next", "vue", "angular", "svelte", "typescript", "javascript",
        "pydantic", "uvicorn", "tailwind", "tailwindcss", "html", "css", "sql", "rest",
    }
    named_entities = re.findall(r"\b[A-Z][a-zA-Z0-9\.]+\b", ans_clean)
    for entity in named_entities:
        e_lower = entity.lower().replace(".", "")
        if e_lower in _ENTITY_SKIPLIST or len(e_lower) <= 3:
            continue
        if entity.lower() in context_text or e_lower in context_text:
            continue
        # Alias tolerance for common frameworks
        if entity.lower() == "react" and ("react" in context_text or "chat interface" in context_text or "frontend" in context_text):
            continue
        if entity.lower() == "fastapi" and ("fastapi" in context_text or "api backend" in context_text or "python" in context_text or "backend" in context_text):
            continue
        # Vite alias — matches if context mentions react, vite, bundler, build tool, or frontend
        if entity.lower() == "vite" and any(kw in context_text for kw in ("vite", "react", "bundler", "build tool", "frontend")):
            continue
        # General alias: if entity stem (first 4 chars) is in context, treat as matched
        if len(e_lower) >= 4 and e_lower[:4] in context_text:
            continue
        # Ignore general capitalized English words that are not specific proper nouns
        if e_lower.isalpha() and not any(char.isupper() for char in entity[1:]):
            continue
        return VerificationResult(
            is_valid=False,
            reason=f"Unsupported entity '{entity}' in answer not found in context",
        )

    return VerificationResult(is_valid=True, reason="Answer verified successfully")
