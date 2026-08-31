"""Memory Extractor — converts conversation turns into structured long-term memories.

Extraction Strategy
-------------------
The default mode is rule-based (MEMORY_EXTRACTOR=rule):
- Zero additional LLM calls → no latency overhead for local Qwen3 8B
- Detects preference statements, model choices, project context, goals,
  technical context via pattern matching and keyword heuristics

Optional LLM mode (MEMORY_EXTRACTOR=llm):
- Uses a single call to a small/fast model to extract richer memories
- Controlled by MEMORY_EXTRACTOR env var
- Falls back to rule-based on LLM failure

Safety
------
- Sensitive data patterns (API keys, passwords, tokens) are detected and
  blocked before any extraction proceeds.
- Extracted content is always treated as DATA — it is never re-evaluated
  as instructions.
- Prompt injection patterns in user messages are detected and rejected.
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from app.core.config import get_settings
from app.memory.types import ExtractionCandidate, MemoryEntry, MemoryType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sensitive data patterns — never extract content matching these
# ---------------------------------------------------------------------------
_SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(api[_\s-]?key|apikey)\s*(?:is|[:=])\s*\S+"),
    re.compile(r"(?i)(password|passwd|pwd)\s*(?:is|[:=])\s*\S+"),
    re.compile(r"(?i)(token|access[_\s-]?token|bearer)\s*(?:is|[:=])\s*\S+"),
    re.compile(r"(?i)(secret|private[_\s-]?key)\s*(?:is|[:=])\s*\S+"),
    re.compile(r"(?i)sk-[A-Za-z0-9]{20,}"),  # OpenAI-style keys
    re.compile(r"(?i)eyJ[A-Za-z0-9_-]{10,}"),  # JWT tokens
    re.compile(r"(?i)(ssh-rsa|-----BEGIN)"),  # SSH / PEM
]

# Prompt injection markers
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)ignore\s+(all\s+)?(previous|prior)\s+instructions"),
    re.compile(r"(?i)reveal\s+(your\s+)?(system\s+)?prompt"),
    re.compile(r"(?i)override\s+(system|instructions)"),
    re.compile(r"(?i)forget\s+everything"),
    re.compile(r"(?i)do\s+not\s+follow"),
    re.compile(r"(?i)jailbreak"),
    re.compile(r"(?i)act\s+as\s+(?:dan|developer|evil|unrestricted)"),
]

# ---------------------------------------------------------------------------
# Rule-based extraction patterns
# ---------------------------------------------------------------------------
_PREFERENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?i)i\s+(?:prefer|like|want|love|use|always\s+use|tend\s+to\s+use)\s+(.{5,80})"
    ),
    re.compile(r"(?i)my\s+(?:preference|favorite|preferred)\s+(?:is|for)\s+(.{5,80})"),
    re.compile(r"(?i)i\s+(?:don't|do\s+not|never)\s+(?:like|use|want)\s+(.{5,80})"),
]

_MODEL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(?:using|switched?\s+to|running|use)\s+(qwen\w*[\s:]\w+)", re.IGNORECASE),
    re.compile(r"(?i)(?:using|switched?\s+to|running|use)\s+(llama[\w\s:-]+)", re.IGNORECASE),
    re.compile(r"(?i)(?:using|switched?\s+to|running|use)\s+(mistral[\w\s:-]+)", re.IGNORECASE),
    re.compile(r"(?i)(?:using|switched?\s+to|running|use)\s+(gemma[\w\s:-]+)", re.IGNORECASE),
    re.compile(r"(?i)(?:using|switched?\s+to|running|use)\s+(phi[\w\s:-]+)", re.IGNORECASE),
    re.compile(r"(?i)my\s+(?:model|llm)\s+is\s+(.{3,40})", re.IGNORECASE),
    re.compile(r"(?i)i\s+(?:use|prefer|chose?|selected?)\s+([\w.:/-]+(?:8b|4b|7b|13b|70b|3\.8b|32b|405b))", re.IGNORECASE),
]

_GOAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(?:i'm|i\s+am)\s+(?:building|creating|developing|working\s+on)\s+(.{5,120})"),
    re.compile(r"(?i)my\s+(?:goal|project|task|aim|objective)\s+is\s+(.{5,120})"),
    re.compile(r"(?i)i\s+(?:want|need|plan)\s+to\s+(?:build|create|develop|implement|add)\s+(.{5,120})"),
]

_TECHNICAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?i)(?:we're|i'm|we\s+are|i\s+am)\s+(?:using|running|on|with)\s+((?:python|node|typescript|react|fastapi|django|postgres|mongodb|redis|docker|kubernetes)[\w\s./]*)",
        re.IGNORECASE,
    ),
    re.compile(r"(?i)(?:tech\s+stack|stack|backend|frontend)\s+is\s+(.{5,120})"),
    re.compile(r"(?i)(?:database|db)\s+is\s+(.{5,60})", re.IGNORECASE),
]

_LOCAL_OPEN_SOURCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(?:prefer|want|use)\s+(?:local|open.?source|self.?host)"),
    re.compile(r"(?i)(?:don't|do\s+not|never)\s+(?:use|want|like)\s+(?:cloud|closed.?source|proprietary)"),
    re.compile(r"(?i)(?:local|open.?source)\s+(?:model|llm|ai)"),
]

_EXPLICIT_FACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(?:remember|note|keep\s+in\s+mind|save)\s+(?:that\s+)?(.{3,120})"),
    re.compile(r"(?i)my\s+(?:name|role|job|title|company|team|timezone|location|working\s+hours)\s+is\s+(.{2,60})"),
    re.compile(r"(?i)(?:i\s+am|i'm)\s+a\s+(.{3,60})"),
    re.compile(r"(?i)(?:i\s+work|we\s+work)\s+(?:at|for|in|as)\s+(.{3,60})"),
]


def _is_sensitive(text: str) -> bool:
    """Return True if text contains sensitive data that must not be stored."""
    return any(p.search(text) for p in _SENSITIVE_PATTERNS)


def _is_injection(text: str) -> bool:
    """Return True if text contains prompt injection patterns."""
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def _sanitize_content(text: str) -> str:
    """Strip leading/trailing whitespace and limit length."""
    return text.strip()[:500]


class MemoryExtractor:
    """Extracts structured long-term memories from conversation turns.

    Usage:
        extractor = MemoryExtractor()
        candidates = extractor.extract(
            user_id=...,
            question="I prefer local open-source models.",
            answer="Understood! I'll keep that in mind.",
            conversation_id=...,
            existing_memories=[...],  # for conflict detection
        )
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._extractor_mode = getattr(settings, "MEMORY_EXTRACTOR", "rule").lower()
        self._enabled = settings.MEMORY_ENABLED and settings.MEMORY_EXTRACTION_ENABLED
        self._min_importance = settings.MEMORY_MIN_IMPORTANCE
        self._similarity_threshold = settings.MEMORY_SIMILARITY_THRESHOLD

    def extract(
        self,
        *,
        user_id: uuid.UUID,
        question: str,
        answer: str,
        conversation_id: uuid.UUID | None = None,
        existing_memories: list[MemoryEntry] | None = None,
    ) -> list[ExtractionCandidate]:
        """Extract memory candidates from a single Q&A turn.

        Args:
            user_id: The user this turn belongs to.
            question: The user's message.
            answer: The assistant's response.
            conversation_id: Source session for provenance.
            existing_memories: Current active memories for conflict detection.

        Returns:
            List of ExtractionCandidate. May be empty if nothing extractable.
        """
        if not self._enabled:
            return []

        # Safety checks — block sensitive or injection-containing content
        if _is_sensitive(question) or _is_sensitive(answer):
            logger.info(
                "[MEMORY EXTRACT] BLOCKED sensitive_content user=%s question=%r",
                user_id,
                question[:40],
            )
            return []

        if _is_injection(question):
            logger.warning(
                "[MEMORY EXTRACT] BLOCKED prompt_injection user=%s question=%r",
                user_id,
                question[:40],
            )
            return []

        if self._extractor_mode == "llm":
            # LLM extraction is handled by the caller via MemoryManager
            # This path is triggered only from rule mode; LLM path goes via manager
            pass

        candidates = self._rule_extract(question, answer)

        # Conflict detection against existing memories
        if existing_memories:
            candidates = self._detect_conflicts(candidates, existing_memories)

        # Filter below minimum importance
        candidates = [c for c in candidates if c.importance >= self._min_importance]

        logger.info(
            "[MEMORY EXTRACT] user=%s question=%r extracted=%d",
            user_id,
            question[:50],
            len(candidates),
        )
        return candidates

    def _rule_extract(self, question: str, answer: str) -> list[ExtractionCandidate]:
        """Rule-based extraction — no LLM calls."""
        candidates: list[ExtractionCandidate] = []
        combined = question  # Prefer extracting from user message

        # 1. Preferences
        for pattern in _PREFERENCE_PATTERNS:
            m = pattern.search(combined)
            if m:
                content = _sanitize_content(f"User preference: {m.group(0).strip()}")
                if not _is_sensitive(content):
                    candidates.append(
                        ExtractionCandidate(
                            memory_type=MemoryType.PREFERENCE,
                            content=content,
                            importance=0.7,
                            confidence=0.8,
                        )
                    )

        # 2. Local/open-source model preference (high-value signal)
        if any(p.search(combined) for p in _LOCAL_OPEN_SOURCE_PATTERNS):
            candidates.append(
                ExtractionCandidate(
                    memory_type=MemoryType.PREFERENCE,
                    content="User prefers local / open-source models over cloud/proprietary options.",
                    importance=0.85,
                    confidence=0.9,
                )
            )

        # 3. Specific model usage
        for pattern in _MODEL_PATTERNS:
            m = pattern.search(combined)
            if m:
                model_name = m.group(1).strip()
                content = _sanitize_content(f"User is using model: {model_name}")
                if not _is_sensitive(content):
                    candidates.append(
                        ExtractionCandidate(
                            memory_type=MemoryType.TECHNICAL_CONTEXT,
                            content=content,
                            importance=0.8,
                            confidence=0.85,
                            metadata={"model": model_name},
                        )
                    )

        # 4. Goals / projects
        for pattern in _GOAL_PATTERNS:
            m = pattern.search(combined)
            if m:
                content = _sanitize_content(f"User goal: {m.group(1).strip()}")
                if not _is_sensitive(content):
                    candidates.append(
                        ExtractionCandidate(
                            memory_type=MemoryType.GOAL,
                            content=content,
                            importance=0.75,
                            confidence=0.8,
                        )
                    )

        # 5. Technical context
        for pattern in _TECHNICAL_PATTERNS:
            m = pattern.search(combined)
            if m:
                content = _sanitize_content(f"Technical context: {m.group(1).strip()}")
                if not _is_sensitive(content):
                    candidates.append(
                        ExtractionCandidate(
                            memory_type=MemoryType.TECHNICAL_CONTEXT,
                            content=content,
                            importance=0.7,
                            confidence=0.75,
                        )
                    )

        # 6. Explicit user profile facts
        for pattern in _EXPLICIT_FACT_PATTERNS:
            m = pattern.search(combined)
            if m:
                content = _sanitize_content(f"User fact: {m.group(0).strip()}")
                if not _is_sensitive(content):
                    candidates.append(
                        ExtractionCandidate(
                            memory_type=MemoryType.USER_PROFILE,
                            content=content,
                            importance=0.85,
                            confidence=0.9,
                        )
                    )

        # Deduplicate similar candidates within this extraction batch
        return self._deduplicate_batch(candidates)

    def _deduplicate_batch(
        self, candidates: list[ExtractionCandidate]
    ) -> list[ExtractionCandidate]:
        """Remove near-duplicate candidates from the same extraction batch."""
        seen_contents: list[str] = []
        deduped: list[ExtractionCandidate] = []
        for c in candidates:
            # Simple word-overlap deduplication within the same batch
            is_dup = False
            for seen in seen_contents:
                words_c = set(c.content.lower().split())
                words_s = set(seen.lower().split())
                if words_c and words_s:
                    overlap = len(words_c & words_s) / max(len(words_c), len(words_s))
                    if overlap > 0.8:
                        is_dup = True
                        break
            if not is_dup:
                deduped.append(c)
                seen_contents.append(c.content)
        return deduped

    def _detect_conflicts(
        self,
        candidates: list[ExtractionCandidate],
        existing_memories: list[MemoryEntry],
    ) -> list[ExtractionCandidate]:
        """Detect conflicts between new candidates and existing memories.

        If a new candidate's content is very similar to an existing memory
        of the same type, set `conflicts_with` so the manager can supersede
        the old memory instead of creating a duplicate.
        """
        updated: list[ExtractionCandidate] = []
        for candidate in candidates:
            conflict_id: uuid.UUID | None = None
            best_overlap = 0.0

            for mem in existing_memories:
                if mem.memory_type != candidate.memory_type:
                    continue
                words_new = set(candidate.content.lower().split())
                words_old = set(mem.content.lower().split())
                if not words_new or not words_old:
                    continue
                overlap = len(words_new & words_old) / max(len(words_new), len(words_old))
                if overlap > self._similarity_threshold and overlap > best_overlap:
                    best_overlap = overlap
                    conflict_id = mem.id

            candidate.conflicts_with = conflict_id
            updated.append(candidate)
        return updated
