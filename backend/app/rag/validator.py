"""Post-generation answer validation and grounding reconciliation layer.

Verifies generated answers against retrieved document chunks to ensure:
1. Multi-part questions address all requested sub-topics.
2. Unsupported requested sub-topics explicitly state that the document does not specify them.
3. Unsupported general claims or hallucinated policies are detected and reconciled.
4. Source fidelity is preserved and model-hallucinated sections are removed.
"""
from __future__ import annotations

import logging
import re
from typing import Sequence

from app.rag.query_understanding import decompose_query_topics
from app.retrieval.ranking import RankedResult

logger = logging.getLogger(__name__)

# Standard missing information phrase required by production policy
NOT_SPECIFIED_PHRASE = "I couldn't find enough information in the available documents to answer this question."
DOCUMENT_NOT_SPECIFIED_PHRASE = "The provided document does not specify this information."


def topic_has_evidence(topic: str, context_text: str) -> bool:
    """Check whether a requested topic has supporting evidence in retrieved document chunks."""
    if not topic or not topic.strip() or not context_text:
        return False

    t_clean = topic.strip().lower()
    c_lower = context_text.lower()

    # Exact phrase check
    if t_clean in c_lower:
        return True

    # Substantive keyword check (strip stopwords and generic conversational verbs)
    stopwords = {
        "the", "a", "an", "our", "their", "its", "rules", "rule", "policy",
        "policies", "guidelines", "guideline", "details", "detail", "process",
        "about", "what", "how", "when", "why", "which", "available", "types",
        "and", "or", "for", "in", "of", "to", "at",
        "is", "are", "was", "were", "do", "does", "did", "have", "has", "had",
        "using", "used", "use", "uses", "stack", "stacks", "tech", "matrix", "selection",
    }
    words = [w for w in re.findall(r"\b[a-z0-9]+\b", t_clean) if w not in stopwords]
    if not words:
        words = [w for w in re.findall(r"\b[a-z0-9]+\b", t_clean) if len(w) >= 3]

    # Semantic equivalents for conceptual/cultural topics
    conceptual_equivalents = {
        "core values": ("code of conduct", "conduct", "principles", "commitments", "ethics", "standards", "values", "culture", "integrity", "accountability"),
        "values": ("code of conduct", "conduct", "principles", "commitments", "ethics", "standards", "values", "culture", "integrity", "accountability"),
        "company values": ("code of conduct", "conduct", "principles", "commitments", "ethics", "standards", "values", "culture", "integrity", "accountability"),
        "our values": ("code of conduct", "conduct", "principles", "commitments", "ethics", "standards", "values", "culture", "integrity", "accountability"),
    }
    for concept_phrase, equivs in conceptual_equivalents.items():
        if concept_phrase in t_clean:
            if any(eq in c_lower for eq in equivs):
                return True

    # If all substantive words are found in context
    if all(w in c_lower for w in words):
        return True

    # For multi-word topics, if any strong key noun is missing, evidence is missing
    # (e.g. for "sick leave", if "sick" is missing from context, topic is unsupported)
    for w in words:
        if len(w) >= 3 and w not in c_lower:
            return False

    return True


def topic_is_acknowledged_as_missing(topic: str, answer_text: str) -> bool:
    """Check if the answer already explicitly states that the document does not specify this topic."""
    if not topic or not answer_text:
        return False

    ans_lower = answer_text.lower()
    t_clean = topic.strip().lower()
    t_substantive = " ".join(
        w for w in re.findall(r"\b[a-z0-9]+\b", t_clean)
        if w not in {"the", "a", "an", "our", "rules", "policy", "policies"}
    ) or t_clean

    # Check for general not specified fallback
    if (
        "not specify this information" in ans_lower
        or "information not found in document" in ans_lower
        or "couldn't find enough information" in ans_lower
        or "could not find enough information" in ans_lower
        or "not specify a dedicated core values" in ans_lower
        or "does not specify a dedicated core values" in ans_lower
        or "not specify" in ans_lower
    ):
        return True

    # Check for specific missing declarations
    missing_patterns = [
        rf"(?:does not|doesn't)\s+(?:specify|contain|mention|state|cover|provide)\s+.*?(?:{re.escape(t_substantive)}|this\b)",
        rf"(?:no|not)\s+(?:separate\s+)?{re.escape(t_substantive)}\s+(?:policy|rule|is\s+specified|mentioned)",
        rf"{re.escape(t_substantive)}.*?(?:not specified|does not specify|not mentioned|not covered|not provided)",
        rf"not\s+(?:explicitly\s+)?(?:specified|mentioned|stated|covered)\s+in\s+the\s+provided\s+document",
    ]
    return any(bool(re.search(p, ans_lower)) for p in missing_patterns)


def validate_and_reconcile_answer(
    question: str,
    answer: str,
    context_chunks: Sequence[RankedResult],
) -> str:
    """Validate generated answer against retrieved context chunks and reconcile unsupported topics.

    Returns:
        Grounding-reconciled final answer.
    """
    if not answer or not answer.strip():
        return NOT_SPECIFIED_PHRASE

    ans_clean = answer.strip()
    if not context_chunks:
        # If no context chunks exist, enforce strict fallback
        return NOT_SPECIFIED_PHRASE

    context_text = " ".join(
        f"{c.document_title or ''} {c.section_title or ''} {c.chunk_text or ''}"
        for c in context_chunks
    ).lower()

    # Decompose question into requested sub-topics
    topics = decompose_query_topics(question)
    if not topics:
        topics = [question.strip()]

    supported_topics: list[str] = []
    unsupported_topics: list[str] = []

    for t in topics:
        if topic_has_evidence(t, context_text):
            supported_topics.append(t)
        else:
            unsupported_topics.append(t)

    logger.info(
        "[ANSWER VALIDATOR] question=%r topics=%s supported=%s unsupported=%s",
        question.strip(),
        topics,
        supported_topics,
        unsupported_topics,
    )

    # 1. Case 1: ALL requested topics are unsupported in the retrieved document
    if not supported_topics and unsupported_topics:
        # Check if the answer already correctly declared not specified
        if topic_is_acknowledged_as_missing(unsupported_topics[0], ans_clean):
            return ans_clean

        # Special case: If asked about Core Values / Company Values
        q_lower = question.lower()
        if any(kw in q_lower for kw in ("core values", "company values", "our values", "values")):
            has_conduct = any(kw in context_text for kw in ("code of conduct", "workplace commitments", "conduct", "integrity", "accountability", "standards"))
            if has_conduct:
                if any(kw in ans_clean.lower() for kw in ("code of conduct", "workplace commitments", "conduct", "integrity", "accountability")):
                    return (
                        "The provided document does not specify a dedicated Core Values section. "
                        "However, it outlines the following standards of behavior and commitments in its Code of Conduct:\n\n"
                        + re.sub(r"(?is)^#*\s*(?:core\s+values|core\s+values\s+and\s+principles).*?\n+", "", ans_clean).strip()
                    )
                return (
                    "The provided document does not specify a dedicated Core Values section. "
                    "However, it outlines workplace commitments and standards of behavior in its Code of Conduct."
                )
            return "The provided document does not specify a dedicated Core Values section."

        logger.warning(
            "[ANSWER VALIDATOR] Detected hallucination for completely unsupported query. Overriding with standard fallback."
        )
        # Use specific topic phrasing when available
        if len(unsupported_topics) == 1:
            clean_t = unsupported_topics[0].strip()
            clean_t = re.sub(r"^(?:the|a|an)\s+", "", clean_t, flags=re.IGNORECASE).strip()
            t_name = clean_t.title()
            return f"The provided document does not specify {t_name}."
        return DOCUMENT_NOT_SPECIFIED_PHRASE


    # 2. Case 2: PARTIALLY SUPPORTED question (some topics supported, some unsupported)
    if supported_topics and unsupported_topics:
        reconciled = ans_clean
        missing_declarations: list[str] = []

        for unsup_topic in unsupported_topics:
            clean_topic = unsup_topic.strip()
            if not topic_is_acknowledged_as_missing(clean_topic, reconciled):
                topic_title = clean_topic.title()
                # Check if model fabricated a section for this unsupported topic
                section_pattern = rf"(?is)(?:###?\s*[*_]*{re.escape(clean_topic)}[*_]*.*?(?=(?:###?|\Z))|\*\*{re.escape(clean_topic)}\*\*.*?(?=(?:\*\*|\Z)))"
                if re.search(section_pattern, reconciled):
                    # Replace fabricated section with not specified statement
                    logger.warning(
                        "[ANSWER VALIDATOR] Removing fabricated section for unsupported topic: %s",
                        clean_topic,
                    )
                    reconciled = re.sub(
                        section_pattern,
                        f"**{topic_title}**:\nThe provided document does not specify a separate {clean_topic} policy.\n\n",
                        reconciled,
                    )
                else:
                    missing_declarations.append(
                        f"**{topic_title}**:\nThe provided document does not specify a separate {clean_topic} policy."
                    )

        if missing_declarations:
            reconciled = reconciled.rstrip() + "\n\n" + "\n\n".join(missing_declarations)

        return reconciled.strip()

    # 3. Case 3: Dedicated check for "Core Values" when document only has Code of Conduct
    q_lower = question.lower()
    if any(kw in q_lower for kw in ("core values", "company values")) and "core value" not in context_text and "core values" not in context_text:
        if not topic_is_acknowledged_as_missing("core values", ans_clean):
            # Model synthesized Code of Conduct as Core Values; prepend or note clarification
            if any(kw in ans_clean.lower() for kw in ("code of conduct", "workplace commitments", "conduct", "integrity", "accountability")):
                ans_clean = (
                    "The provided document does not specify a dedicated Core Values section. "
                    "However, it outlines the following standards of behavior and commitments in its Code of Conduct:\n\n"
                    + re.sub(r"(?is)^#*\s*(?:core\s+values|core\s+values\s+and\s+principles).*?\n+", "", ans_clean).strip()
                )

    return ans_clean
