"""Core evaluation metrics: Hit@K, citations, groundedness, refusal, and versioning."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


REFUSAL_PHRASES = [
    "information not found",
    "insufficient information",
    "cannot find",
    "not mentioned",
    "no information available",
    "does not mention",
    "does not contain",
    "not present in the document",
    "not found in the provided",
]


def _normalize_str(val: str | None) -> str:
    return (val or "").strip().lower()


def calculate_hit_at_k(
    retrieved_docs: list[str],
    expected_docs: str | list[str] | None,
    k: int,
) -> bool:
    """Return True if at least one expected document is in retrieved_docs[:k]."""
    if not expected_docs:
        return True

    exp_list = [expected_docs] if isinstance(expected_docs, str) else expected_docs
    exp_set = {_normalize_str(d) for d in exp_list if d}
    if not exp_set:
        return True

    k_slice = retrieved_docs[:k]
    for doc in k_slice:
        norm_doc = _normalize_str(doc)
        if any(e in norm_doc or norm_doc in e for e in exp_set):
            return True
    return False


def compute_hit_at_k_metrics(
    retrieved_docs: list[str],
    expected_docs: str | list[str] | None,
) -> dict[str, bool]:
    """Calculate Hit@1, Hit@3, Hit@5, Hit@10 for a single test query."""
    return {
        "hit@1": calculate_hit_at_k(retrieved_docs, expected_docs, 1),
        "hit@3": calculate_hit_at_k(retrieved_docs, expected_docs, 3),
        "hit@5": calculate_hit_at_k(retrieved_docs, expected_docs, 5),
        "hit@10": calculate_hit_at_k(retrieved_docs, expected_docs, 10),
    }


def validate_version_correctness(
    retrieved_versions: list[str],
    expected_version: str | list[str] | None,
) -> bool:
    """Validate whether retrieved versions contain the expected version and avoid outdated versions."""
    if not expected_version:
        return True

    exp_list = [expected_version] if isinstance(expected_version, str) else expected_version
    exp_set = {_normalize_str(v) for v in exp_list if v}

    norm_retrieved = {_normalize_str(v) for v in retrieved_versions if v}
    if not norm_retrieved:
        return False

    return any(e in norm_r or norm_r in e for e in exp_set for norm_r in norm_retrieved)


def validate_citations(
    citations: list[dict[str, Any]],
    retrieved_chunks: list[dict[str, Any]],
    expected_docs: str | list[str] | None,
    expected_version: str | list[str] | None = None,
) -> dict[str, Any]:
    """Verify generated citations against retrieved context and expected source targets."""
    has_citations = len(citations) > 0
    exp_list = [expected_docs] if isinstance(expected_docs, str) else (expected_docs or [])
    exp_set = {_normalize_str(d) for d in exp_list if d}

    retrieved_chunk_ids = {str(c.get("chunk_id", c.get("id", ""))) for c in retrieved_chunks if c}

    cited_doc_correct = True
    cited_version_correct = True
    cited_chunks_in_context = True
    unrelated_citations = False

    if has_citations:
        for cite in citations:
            doc_name = _normalize_str(cite.get("document_title", cite.get("document_name", "")))
            ver_id = _normalize_str(str(cite.get("document_version_id", cite.get("version_id", ""))))
            c_id = str(cite.get("chunk_id", cite.get("id", "")))

            if exp_set and not any(e in doc_name or doc_name in e for e in exp_set):
                cited_doc_correct = False
                unrelated_citations = True

            if expected_version:
                exp_vers = [expected_version] if isinstance(expected_version, str) else expected_version
                exp_ver_set = {_normalize_str(v) for v in exp_vers}
                if not any(v in ver_id or ver_id in v for v in exp_ver_set):
                    cited_version_correct = False

            if retrieved_chunk_ids and c_id and c_id not in retrieved_chunk_ids:
                cited_chunks_in_context = False
    else:
        if exp_set:
            cited_doc_correct = False

    score = 100.0
    if not has_citations and exp_set:
        score = 0.0
    elif not cited_doc_correct:
        score -= 40.0
    elif not cited_version_correct:
        score -= 30.0
    elif not cited_chunks_in_context:
        score -= 20.0

    return {
        "citation_exists": has_citations,
        "cited_document_correct": cited_doc_correct,
        "cited_version_correct": cited_version_correct,
        "cited_chunk_in_retrieved_context": cited_chunks_in_context,
        "no_unrelated_citations": not unrelated_citations,
        "citation_score": max(0.0, score),
    }


def validate_no_answer_refusal(answer: str, is_negative: bool) -> dict[str, Any]:
    """Check refusal correctness for out-of-context / negative questions."""
    norm_answer = _normalize_str(answer)
    has_refusal = any(phrase in norm_answer for phrase in REFUSAL_PHRASES)

    if is_negative:
        correct_refusal = has_refusal
    else:
        correct_refusal = not has_refusal

    return {
        "has_refusal": has_refusal,
        "correct_refusal": correct_refusal,
        "refusal_score": 100.0 if correct_refusal else 0.0,
    }


def evaluate_answer_groundedness(
    answer: str,
    retrieved_context_texts: list[str],
    expected_key_facts: list[str],
    is_negative: bool = False,
) -> dict[str, Any]:
    """Evaluate whether answer is grounded in context and contains expected key facts."""
    if is_negative:
        refusal_res = validate_no_answer_refusal(answer, is_negative=True)
        return {
            "grounded": refusal_res["has_refusal"],
            "expected_facts_rate": 100.0 if refusal_res["has_refusal"] else 0.0,
            "unsupported_claims": not refusal_res["has_refusal"],
            "answer_correctness": 100.0 if refusal_res["has_refusal"] else 0.0,
        }

    norm_answer = _normalize_str(answer)
    combined_context = _normalize_str(" ".join(retrieved_context_texts))

    # Facts matching ratio
    matched_facts = 0
    total_facts = len(expected_key_facts)
    if total_facts > 0:
        for fact in expected_key_facts:
            norm_fact = _normalize_str(fact)
            if norm_fact in norm_answer or any(word in norm_answer for word in norm_fact.split() if len(word) > 4):
                matched_facts += 1
        facts_rate = (matched_facts / total_facts) * 100.0
    else:
        facts_rate = 100.0

    # Groundedness: check if key answer terms appear in context or if answer is empty/refusal
    grounded = True
    unsupported_claims = False
    if combined_context and norm_answer:
        cleaned_words = [w.strip(".,!?;:\"'()[]{}") for w in norm_answer.split()]
        words = [w for w in cleaned_words if len(w) >= 3 and any(c.isalnum() for c in w)]
        if words:
            in_context = sum(1 for w in words if w in combined_context)
            ratio = in_context / len(words)
            if ratio < 0.3:
                grounded = False
                unsupported_claims = True

    # Composite correctness score
    correctness = facts_rate * 0.7 + (100.0 if grounded else 0.0) * 0.3

    return {
        "grounded": grounded,
        "expected_facts_rate": round(facts_rate, 1),
        "unsupported_claims": unsupported_claims,
        "answer_correctness": round(correctness, 1),
    }


@dataclass
class LLMJudgeResult:
    """Recorded result from an optional LLM judge evaluation."""

    evaluated: bool = False
    judge_model: str = "none"
    groundedness_score: float = 0.0
    correctness_score: float = 0.0
    reasoning: str = ""
    criteria_used: str = ""
