"""Automated unit test suite for RAG evaluation system components."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.baseline import compare_baseline, load_baseline_results, save_baseline_results
from app.eval.dataset import EvalDataset, EvalTestCase
from app.eval.formatter import (
    format_baseline_comparison,
    format_per_question_results,
    format_summary_report,
)
from app.eval.metrics import (
    calculate_hit_at_k,
    compute_hit_at_k_metrics,
    evaluate_answer_groundedness,
    validate_citations,
    validate_no_answer_refusal,
    validate_version_correctness,
)


def test_dataset_loading(tmp_path: Path) -> None:
    """Test loading and parsing of evaluation dataset fixtures."""
    data = [
        {
            "id": 1,
            "category": "direct_factual",
            "question": "What is Talk to My Data?",
            "expected_documents": ["PRD_Talk_to_My_Data.docx"],
            "expected_version": "v1.0",
            "required_keywords": ["RAG", "Text-to-SQL"],
        },
        {
            "id": 2,
            "category": "negative questions",
            "question": "Who directed Inception?",
            "expected_answer": "Information not found.",
            "is_negative": True,
        },
    ]
    file_path = tmp_path / "test_dataset.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    ds = EvalDataset.load_from_file(file_path)
    assert len(ds.test_cases) == 2

    tc1 = ds.test_cases[0]
    assert tc1.id == 1
    assert tc1.question == "What is Talk to My Data?"
    assert tc1.expected_document == ["PRD_Talk_to_My_Data.docx"]
    assert tc1.expected_version == "v1.0"
    assert "RAG" in tc1.expected_key_facts
    assert not tc1.is_negative

    tc2 = ds.test_cases[1]
    assert tc2.id == 2
    assert tc2.is_negative


def test_hit_at_k_calculation() -> None:
    """Test Hit@1, Hit@3, Hit@5, Hit@10 retrieval metrics."""
    retrieved = ["docA.pdf", "docB.docx", "docC.pdf", "docD.txt", "docE.md"]
    expected = "docC.pdf"

    assert not calculate_hit_at_k(retrieved, expected, k=1)
    assert not calculate_hit_at_k(retrieved, expected, k=2)
    assert calculate_hit_at_k(retrieved, expected, k=3)
    assert calculate_hit_at_k(retrieved, expected, k=5)
    assert calculate_hit_at_k(retrieved, expected, k=10)

    metrics = compute_hit_at_k_metrics(retrieved, expected)
    assert metrics["hit@1"] is False
    assert metrics["hit@3"] is True
    assert metrics["hit@5"] is True
    assert metrics["hit@10"] is True


def test_citation_validation() -> None:
    """Test citation accuracy, document matching, and context verification."""
    retrieved_chunks = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
    citations = [
        {
            "chunk_id": "c1",
            "document_title": "PRD_Talk_to_My_Data.docx",
            "document_version_id": "v2.0",
        }
    ]
    expected_doc = "PRD_Talk_to_My_Data.docx"
    expected_ver = "v2.0"

    res = validate_citations(citations, retrieved_chunks, expected_doc, expected_ver)
    assert res["citation_exists"] is True
    assert res["cited_document_correct"] is True
    assert res["cited_version_correct"] is True
    assert res["cited_chunk_in_retrieved_context"] is True
    assert res["no_unrelated_citations"] is True
    assert res["citation_score"] == 100.0


def test_version_validation() -> None:
    """Test document version correctness validation."""
    retrieved_versions = ["v1.0", "v2.0"]
    assert validate_version_correctness(retrieved_versions, "v2.0") is True
    assert validate_version_correctness(retrieved_versions, "v3.0") is False
    assert validate_version_correctness(retrieved_versions, None) is True


def test_no_answer_detection() -> None:
    """Test refusal detection for out-of-context / negative questions."""
    refusal_answer = "Information not found in document excerpts."
    invented_answer = "Inception was directed by Christopher Nolan in 2010."

    res_refusal = validate_no_answer_refusal(refusal_answer, is_negative=True)
    assert res_refusal["has_refusal"] is True
    assert res_refusal["correct_refusal"] is True

    res_invented = validate_no_answer_refusal(invented_answer, is_negative=True)
    assert res_invented["has_refusal"] is False
    assert res_invented["correct_refusal"] is False


def test_answer_groundedness() -> None:
    """Test answer groundedness and expected facts matching."""
    answer = "Talk to My Data uses RAG and Text-to-SQL for enterprise search."
    context = ["Talk to My Data provides RAG and Text-to-SQL capabilities."]
    facts = ["RAG", "Text-to-SQL"]

    res = evaluate_answer_groundedness(answer, context, facts, is_negative=False)
    assert res["grounded"] is True
    assert res["expected_facts_rate"] == 100.0
    assert res["unsupported_claims"] is False
    assert res["answer_correctness"] == 100.0


def test_baseline_comparison(tmp_path: Path) -> None:
    """Test saving baseline, loading baseline, and computing comparison diffs."""
    baseline_summary = {
        "hit_at_1_pct": 80.0,
        "hit_at_3_pct": 90.0,
        "grounded_answer_rate": 85.0,
        "avg_retrieval_latency_ms": 150.0,
        "overall_accuracy_pct": 85.0,
    }
    baseline_results = [
        {"id": 1, "question": "Q1", "passed": True},
        {"id": 2, "question": "Q2", "passed": True},
    ]
    b_file = tmp_path / "baseline.json"
    save_baseline_results(baseline_summary, baseline_results, b_file)

    loaded = load_baseline_results(b_file)
    assert loaded["summary"]["hit_at_1_pct"] == 80.0

    current_run = {
        "summary": {
            "hit_at_1_pct": 85.0,  # Improved
            "hit_at_3_pct": 90.0,  # Unchanged
            "grounded_answer_rate": 80.0,  # Degraded
            "avg_retrieval_latency_ms": 120.0,  # Improved (lower latency)
            "overall_accuracy_pct": 80.0,
        },
        "results": [
            {"id": 1, "question": "Q1", "passed": True},
            {"id": 2, "question": "Q2", "passed": False, "issues": ["Retrieval failed"]},
        ],
    }

    comp = compare_baseline(current_run, loaded)

    improved_names = [item["metric"] for item in comp["improved_metrics"]]
    assert "Hit@1" in improved_names
    assert "Average Retrieval Latency (ms)" in improved_names

    degraded_names = [item["metric"] for item in comp["degraded_metrics"]]
    assert "Grounded Answer Rate" in degraded_names

    assert len(comp["newly_failing_questions"]) == 1
    assert comp["newly_failing_questions"][0]["id"] == "2"


def test_evaluation_formatting() -> None:
    """Test summary report formatting matches required structure."""
    summary = {
        "hit_at_1_pct": 90.0,
        "hit_at_3_pct": 95.0,
        "hit_at_5_pct": 100.0,
        "hit_at_10_pct": 100.0,
        "citation_accuracy": 95.0,
        "correct_source_rate": 90.0,
        "grounded_answer_rate": 98.0,
        "expected_facts_rate": 95.0,
        "unsupported_answer_rate": 2.0,
        "correct_refusal_rate": 100.0,
        "avg_retrieval_latency_ms": 45.2,
        "avg_total_latency_ms": 320.5,
    }
    report = format_summary_report(summary)

    assert "## Retrieval" in report
    assert "Hit@1: 90.0%" in report
    assert "Hit@5: 100.0%" in report
    assert "## Citation" in report
    assert "Citation accuracy: 95.0%" in report
    assert "## Answer" in report
    assert "Grounded answer rate: 98.0%" in report
    assert "## No-answer" in report
    assert "Correct refusal rate: 100.0%" in report
    assert "## Performance" in report
    assert "Average retrieval latency: 45.2 ms" in report
