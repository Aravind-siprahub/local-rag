"""Automated RAG evaluation runner module."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from app.eval.baseline import compare_baseline, load_baseline_results, save_baseline_results
from app.eval.dataset import EvalDataset, EvalTestCase
from app.eval.formatter import (
    format_baseline_comparison,
    format_per_question_results,
    format_summary_report,
)
from app.eval.metrics import (
    compute_hit_at_k_metrics,
    evaluate_answer_groundedness,
    validate_citations,
    validate_no_answer_refusal,
    validate_version_correctness,
)

DEFAULT_DATASET_PATH = Path(__file__).parent.parent.parent / "eval" / "benchmark_dataset.json"
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000/api/chat")


def run_evaluation_suite(
    *,
    dataset_path: str | Path | None = None,
    api_url: str | None = None,
    output_dir: str | Path | None = None,
    save_baseline_path: str | Path | None = None,
    compare_baseline_path: str | Path | None = None,
    silent: bool = False,
) -> dict[str, Any]:
    """Execute complete evaluation suite against RAG pipeline."""
    ds_path = Path(dataset_path or DEFAULT_DATASET_PATH)
    target_api = api_url or DEFAULT_API_URL
    out_dir = Path(output_dir or ds_path.parent)

    dataset = EvalDataset.load_from_file(ds_path)
    total_cases = len(dataset.test_cases)

    if not silent:
        print("================================================================================")
        print("                   LOCAL RAG EVALUATION SUITE (AUTOMATED)                       ")
        print("================================================================================")
        print(f"Target API Endpoint: {target_api}")
        print(f"Dataset Path: {ds_path}")
        print(f"Total Test Cases: {total_cases}\n")

    # 1. Verify API Connectivity
    health_url = target_api.replace("/api/chat", "/api/health")
    try:
        res = httpx.get(health_url, timeout=5)
        if not silent and res.status_code == 200:
            print(f"[HEALTH CHECK OK] API live: {res.json()}")
    except Exception as err:
        if not silent:
            print(f"[API WARNING] Live API check failed at {target_api}: {err}")

    results: list[dict[str, Any]] = []

    total_hit1 = 0
    total_hit3 = 0
    total_hit5 = 0
    total_hit10 = 0
    total_citation_acc = 0.0
    total_correct_source = 0
    total_grounded = 0
    total_expected_facts = 0.0
    total_unsupported = 0
    total_refusals_correct = 0
    total_negative_cases = 0

    total_retrieval_lat_ms = 0
    total_total_lat_ms = 0
    passed_count = 0

    eval_session_id = os.getenv("EVAL_SESSION_ID", "00000000-0000-0000-0000-000000000000")
    eval_auth_token = os.getenv("EVAL_AUTH_TOKEN")
    if not eval_auth_token:
        token_url = target_api.replace("/api/chat", "/api/auth/token")
        try:
            tok_res = httpx.post(token_url, json={}, timeout=5)
            if tok_res.status_code == 200:
                eval_auth_token = tok_res.json().get("access_token")
        except Exception:
            pass

    eval_headers = {}
    if eval_auth_token:
        eval_headers["Authorization"] = f"Bearer {eval_auth_token}"

    for idx, test_case in enumerate(dataset.test_cases, start=1):
        start_time = time.time()
        actual_answer = ""
        citations: list[dict[str, Any]] = []
        retrieved_docs: list[str] = []
        retrieved_versions: list[str] = []
        retrieved_chunk_ids: list[str] = []
        retrieved_context_texts: list[str] = []
        similarity_scores: list[float] = []

        retrieval_lat_ms = 0
        total_lat_ms = 0

        try:
            resp = httpx.post(
                target_api,
                json={"question": test_case.question, "session_id": eval_session_id},
                headers=eval_headers,
                timeout=180,
            )
            total_lat_ms = int((time.time() - start_time) * 1000)

            if resp.status_code == 200:
                raw_json = resp.json()
                actual_answer = raw_json.get("answer", "")
                citations = raw_json.get("citations", [])
                retrieval_lat_ms = raw_json.get("retrieval_duration_ms", raw_json.get("processing_time_ms", total_lat_ms))
            else:
                actual_answer = f"HTTP {resp.status_code}: {resp.text}"
        except Exception as err:
            total_lat_ms = int((time.time() - start_time) * 1000)
            actual_answer = f"Exception: {err}"

        # Populate retrieved documents/versions/chunks from citations
        for cite in citations:
            doc_name = cite.get("document_title", cite.get("document_name", ""))
            ver_id = str(cite.get("document_version_id", cite.get("version_id", "")))
            c_id = str(cite.get("chunk_id", cite.get("id", "")))
            score = float(cite.get("similarity_score", 0.0))
            txt = cite.get("chunk_text", cite.get("preview", ""))

            if doc_name:
                retrieved_docs.append(doc_name)
            if ver_id:
                retrieved_versions.append(ver_id)
            if c_id:
                retrieved_chunk_ids.append(c_id)
            if txt:
                retrieved_context_texts.append(txt)
            similarity_scores.append(score)

        # Calculate metrics
        hit_metrics = compute_hit_at_k_metrics(retrieved_docs, test_case.expected_document)
        version_ok = validate_version_correctness(retrieved_versions, test_case.expected_version)
        cite_res = validate_citations(citations, [{"chunk_id": c} for c in retrieved_chunk_ids], test_case.expected_document, test_case.expected_version)
        ans_res = evaluate_answer_groundedness(actual_answer, retrieved_context_texts, test_case.expected_key_facts, is_negative=test_case.is_negative)
        refusal_res = validate_no_answer_refusal(actual_answer, test_case.is_negative)

        if hit_metrics["hit@1"]:
            total_hit1 += 1
        if hit_metrics["hit@3"]:
            total_hit3 += 1
        if hit_metrics["hit@5"]:
            total_hit5 += 1
        if hit_metrics["hit@10"]:
            total_hit10 += 1

        total_citation_acc += cite_res["citation_score"]
        if cite_res["cited_document_correct"]:
            total_correct_source += 1

        if ans_res["grounded"]:
            total_grounded += 1
        total_expected_facts += ans_res["expected_facts_rate"]
        if ans_res["unsupported_claims"]:
            total_unsupported += 1

        if test_case.is_negative:
            total_negative_cases += 1
            if refusal_res["correct_refusal"]:
                total_refusals_correct += 1

        total_retrieval_lat_ms += retrieval_lat_ms
        total_total_lat_ms += total_lat_ms

        citation_passed = cite_res["citation_score"] >= 70.0
        answer_passed = ans_res["answer_correctness"] >= 70.0 if not test_case.is_negative else refusal_res["correct_refusal"]
        overall_pass = (hit_metrics["hit@5"] or test_case.is_negative) and citation_passed and answer_passed and version_ok

        issues: list[str] = []
        if not hit_metrics["hit@5"] and not test_case.is_negative:
            issues.append("Expected document missing from top-5 retrieved context")
        if not version_ok:
            issues.append("Retrieved version did not match expected current version")
        if not citation_passed:
            issues.append("Citation validation failed or missing expected source")
        if not answer_passed:
            if test_case.is_negative:
                issues.append("Failed to refuse out-of-context question")
            else:
                issues.append("Answer correctness/groundedness threshold not met")

        if overall_pass:
            passed_count += 1

        res_entry = {
            "id": test_case.id,
            "category": test_case.category,
            "question": test_case.question,
            "expected_source": str(test_case.expected_document),
            "retrieved_source": str(retrieved_docs[:3]),
            "expected_version": str(test_case.expected_version),
            "retrieved_versions": retrieved_versions,
            "hit@1": hit_metrics["hit@1"],
            "hit@3": hit_metrics["hit@3"],
            "hit@5": hit_metrics["hit@5"],
            "hit@10": hit_metrics["hit@10"],
            "citation_passed": citation_passed,
            "answer_passed": answer_passed,
            "citation_score": cite_res["citation_score"],
            "answer_correctness": ans_res["answer_correctness"],
            "is_negative": test_case.is_negative,
            "latency_ms": total_lat_ms,
            "retrieval_latency_ms": retrieval_lat_ms,
            "passed": overall_pass,
            "issues": issues,
            "actual_answer": actual_answer,
        }
        results.append(res_entry)

    # Compute Summary
    summary = {
        "total_cases": total_cases,
        "passed_cases": passed_count,
        "overall_accuracy_pct": round((passed_count / total_cases) * 100.0, 1) if total_cases else 0.0,
        "hit_at_1_pct": round((total_hit1 / total_cases) * 100.0, 1) if total_cases else 0.0,
        "hit_at_3_pct": round((total_hit3 / total_cases) * 100.0, 1) if total_cases else 0.0,
        "hit_at_5_pct": round((total_hit5 / total_cases) * 100.0, 1) if total_cases else 0.0,
        "hit_at_10_pct": round((total_hit10 / total_cases) * 100.0, 1) if total_cases else 0.0,
        "citation_accuracy": round(total_citation_acc / total_cases, 1) if total_cases else 0.0,
        "correct_source_rate": round((total_correct_source / total_cases) * 100.0, 1) if total_cases else 0.0,
        "grounded_answer_rate": round((total_grounded / total_cases) * 100.0, 1) if total_cases else 0.0,
        "expected_facts_rate": round(total_expected_facts / total_cases, 1) if total_cases else 0.0,
        "unsupported_answer_rate": round((total_unsupported / total_cases) * 100.0, 1) if total_cases else 0.0,
        "correct_refusal_rate": round((total_refusals_correct / total_negative_cases) * 100.0, 1) if total_negative_cases else 100.0,
        "avg_retrieval_latency_ms": round(total_retrieval_lat_ms / total_cases, 1) if total_cases else 0.0,
        "avg_total_latency_ms": round(total_total_lat_ms / total_cases, 1) if total_cases else 0.0,
    }

    formatted_summary = format_summary_report(summary)
    per_q_report = format_per_question_results(results)

    if not silent:
        print(formatted_summary)
        print("\n" + per_q_report)

    # Export Report Files
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    csv_path = out_dir / "evaluation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Category", "Question", "Passed", "Hit@1", "Hit@3", "Hit@5", "Hit@10", "CitationPassed", "AnswerPassed", "LatencyMS", "Issues"])
        for r in results:
            writer.writerow([r["id"], r["category"], r["question"], r["passed"], r["hit@1"], r["hit@3"], r["hit@5"], r["hit@10"], r["citation_passed"], r["answer_passed"], r["latency_ms"], "; ".join(r["issues"])])

    md_path = out_dir / "evaluation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Local RAG Evaluation Report\n\n")
        f.write(formatted_summary + "\n\n")
        f.write(per_q_report + "\n")

    if save_baseline_path:
        b_path = save_baseline_results(summary, results, save_baseline_path)
        if not silent:
            print(f"\n[BASELINE SAVED] Saved evaluation baseline to: {b_path}")

    if compare_baseline_path:
        baseline_data = load_baseline_results(compare_baseline_path)
        comparison = compare_baseline({"summary": summary, "results": results}, baseline_data)
        comp_report = format_baseline_comparison(comparison)
        if not silent:
            print("\n" + comp_report)

    return {"summary": summary, "results": results}
