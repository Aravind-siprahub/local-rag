"""Baseline saving, loading, and comparison framework."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_baseline_results(summary: dict[str, Any], results: list[dict[str, Any]], filepath: str | Path) -> Path:
    """Save baseline evaluation results to JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": summary,
        "results": results,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return path


def load_baseline_results(filepath: str | Path) -> dict[str, Any]:
    """Load baseline evaluation results from JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Baseline file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_baseline(current_run: dict[str, Any], baseline_run: dict[str, Any]) -> dict[str, Any]:
    """Compare current evaluation run against baseline evaluation run."""
    curr_sum = current_run.get("summary", {})
    base_sum = baseline_run.get("summary", {})

    curr_res = {str(r.get("id")): r for r in current_run.get("results", [])}
    base_res = {str(r.get("id")): r for r in baseline_run.get("results", [])}

    improved_metrics: list[dict[str, Any]] = []
    degraded_metrics: list[dict[str, Any]] = []
    unchanged_metrics: list[dict[str, Any]] = []

    # Map of metric keys to evaluate (higher is better for most, lower is better for latency / unsupported)
    higher_is_better_keys = {
        "hit_at_1_pct": "Hit@1",
        "hit_at_3_pct": "Hit@3",
        "hit_at_5_pct": "Hit@5",
        "hit_at_10_pct": "Hit@10",
        "citation_accuracy": "Citation Accuracy",
        "correct_source_rate": "Correct Source Rate",
        "grounded_answer_rate": "Grounded Answer Rate",
        "expected_facts_rate": "Expected Facts Rate",
        "correct_refusal_rate": "Correct Refusal Rate",
        "overall_accuracy_pct": "Overall Accuracy",
    }

    lower_is_better_keys = {
        "unsupported_answer_rate": "Unsupported Answer Rate",
        "avg_retrieval_latency_ms": "Average Retrieval Latency (ms)",
        "avg_total_latency_ms": "Average Total Latency (ms)",
    }

    for key, label in higher_is_better_keys.items():
        c_val = float(curr_sum.get(key, 0.0))
        b_val = float(base_sum.get(key, 0.0))
        diff = round(c_val - b_val, 2)

        entry = {"metric": label, "baseline": b_val, "current": c_val, "diff": diff}
        if diff > 0.01:
            improved_metrics.append(entry)
        elif diff < -0.01:
            degraded_metrics.append(entry)
        else:
            unchanged_metrics.append({"metric": label, "value": c_val})

    for key, label in lower_is_better_keys.items():
        c_val = float(curr_sum.get(key, 0.0))
        b_val = float(base_sum.get(key, 0.0))
        diff = round(c_val - b_val, 2)

        entry = {"metric": label, "baseline": b_val, "current": c_val, "diff": diff}
        if diff < -0.01:
            improved_metrics.append(entry)
        elif diff > 0.01:
            degraded_metrics.append(entry)
        else:
            unchanged_metrics.append({"metric": label, "value": c_val})

    # Find newly failing questions
    newly_failing: list[dict[str, Any]] = []
    for q_id, c_item in curr_res.items():
        b_item = base_res.get(q_id)
        c_pass = c_item.get("passed", False)
        b_pass = b_item.get("passed", True) if b_item else True

        if b_pass and not c_pass:
            newly_failing.append({
                "id": q_id,
                "question": c_item.get("question", ""),
                "baseline_status": "PASS" if b_pass else "FAIL",
                "current_status": "FAIL",
                "issues": c_item.get("issues", []),
            })

    return {
        "improved_metrics": improved_metrics,
        "degraded_metrics": degraded_metrics,
        "unchanged_metrics": unchanged_metrics,
        "newly_failing_questions": newly_failing,
    }
