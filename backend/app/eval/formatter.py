"""Report formatting for CLI, Markdown, and JSON evaluation outputs."""
from __future__ import annotations

from typing import Any


def format_summary_report(summary: dict[str, Any]) -> str:
    """Format evaluation summary report matching exact required layout."""
    lines = [
        "## Retrieval",
        f"Hit@1: {summary.get('hit_at_1_pct', 0.0):.1f}%",
        f"Hit@3: {summary.get('hit_at_3_pct', 0.0):.1f}%",
        f"Hit@5: {summary.get('hit_at_5_pct', 0.0):.1f}%",
        f"Hit@10: {summary.get('hit_at_10_pct', 0.0):.1f}%",
        "",
        "## Citation",
        f"Citation accuracy: {summary.get('citation_accuracy', 0.0):.1f}%",
        f"Correct source rate: {summary.get('correct_source_rate', 0.0):.1f}%",
        "",
        "## Answer",
        f"Grounded answer rate: {summary.get('grounded_answer_rate', 0.0):.1f}%",
        f"Expected facts rate: {summary.get('expected_facts_rate', 0.0):.1f}%",
        f"Unsupported answer rate: {summary.get('unsupported_answer_rate', 0.0):.1f}%",
        "",
        "## No-answer",
        f"Correct refusal rate: {summary.get('correct_refusal_rate', 0.0):.1f}%",
        "",
        "## Performance",
        f"Average retrieval latency: {summary.get('avg_retrieval_latency_ms', 0.0):.1f} ms",
        f"Average total latency: {summary.get('avg_total_latency_ms', 0.0):.1f} ms",
    ]
    return "\n".join(lines)


def format_per_question_results(results: list[dict[str, Any]]) -> str:
    """Format per-question evaluation results into readable text output."""
    lines = ["## PER-QUESTION RESULTS\n"]
    for idx, r in enumerate(results, start=1):
        status = "PASS" if r.get("passed", False) else "FAIL"
        hit_k_str = f"Hit@1: {r.get('hit@1')}, Hit@3: {r.get('hit@3')}, Hit@5: {r.get('hit@5')}"
        lines.append(
            f"[{idx}] Q: {r.get('question')}\n"
            f"    Status: {status}\n"
            f"    Expected Source: {r.get('expected_source')}\n"
            f"    Retrieved Source: {r.get('retrieved_source')}\n"
            f"    Hit@K: {hit_k_str}\n"
            f"    Citation Result: {'PASS' if r.get('citation_passed') else 'FAIL'}\n"
            f"    Answer Result: {'PASS' if r.get('answer_passed') else 'FAIL'}\n"
            f"    Latency: {r.get('latency_ms')} ms (Retrieval: {r.get('retrieval_latency_ms', 0)} ms)"
        )
        if r.get("issues"):
            lines.append(f"    Issues: {', '.join(r.get('issues', []))}")
        lines.append("")
    return "\n".join(lines)


def format_baseline_comparison(comparison: dict[str, Any]) -> str:
    """Format BASELINE vs CURRENT comparison report."""
    lines = [
        "================================================================================",
        "                       BASELINE vs CURRENT COMPARISON                           ",
        "================================================================================",
        "",
        "### Improved Metrics",
    ]
    improved = comparison.get("improved_metrics", [])
    if improved:
        for item in improved:
            lines.append(f"  [+] {item['metric']}: Baseline {item['baseline']} -> Current {item['current']} (Diff: +{item['diff']})")
    else:
        lines.append("  (None)")

    lines.append("\n### Degraded Metrics")
    degraded = comparison.get("degraded_metrics", [])
    if degraded:
        for item in degraded:
            lines.append(f"  [-] {item['metric']}: Baseline {item['baseline']} -> Current {item['current']} (Diff: {item['diff']})")
    else:
        lines.append("  (None)")

    lines.append("\n### Unchanged Metrics")
    unchanged = comparison.get("unchanged_metrics", [])
    if unchanged:
        for item in unchanged:
            lines.append(f"  [=] {item['metric']}: {item['value']}")
    else:
        lines.append("  (None)")

    lines.append("\n### Newly Failing Questions")
    newly_failing = comparison.get("newly_failing_questions", [])
    if newly_failing:
        for item in newly_failing:
            lines.append(f"  [!] Q{item['id']}: {item['question']}")
            lines.append(f"      Baseline: {item['baseline_status']} | Current: {item['current_status']}")
            if item.get("issues"):
                lines.append(f"      Issues: {', '.join(item['issues'])}")
    else:
        lines.append("  (None)")

    lines.append("\n================================================================================")
    return "\n".join(lines)
