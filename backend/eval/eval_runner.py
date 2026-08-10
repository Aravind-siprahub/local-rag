"""Automated RAG Evaluation Runner Framework.

Executes benchmark test suite against live API, evaluates quality metrics,
and generates evaluation_report.html, evaluation_report.md, evaluation_results.csv,
and evaluation_results.json.
"""
import csv
import json
import os
import sys
import time
from datetime import datetime
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000/api/chat")
BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "benchmark_dataset.json")
OUTPUT_DIR = os.getenv("EVAL_OUTPUT_DIR", os.path.dirname(__file__))

def compute_ngram_overlap(text1: str, text2: str, n: int = 5) -> float:
    """Compute n-gram overlap fraction to detect verbatim copying."""
    words1 = [w.lower() for w in text1.split() if len(w) > 2]
    words2 = [w.lower() for w in text2.split() if len(w) > 2]

    if len(words1) < n or len(words2) < n:
        return 0.0

    ngrams1 = set(tuple(words1[i:i+n]) for i in range(len(words1)-n+1))
    ngrams2 = set(tuple(words2[i:i+n]) for i in range(len(words2)-n+1))

    if not ngrams1:
        return 0.0

    intersection = ngrams1.intersection(ngrams2)
    return len(intersection) / len(ngrams1)

def run_evaluation():
    print(f"================================================================================")
    print(f"            AUTOMATED RAG EVALUATION RUNNER (PRINCIPAL AI ENGINEER)             ")
    print(f"================================================================================")
    print(f"Target API Endpoint: {API_URL}")
    print(f"Benchmark File: {BENCHMARK_PATH}")
    print(f"Output Directory: {OUTPUT_DIR}\n")

    # 1. Verify API Connectivity
    health_url = API_URL.replace("/api/chat", "/api/health")
    try:
        health_res = requests.get(health_url, timeout=5)
        if health_res.status_code == 200:
            print(f"[HEALTH CHECK OK] API is live and responding: {health_res.json()}")
        else:
            print(f"[HEALTH CHECK WARNING] API returned status {health_res.status_code}")
    except Exception as err:
        print(f"[CRITICAL ERROR] Live API endpoint is unavailable at {API_URL}: {err}")
        print("Aborting evaluation as instructed: No simulated or mocked results allowed.")
        sys.exit(1)

    # 2. Load Benchmark Dataset
    if not os.path.exists(BENCHMARK_PATH):
        print(f"[CRITICAL ERROR] Benchmark dataset not found at {BENCHMARK_PATH}")
        sys.exit(1)

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    print(f"\nLoaded {len(benchmark)} benchmark test cases across categories.\n")

    results = []
    total_tests = len(benchmark)
    passed_count = 0
    failed_count = 0

    total_latency_ms = 0
    total_grounding_score = 0
    total_correctness_score = 0
    total_citation_score = 0
    total_completeness_score = 0

    reasoning_leakage_count = 0
    verbatim_copy_count = 0

    category_stats = {}

    for idx, test_case in enumerate(benchmark, start=1):
        q_id = test_case.get("id", idx)
        category = test_case.get("category", "general")
        question = test_case["question"]
        expected_answer = test_case["expected_answer"]
        req_keywords = test_case.get("required_keywords", [])
        forbid_keywords = test_case.get("forbidden_keywords", [])
        exp_docs = test_case.get("expected_documents", [])

        if category not in category_stats:
            category_stats[category] = {"total": 0, "passed": 0, "failed": 0}
        category_stats[category]["total"] += 1

        print(f"[{idx}/{total_tests}] [{category.upper()}] Q: {question[:65]}...")

        eval_session_id = os.getenv("EVAL_SESSION_ID", "00000000-0000-0000-0000-000000000000")
        start_time = time.time()
        try:
            resp = requests.post(
                API_URL,
                json={"question": question, "session_id": eval_session_id},
                timeout=180,
            )
            latency_ms = int((time.time() - start_time) * 1000)
            total_latency_ms += latency_ms

            if resp.status_code == 200:
                raw_json = resp.json()
                actual_answer = raw_json.get("answer", "")
                citations = raw_json.get("citations", [])
                processing_time_ms = raw_json.get("processing_time_ms", latency_ms)
            else:
                raw_json = {"error": resp.text}
                actual_answer = f"HTTP {resp.status_code}: {resp.text}"
                citations = []
                processing_time_ms = latency_ms
        except Exception as err:
            latency_ms = int((time.time() - start_time) * 1000)
            raw_json = {"exception": str(err)}
            actual_answer = f"Exception: {err}"
            citations = []
            processing_time_ms = latency_ms

        # AUTOMATED SCORING METRICS
        # 1. Reasoning Leakage Check
        has_leakage = False
        lower_ans = actual_answer.lower()
        if "<think>" in lower_ans or "</think>" in lower_ans or "let me analyze" in lower_ans or "looking at passage" in lower_ans or "passage 1 is" in lower_ans:
            has_leakage = True
            reasoning_leakage_count += 1

        for kw in forbid_keywords:
            if kw.lower() in lower_ans:
                has_leakage = True

        # 2. Verbatim Copying Check
        verbatim_overlap = 0.0
        for cite in citations:
            snippet = cite.get("preview", "")
            if snippet:
                overlap = compute_ngram_overlap(actual_answer, snippet, n=5)
                if overlap > verbatim_overlap:
                    verbatim_overlap = overlap

        is_verbatim = verbatim_overlap > 0.40
        if is_verbatim:
            verbatim_copy_count += 1

        # 3. Grounding & Refusal Check
        if category == "negative questions":
            is_refusal = "information not found" in lower_ans
            grounding_score = 100.0 if is_refusal else 0.0
        else:
            grounding_score = 100.0 if len(citations) > 0 or "information not found" not in lower_ans else 50.0

        # 4. Keyword Completeness & Correctness
        matched_keywords = [kw for kw in req_keywords if kw.lower() in lower_ans]
        completeness_ratio = len(matched_keywords) / len(req_keywords) if req_keywords else 1.0
        completeness_score = round(completeness_ratio * 100.0, 1)

        # 5. Citation Accuracy
        if exp_docs:
            cited_docs = [c.get("document_title", "") for c in citations]
            doc_matches = [doc for doc in exp_docs if any(doc.lower() in cd.lower() for cd in cited_docs)]
            citation_score = round((len(doc_matches) / len(exp_docs)) * 100.0, 1)
        else:
            citation_score = 100.0 if len(citations) == 0 else 80.0

        # 6. Overall Correctness Score
        correctness_score = 100.0
        issues = []
        failure_domain = "none"

        if category == "negative questions" and "information not found" not in lower_ans:
            correctness_score -= 50.0
            issues.append("Failed strict negative question refusal constraint")
            failure_domain = "prompt"

        if has_leakage:
            correctness_score -= 30.0
            issues.append("Reasoning leakage detected in response")
            failure_domain = "sanitization"

        if is_verbatim:
            correctness_score -= 20.0
            issues.append(f"Verbatim copying threshold exceeded ({verbatim_overlap:.1%})")
            failure_domain = "prompt"

        if completeness_ratio < 0.5:
            correctness_score -= 30.0
            issues.append(f"Low keyword completeness ({len(matched_keywords)}/{len(req_keywords)} keywords)")
            if failure_domain == "none":
                failure_domain = "LLM"

        if exp_docs and citation_score < 50.0:
            correctness_score -= 20.0
            issues.append("Expected document citations missing")
            if failure_domain == "none":
                failure_domain = "retrieval"

        correctness_score = max(0.0, correctness_score)
        passed = correctness_score >= 70.0

        if passed:
            passed_count += 1
            category_stats[category]["passed"] += 1
        else:
            failed_count += 1
            category_stats[category]["failed"] += 1

        total_grounding_score += grounding_score
        total_correctness_score += correctness_score
        total_citation_score += citation_score
        total_completeness_score += completeness_score

        root_cause = "Passed verification"
        if not passed:
            if failure_domain == "retrieval":
                root_cause = "Vector similarity search failed to retrieve target document chunks."
            elif failure_domain == "prompt":
                root_cause = "Prompt constraints violated (refusal or verbatim copying threshold)."
            elif failure_domain == "sanitization":
                root_cause = "Output sanitizer failed to strip reasoning scratchpad tags."
            elif failure_domain == "LLM":
                root_cause = "LLM completion omitted required factual keywords."

        result_entry = {
            "id": q_id,
            "category": category,
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "citations": citations,
            "latency_ms": latency_ms,
            "processing_time_ms": processing_time_ms,
            "correctness_score": correctness_score,
            "grounding_score": grounding_score,
            "citation_score": citation_score,
            "completeness_score": completeness_score,
            "has_reasoning_leakage": has_leakage,
            "is_verbatim_copy": is_verbatim,
            "verbatim_overlap": round(verbatim_overlap, 3),
            "passed": passed,
            "issues": issues,
            "failure_domain": failure_domain,
            "root_cause": root_cause,
            "raw_response": raw_json
        }
        results.append(result_entry)

    # 3. Compute Final Summary Metrics
    avg_latency = round(total_latency_ms / total_tests, 1)
    accuracy_pct = round((passed_count / total_tests) * 100.0, 1)
    avg_grounding_pct = round(total_grounding_score / total_tests, 1)
    avg_correctness_pct = round(total_correctness_score / total_tests, 1)
    avg_citation_pct = round(total_citation_score / total_tests, 1)
    avg_completeness_pct = round(total_completeness_score / total_tests, 1)
    leakage_pct = round((reasoning_leakage_count / total_tests) * 100.0, 1)
    verbatim_pct = round((verbatim_copy_count / total_tests) * 100.0, 1)

    print("\n================================================================================")
    print("                    EVALUATION RUN COMPLETE - SUMMARY                           ")
    print("================================================================================")
    print(f"Total Tests Executed: {total_tests}")
    print(f"Passed: {passed_count} | Failed: {failed_count}")
    print(f"Accuracy: {accuracy_pct}%")
    print(f"Average Latency: {avg_latency} ms ({avg_latency/1000:.2f} s)")
    print(f"Grounding Score: {avg_grounding_pct}%")
    print(f"Citation Accuracy: {avg_citation_pct}%")
    print(f"Reasoning Leakage Rate: {leakage_pct}%")
    print(f"Verbatim Copying Rate: {verbatim_pct}%")
    print("================================================================================\n")

    # 4. Generate JSON Report
    json_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed_count": passed_count,
                "failed_count": failed_count,
                "accuracy_pct": accuracy_pct,
                "avg_latency_ms": avg_latency,
                "avg_grounding_pct": avg_grounding_pct,
                "avg_correctness_pct": avg_correctness_pct,
                "avg_citation_pct": avg_citation_pct,
                "avg_completeness_pct": avg_completeness_pct,
                "reasoning_leakage_pct": leakage_pct,
                "verbatim_copy_pct": verbatim_pct,
                "production_readiness_score": accuracy_pct
            },
            "category_breakdown": category_stats,
            "results": results
        }, f, indent=2)
    print(f"Generated JSON: {json_path}")

    # 5. Generate CSV Report
    csv_path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Category", "Question", "Passed", "Correctness", "Grounding", "CitationScore", "LatencyMS", "FailureDomain", "RootCause"])
        for r in results:
            writer.writerow([r["id"], r["category"], r["question"], r["passed"], r["correctness_score"], r["grounding_score"], r["citation_score"], r["latency_ms"], r["failure_domain"], r["root_cause"]])
    print(f"Generated CSV: {csv_path}")

    # 6. Generate Markdown Report
    md_path = os.path.join(OUTPUT_DIR, "evaluation_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Enterprise RAG Automated Evaluation Report\n\n")
        f.write(f"**Execution Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Summary Metrics\n\n")
        f.write(f"| Metric | Value | Target |\n|---|---|---|\n")
        f.write(f"| Total Benchmark Tests | **{total_tests}** | 100 |\n")
        f.write(f"| Overall Accuracy % | **{accuracy_pct}%** | > 95% |\n")
        f.write(f"| Grounding % | **{avg_grounding_pct}%** | 100% |\n")
        f.write(f"| Citation Accuracy % | **{avg_citation_pct}%** | > 95% |\n")
        f.write(f"| Reasoning Leakage % | **{leakage_pct}%** | 0.0% |\n")
        f.write(f"| Verbatim Copying % | **{verbatim_pct}%** | < 5.0% |\n")
        f.write(f"| Average Latency | **{avg_latency} ms** ({avg_latency/1000:.2f} s) | < 3000ms (GPU) |\n")
        f.write(f"| Production Readiness Score | **{accuracy_pct}%** | > 90% |\n\n")

        f.write(f"## Category Breakdown\n\n")
        f.write(f"| Category | Total | Passed | Failed | Pass Rate |\n|---|---|---|---|---|\n")
        for cat, stats in category_stats.items():
            crate = round((stats['passed']/stats['total'])*100.0, 1)
            f.write(f"| {cat.title()} | {stats['total']} | {stats['passed']} | {stats['failed']} | {crate}% |\n")

        f.write(f"\n## Failed Tests Analysis\n\n")
        failed_tests = [r for r in results if not r["passed"]]
        if not failed_tests:
            f.write("All 100 benchmark test cases passed successfully!\n")
        else:
            for ft in failed_tests:
                f.write(f"### Q{ft['id']}: {ft['question']}\n\n")
                f.write(f"- **Category**: {ft['category']}\n")
                f.write(f"- **Expected Answer**: {ft['expected_answer']}\n")
                f.write(f"- **Actual Answer**: {ft['actual_answer']}\n")
                f.write(f"- **Failure Domain**: `{ft['failure_domain']}`\n")
                f.write(f"- **Root Cause**: {ft['root_cause']}\n")
                f.write(f"- **Issues**: {', '.join(ft['issues'])}\n\n")

    print(f"Generated Markdown: {md_path}")

    # 7. Generate HTML Report
    html_path = os.path.join(OUTPUT_DIR, "evaluation_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Enterprise RAG Automated Evaluation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #0f172a; color: #f8fafc; }}
        h1, h2, h3 {{ color: #38bdf8; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 32px; }}
        .metric-card {{ background: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155; }}
        .metric-val {{ font-size: 28px; font-weight: bold; color: #4ade80; margin-top: 8px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 16px; background: #1e293b; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #334155; color: #f8fafc; }}
        .badge-pass {{ background: #166534; color: #4ade80; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
        .badge-fail {{ background: #991b1b; color: #fca5a5; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>Enterprise RAG Automated Evaluation Report</h1>
    <p>Executed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} against live API ({API_URL})</p>

    <div class="metrics-grid">
        <div class="metric-card"><div>Overall Accuracy</div><div class="metric-val">{accuracy_pct}%</div></div>
        <div class="metric-card"><div>Grounding Score</div><div class="metric-val">{avg_grounding_pct}%</div></div>
        <div class="metric-card"><div>Citation Accuracy</div><div class="metric-val">{avg_citation_pct}%</div></div>
        <div class="metric-card"><div>Avg Latency</div><div class="metric-val">{avg_latency} ms</div></div>
    </div>

    <h2>Category Breakdown</h2>
    <table>
        <thead><tr><th>Category</th><th>Total</th><th>Passed</th><th>Failed</th><th>Pass Rate</th></tr></thead>
        <tbody>
""")
        for cat, stats in category_stats.items():
            crate = round((stats['passed']/stats['total'])*100.0, 1)
            f.write(f"<tr><td>{cat.title()}</td><td>{stats['total']}</td><td>{stats['passed']}</td><td>{stats['failed']}</td><td>{crate}%</td></tr>")

        f.write("""
        </tbody>
    </table>
</body>
</html>
""")
    print(f"Generated HTML: {html_path}\n")

if __name__ == "__main__":
    run_evaluation()
