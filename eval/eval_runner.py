"""Automated RAG Evaluation Runner Framework.

Executes benchmark test suite against live API, evaluates quality metrics across
multiple providers (Ollama, OpenRouter, NVIDIA), and generates evaluation reports
including comparison summaries.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import requests

# Ensure backend directory is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000/api/chat")
BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "benchmark_dataset.json")
DEFAULT_OUTPUT_DIR = os.getenv("EVAL_OUTPUT_DIR", os.path.dirname(__file__))



def compute_ngram_overlap(text1: str, text2: str, n: int = 5) -> float:
    """Compute n-gram overlap fraction to detect verbatim copying."""
    words1 = [w.lower() for w in text1.split() if len(w) > 2]
    words2 = [w.lower() for w in text2.split() if len(w) > 2]

    if len(words1) < n or len(words2) < n:
        return 0.0

    ngrams1 = set(tuple(words1[i:i + n]) for i in range(len(words1) - n + 1))
    ngrams2 = set(tuple(words2[i:i + n]) for i in range(len(words2) - n + 1))

    if not ngrams1:
        return 0.0

    intersection = ngrams1.intersection(ngrams2)
    return len(intersection) / len(ngrams1)


def generate_comparison_summary(output_dir: str):
    """Generate human-readable markdown comparison summary across benchmark runs."""
    pattern = os.path.join(output_dir, "*_evaluation_results.json")
    json_files = glob.glob(pattern)
    if not json_files:
        main_json = os.path.join(output_dir, "evaluation_results.json")
        if os.path.exists(main_json):
            json_files = [main_json]

    if not json_files:
        return

    runs = []
    for filepath in json_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "provider" in data and "model" in data:
                    runs.append(data)
        except Exception:
            continue

    if not runs:
        return

    summary_md_path = os.path.join(output_dir, "model_comparison_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Sipra Local RAG - Model Comparison Summary Report\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| Provider | Model | Questions | Success | Failed | Avg Latency (ms) | Avg TTFT (ms) | Tokens/sec | Total Tokens | Est. Cost |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")

        for r in runs:
            prov = r.get("provider", "unknown")
            mdl = r.get("model", "unknown")
            q_cnt = r.get("questions", 0)
            succ = r.get("successful", 0)
            fail = r.get("failed", 0)
            lat = f"{r.get('avg_latency_ms'):.1f}" if r.get("avg_latency_ms") is not None else "N/A"
            ttft = f"{r.get('avg_ttft_ms'):.1f}" if r.get("avg_ttft_ms") is not None else "N/A"
            tps = f"{r.get('avg_tokens_per_second'):.1f}" if r.get("avg_tokens_per_second") is not None else "N/A"
            tot_tok = (r.get("total_prompt_tokens", 0) or 0) + (r.get("total_completion_tokens", 0) or 0)
            cost_val = r.get("estimated_cost")
            cost_str = f"${cost_val:.4f}" if isinstance(cost_val, (int, float)) else "cost: unavailable"

            f.write(f"| **{prov}** | `{mdl}` | {q_cnt} | {succ} | {fail} | {lat} | {ttft} | {tps} | {tot_tok} | {cost_str} |\n")

        f.write("\n\n---\n*Note: All benchmark evaluations run against identical Sipra RAG retrieval context and question set.*\n")
    print(f"Generated Comparison Summary: {summary_md_path}")


def run_evaluation(
    provider: str = "ollama",
    model: str | None = None,
    api_url: str | None = None,
    output_dir: str | None = None,
):
    target_api = api_url or DEFAULT_API_URL
    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    target_provider = provider.lower()
    target_model = model or ("qwen3:8b" if target_provider == "ollama" else "default")

    print("================================================================================")
    print("            AUTOMATED RAG EVALUATION RUNNER (SIPRA LOCAL RAG)                  ")
    print("================================================================================")
    print(f"Target API Endpoint: {target_api}")
    print(f"LLM Provider:        {target_provider}")
    print(f"LLM Model:           {target_model}")
    print(f"Benchmark Dataset:   {BENCHMARK_PATH}")
    print(f"Output Directory:    {out_dir}\n")

    # 1. Verify API Connectivity
    health_url = target_api.replace("/api/chat", "/api/health")
    try:
        health_res = requests.get(health_url, timeout=5)
        if health_res.status_code == 200:
            print(f"[HEALTH CHECK OK] API is live and responding: {health_res.json()}")
        else:
            print(f"[HEALTH CHECK WARNING] API returned status {health_res.status_code}")
    except Exception as err:
        print(f"[CRITICAL ERROR] Live API endpoint is unavailable at {target_api}: {err}")
        print("Aborting evaluation as instructed: No simulated or mocked results allowed.")
        sys.exit(1)

    # 2. Load Benchmark Dataset
    if not os.path.exists(BENCHMARK_PATH):
        print(f"[CRITICAL ERROR] Benchmark dataset not found at {BENCHMARK_PATH}")
        sys.exit(1)

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    total_tests = len(benchmark)
    print(f"\nLoaded {total_tests} benchmark test cases across categories.\n")

    results = []
    passed_count = 0
    failed_count = 0

    total_latency_ms = 0
    total_ttft_ms = 0
    ttft_samples = 0
    total_tps = 0.0
    tps_samples = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost_usd = 0.0
    has_cost = False

    total_grounding_score = 0
    total_correctness_score = 0
    total_citation_score = 0
    total_completeness_score = 0
    reasoning_leakage_count = 0
    verbatim_copy_count = 0

    category_stats = {}

    # 1.5 Fetch Evaluation Authentication Token
    auth_headers = {}
    eval_auth_token = os.getenv("EVAL_AUTH_TOKEN")
    if not eval_auth_token:
        demo_token_url = target_api.replace("/api/chat", "/api/auth/demo-token")
        try:
            tok_res = requests.post(demo_token_url, json={}, timeout=5)
            if tok_res.status_code == 200:
                eval_auth_token = tok_res.json().get("access_token")
                print(f"[AUTH OK] Successfully issued JWT access token for evaluation.")
        except Exception as tok_err:
            print(f"[AUTH WARNING] Could not fetch demo token: {tok_err}")

    if eval_auth_token:
        auth_headers["Authorization"] = f"Bearer {eval_auth_token}"

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

        start_time = time.time()
        payload = {
            "question": question,
            "provider": target_provider,
        }
        if model:
            payload["model"] = model

        try:
            resp = requests.post(target_api, json=payload, headers=auth_headers, timeout=180)

            latency_ms = int((time.time() - start_time) * 1000)
            total_latency_ms += latency_ms

            if resp.status_code == 200:
                raw_json = resp.json()
                actual_answer = raw_json.get("answer", "")
                citations = raw_json.get("citations", [])
                processing_time_ms = raw_json.get("processing_time_ms", latency_ms)
                returned_model = raw_json.get("model", target_model)

                token_usage = raw_json.get("token_usage") or {}
                p_tok = token_usage.get("prompt_tokens")
                c_tok = token_usage.get("completion_tokens")
                if p_tok is not None:
                    total_prompt_tokens += p_tok
                if c_tok is not None:
                    total_completion_tokens += c_tok
                    if processing_time_ms > 0:
                        tps = c_tok / (processing_time_ms / 1000.0)
                        total_tps += tps
                        tps_samples += 1

            else:
                raw_json = {"error": resp.text}
                actual_answer = f"HTTP {resp.status_code}: {resp.text}"
                citations = []
                processing_time_ms = latency_ms
                returned_model = target_model
        except Exception as err:
            latency_ms = int((time.time() - start_time) * 1000)
            raw_json = {"exception": str(err)}
            actual_answer = f"Exception: {err}"
            citations = []
            processing_time_ms = latency_ms
            returned_model = target_model

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
            snippet = cite.get("preview", "") or cite.get("chunk_text", "")
            if snippet:
                overlap = compute_ngram_overlap(actual_answer, snippet, n=5)
                if overlap > verbatim_overlap:
                    verbatim_overlap = overlap

        is_verbatim = verbatim_overlap > 0.40
        if is_verbatim:
            verbatim_copy_count += 1

        # 3. Grounding & Refusal Check
        if category == "negative questions":
            is_refusal = "information not found" in lower_ans or "not mentioned" in lower_ans
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

        if category == "negative questions" and ("information not found" not in lower_ans and "not mentioned" not in lower_ans):
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
        }
        results.append(result_entry)

    # Compute Summary
    avg_latency = round(total_latency_ms / total_tests, 1) if total_tests else 0.0
    avg_ttft = round(total_ttft_ms / ttft_samples, 1) if ttft_samples else None
    avg_tps = round(total_tps / tps_samples, 1) if tps_samples else None
    accuracy_pct = round((passed_count / total_tests) * 100.0, 1) if total_tests else 0.0
    avg_grounding_pct = round(total_grounding_score / total_tests, 1) if total_tests else 0.0
    avg_citation_pct = round(total_citation_score / total_tests, 1) if total_tests else 0.0
    leakage_pct = round((reasoning_leakage_count / total_tests) * 100.0, 1) if total_tests else 0.0
    verbatim_pct = round((verbatim_copy_count / total_tests) * 100.0, 1) if total_tests else 0.0

    print("\n================================================================================")
    print("                    EVALUATION RUN COMPLETE - SUMMARY                           ")
    print("================================================================================")
    print(f"Provider:              {target_provider}")
    print(f"Model:                 {target_model}")
    print(f"Total Tests Executed:  {total_tests}")
    print(f"Passed: {passed_count} | Failed: {failed_count}")
    print(f"Accuracy:              {accuracy_pct}%")
    print(f"Average Latency:       {avg_latency} ms ({avg_latency/1000:.2f} s)")
    print(f"Average TTFT:          {avg_ttft if avg_ttft is not None else 'N/A'}")
    print(f"Average Tokens/sec:    {avg_tps if avg_tps is not None else 'N/A'}")
    print(f"Total Tokens:          {total_prompt_tokens + total_completion_tokens} (prompt={total_prompt_tokens}, completion={total_completion_tokens})")
    print(f"Estimated Cost:        ${total_cost_usd:.4f}" if has_cost else "Estimated Cost:        cost: unavailable")
    print("================================================================Threshold\n")

    os.makedirs(out_dir, exist_ok=True)
    filename_prefix = f"{target_provider}_{target_model.replace('/', '_').replace(':', '_')}"

    # Machine-readable output JSON with exact required schema
    json_data = {
        "provider": target_provider,
        "model": target_model,
        "questions": total_tests,
        "successful": passed_count,
        "failed": failed_count,
        "avg_ttft_ms": avg_ttft,
        "avg_latency_ms": avg_latency,
        "avg_tokens_per_second": avg_tps,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "estimated_cost": round(total_cost_usd, 4) if has_cost else None,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "accuracy_pct": accuracy_pct,
            "avg_grounding_pct": avg_grounding_pct,
            "avg_citation_pct": avg_citation_pct,
            "reasoning_leakage_pct": leakage_pct,
            "verbatim_copy_pct": verbatim_pct,
        },
        "category_breakdown": category_stats,
        "results": results,
    }

    # Write both model-specific JSON and main evaluation_results.json
    model_json_path = os.path.join(out_dir, f"{filename_prefix}_evaluation_results.json")
    with open(model_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    main_json_path = os.path.join(out_dir, "evaluation_results.json")
    with open(main_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"Generated JSON: {model_json_path}")

    # Generate CSV Report
    csv_path = os.path.join(out_dir, f"{filename_prefix}_evaluation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Category", "Question", "Passed", "Correctness", "Grounding", "CitationScore", "LatencyMS", "FailureDomain", "RootCause"])
        for r in results:
            writer.writerow([r["id"], r["category"], r["question"], r["passed"], r["correctness_score"], r["grounding_score"], r["citation_score"], r["latency_ms"], r["failure_domain"], r["root_cause"]])
    print(f"Generated CSV: {csv_path}")

    # Generate Comparison Report
    generate_comparison_summary(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sipra Local RAG Model Evaluation Runner")
    parser.add_argument(
        "--provider",
        choices=["ollama", "openrouter", "nvidia"],
        default="ollama",
        help="LLM provider to evaluate (default: ollama)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Target model identifier (e.g. 'qwen3:8b', 'meta-llama/llama-3.3-70b-instruct', 'nvidia/nemotron-4-340b-instruct')",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help="API chat endpoint URL",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save evaluation reports",
    )

    args = parser.parse_args()
    run_evaluation(
        provider=args.provider,
        model=args.model,
        api_url=args.api_url,
        output_dir=args.output_dir,
    )
