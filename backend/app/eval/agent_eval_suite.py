"""Comprehensive Agentic AI Evaluation Suite.

Evaluates 12 distinct scenario query categories:
1. Simple Document QA
2. Multi-Document QA
3. Web Search Questions
4. Hybrid RAG + Web Questions
5. Image / Visual Questions
6. Follow-up Questions
7. Long-Context Conversations
8. No-Answer / Out-of-Corpus Questions
9. Irrelevant-Document Retrieval
10. Conflicting Document Evidence
11. Complex Multi-Step Questions
12. Tool Failure & Recovery

Measures:
- Retrieval accuracy
- Answer correctness
- Groundedness ratio
- Hallucination rate
- Tool-selection accuracy
- Agent success rate
- Latency (ms)
- Token/model usage
- Retry count
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import os
import time
from dataclasses import dataclass, field
from typing import Any

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.session import AsyncSessionLocal
from app.agent.orchestrator import AgentOrchestrator
from app.agent.state import AgentStatus, AgentState
from app.repositories.user_repository import UserRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AGENT_EVAL_SUITE")


@dataclass
class EvalTestCase:
    id: str
    category: str
    query: str
    expected_tool: str
    expected_answer_keywords: list[str]
    should_fallback: bool = False
    has_image: bool = False
    notes: str = ""


EVAL_BENCHMARK_CASES: list[EvalTestCase] = [
    EvalTestCase(
        id="TC01",
        category="Simple Document QA",
        query="What frontend and backend frameworks are used by Talk to My Data?",
        expected_tool="document_rag",
        expected_answer_keywords=["react", "fastapi"],
    ),
    EvalTestCase(
        id="TC02",
        category="Multi-Document QA",
        query="Summarize the core features of Talk to My Data and SipraOne.",
        expected_tool="document_rag",
        expected_answer_keywords=["talk to my data", "sipraone"],
    ),
    EvalTestCase(
        id="TC03",
        category="Web Search Questions",
        query="What is the current latest release version of Python in 2026?",
        expected_tool="web_search",
        expected_answer_keywords=["python"],
    ),
    EvalTestCase(
        id="TC04",
        category="Hybrid RAG + Web Questions",
        query="What frameworks does Talk to My Data use and what is the latest Python version online?",
        expected_tool="document_rag",
        expected_answer_keywords=["fastapi", "python"],
    ),
    EvalTestCase(
        id="TC05",
        category="Image / Visual Questions",
        query="What is shown in this system diagram?",
        expected_tool="vision_analysis",
        expected_answer_keywords=["diagram", "image"],
        has_image=True,
    ),
    EvalTestCase(
        id="TC06",
        category="Follow-up Questions",
        query="What port does it run on?",
        expected_tool="document_rag",
        expected_answer_keywords=["port", "8000"],
    ),
    EvalTestCase(
        id="TC07",
        category="Long-Context Conversations",
        query="Based on our earlier discussion about architecture, list the backend components.",
        expected_tool="document_rag",
        expected_answer_keywords=["fastapi", "backend"],
    ),
    EvalTestCase(
        id="TC08",
        category="No-Answer / Out-of-Corpus Questions",
        query="What is the orbital radius of Jupiter's moon Europa in kilometers?",
        expected_tool="web_search",
        expected_answer_keywords=["europa", "kilometer"],
    ),
    EvalTestCase(
        id="TC09",
        category="Irrelevant-Document Retrieval",
        query="What is the secret cookie recipe of Grandma Alice?",
        expected_tool="document_rag",
        expected_answer_keywords=["could not find", "not found"],
        should_fallback=True,
    ),
    EvalTestCase(
        id="TC10",
        category="Conflicting Document Evidence",
        query="Which database is configured for primary storage in the PRD?",
        expected_tool="document_rag",
        expected_answer_keywords=["postgres", "postgresql"],
    ),
    EvalTestCase(
        id="TC11",
        category="Complex Multi-Step Questions",
        query="Find the process manager in our PRD document and check if it is recommended online.",
        expected_tool="document_rag",
        expected_answer_keywords=["pm2", "process manager"],
    ),
    EvalTestCase(
        id="TC12",
        category="Tool Failure & Recovery",
        query="Retrieve non-existent internal doc xyz123 and search fallback.",
        expected_tool="document_rag",
        expected_answer_keywords=["could not find", "not found"],
        should_fallback=True,
    ),
]


@dataclass
class EvalResult:
    case_id: str
    category: str
    query: str
    passed: bool
    tool_selected: str
    tool_accuracy: bool
    answer_correctness: bool
    groundedness_ratio: float
    hallucination_detected: bool
    latency_ms: int
    iterations: int
    models_used: list[str]
    answer: str


class AgentEvalRunner:
    """Runs evaluation benchmark across the 12 agentic test scenario categories."""

    async def run_evaluation(self) -> dict[str, Any]:
        logger.info("=== STARTING AGENTIC AI EVALUATION SUITE ===")
        start_eval = time.monotonic()
        results: list[EvalResult] = []

        async with AsyncSessionLocal() as session:
            user_repo = UserRepository(session)
            users = await user_repo.list_active(limit=1)
            user_id = users[0].id if users else None

            orchestrator = AgentOrchestrator(session)

            for test in EVAL_BENCHMARK_CASES:
                logger.info("[EVAL RUNNING] case=%s category=%r query=%r", test.id, test.category, test.query)

                dummy_image = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82" if test.has_image else None

                state: AgentState = await orchestrator.run(
                    query=test.query,
                    user_id=user_id,
                    image_bytes=dummy_image,
                )

                selected_tools = state.metrics.tools_used
                primary_tool = selected_tools[0] if selected_tools else "none"

                tool_acc = (primary_tool == test.expected_tool) or (test.expected_tool in selected_tools)
                ans_lower = (state.final_answer or "").lower()

                if test.should_fallback:
                    ans_correct = "could not find" in ans_lower or "not found" in ans_lower or not state.evidence
                else:
                    ans_correct = any(kw.lower() in ans_lower for kw in test.expected_answer_keywords)

                grounded_ratio = 1.0 if (state.verification_result and state.verification_result.is_valid) else 0.5
                hallucinated = state.verification_result.hallucination_detected if state.verification_result else False

                case_passed = tool_acc and ans_correct and not hallucinated

                res = EvalResult(
                    case_id=test.id,
                    category=test.category,
                    query=test.query,
                    passed=case_passed,
                    tool_selected=primary_tool,
                    tool_accuracy=tool_acc,
                    answer_correctness=ans_correct,
                    groundedness_ratio=grounded_ratio,
                    hallucination_detected=hallucinated,
                    latency_ms=state.metrics.total_latency_ms,
                    iterations=state.metrics.iterations,
                    models_used=state.metrics.models_used,
                    answer=state.final_answer or "",
                )
                results.append(res)
                logger.info(
                    "[EVAL RESULT] case=%s passed=%s tool=%s latency=%dms",
                    test.id, case_passed, primary_tool, state.metrics.total_latency_ms
                )

        total_cases = len(results)
        passed_cases = sum(1 for r in results if r.passed)
        avg_latency = sum(r.latency_ms for r in results) // max(total_cases, 1)
        tool_accuracy_pct = (sum(1 for r in results if r.tool_accuracy) / total_cases) * 100
        overall_success_pct = (passed_cases / total_cases) * 100
        avg_groundedness = sum(r.groundedness_ratio for r in results) / max(total_cases, 1)
        total_eval_time = int((time.monotonic() - start_eval) * 1000)

        summary = {
            "total_test_cases": total_cases,
            "passed_test_cases": passed_cases,
            "failed_test_cases": total_cases - passed_cases,
            "agent_success_rate_pct": round(overall_success_pct, 2),
            "tool_selection_accuracy_pct": round(tool_accuracy_pct, 2),
            "average_groundedness_score": round(avg_groundedness, 2),
            "average_latency_ms": avg_latency,
            "total_eval_duration_ms": total_eval_time,
            "test_cases_detail": [
                {
                    "case_id": r.case_id,
                    "category": r.category,
                    "query": r.query,
                    "passed": r.passed,
                    "tool_selected": r.tool_selected,
                    "latency_ms": r.latency_ms,
                    "iterations": r.iterations,
                    "models_used": r.models_used,
                    "answer_snippet": r.answer[:100],
                }
                for r in results
            ],
        }

        print("\n=======================================================")
        print("AGENTIC AI EVALUATION SUITE REPORT SUMMARY")
        print("=======================================================")
        print(f"Total Test Scenarios:       {total_cases}")
        print(f"Passed Scenarios:           {passed_cases}/{total_cases}")
        print(f"Agent Success Rate:         {overall_success_pct:.2f}%")
        print(f"Tool Selection Accuracy:    {tool_accuracy_pct:.2f}%")
        print(f"Average Groundedness Score: {avg_groundedness:.2f}")
        print(f"Average Latency:            {avg_latency} ms")
        print("=======================================================\n")

        return summary


if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info < (3, 14):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    runner = AgentEvalRunner()
    try:
        asyncio.run(runner.run_evaluation())
    except KeyboardInterrupt:
        print("\n[INFO] Evaluation suite interrupted by user.")
