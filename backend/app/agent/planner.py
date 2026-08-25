"""Planning and intent understanding engine for task decomposition and tool selection."""
from __future__ import annotations

import logging
from typing import Sequence

from app.agent.state import AgentState, PlanStep
from app.rag.intent_router import classify, Route
from app.rag.query_understanding import extract_query_intent, QueryIntent

logger = logging.getLogger(__name__)


class Planner:
    """Decomposes user queries into structured execution plans and selects tools."""

    async def create_plan(
        self,
        state: AgentState,
        *,
        document_titles: Sequence[str] | None = None,
        has_image: bool = False,
        has_doc_filter: bool = False,
    ) -> list[PlanStep]:
        """Analyze query and attachments to build an execution plan."""
        query = state.user_query.strip()
        intent = extract_query_intent(query)
        state.intent = intent.category.name if hasattr(intent, "category") else "GENERAL"

        route = classify(query, document_titles=document_titles)
        plan: list[PlanStep] = []

        if has_image:
            plan.append(
                PlanStep(
                    step_number=1,
                    description="Analyze visual contents of the uploaded image/screenshot/chart.",
                    target_tool="vision_analysis",
                    expected_outcome="Extracted visual text, numerical data, and structural observations.",
                )
            )

        if route in (Route.DOCUMENT_QA, Route.RAG) or has_doc_filter or _has_document_keywords(query):
            step_num = len(plan) + 1
            plan.append(
                PlanStep(
                    step_number=step_num,
                    description="Perform hybrid vector retrieval over uploaded project documents.",
                    target_tool="document_rag",
                    expected_outcome="Relevant document chunks and evidence statements.",
                )
            )

        elif route == Route.WEB or _is_explicit_web_query(query):
            step_num = len(plan) + 1
            plan.append(
                PlanStep(
                    step_number=step_num,
                    description="Search the live web for current real-time or external facts.",
                    target_tool="web_search",
                    expected_outcome="Current web search hits and web evidence.",
                )
            )

        if not plan:
            # Default to Document RAG, with Web Search fallback option
            plan.append(
                PlanStep(
                    step_number=1,
                    description="Retrieve knowledge from project documents.",
                    target_tool="document_rag",
                    expected_outcome="Verified knowledge context.",
                )
            )

        state.plan = plan
        state.current_step_index = 0
        logger.info(
            "[PLANNER SUCCESS] query=%r route=%s steps_count=%d steps=%s",
            query, route.value, len(plan), [s.target_tool for s in plan]
        )
        return plan


def _has_document_keywords(query: str) -> bool:
    q_low = query.lower()
    return any(
        kw in q_low
        for kw in (
            "document", "doc", "file", "prd", "pdf", "policy", "architecture",
            "frontend", "backend", "framework", "database", "stack", "deployment",
            "talk to my data", "siprahub", "airis", "process manager", "port",
        )
    )


def _is_explicit_web_query(query: str) -> bool:
    q_low = query.lower()
    return any(
        kw in q_low
        for kw in (
            "search web", "web search", "search online", "latest version", "weather",
            "current date", "news", "today", "google", "online", "latest news",
            "current price", "current prices", "recent release", "recent releases",
            "current documentation", "event", "events", "price of", "latest",
            "recent", "release date", "stock price", "current info", "live search"
        )
    )
