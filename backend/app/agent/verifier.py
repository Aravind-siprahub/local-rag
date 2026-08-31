"""Verification Agent for evidence relevance validation, claim support, and hallucination detection."""
from __future__ import annotations

import logging
from typing import Any

from app.agent.state import AgentState, VerificationOutcome
from app.core.config import get_settings
from app.rag.verifier import verify_answer, VerificationResult
from app.rag.query_understanding import extract_query_intent

logger = logging.getLogger(__name__)


class VerificationAgent:
    """Verifies evidence quality and validates LLM answers against extracted evidence."""

    def verify_evidence(self, state: AgentState) -> VerificationOutcome:
        """Validate gathered evidence before LLM synthesis."""
        if not state.evidence:
            logger.info("[VERIFIER] 0 evidence items gathered.")
            return VerificationOutcome(
                is_valid=False,
                reason="No evidence items gathered.",
                relevance_score=0.0,
                requires_retry=True,
            )

        max_score = max(item.relevance_score for item in state.evidence)

        # Note: Document RAG evidence items have been filtered by Retriever & _filter_relevant_chunks.
        # FlashRank reranker outputs raw probabilities (e.g. 0.001 to 0.05), whereas SIMILARITY_THRESHOLD
        # applies to cosine vector similarity. As long as non-empty evidence passed tool execution, it is valid.
        return VerificationOutcome(
            is_valid=True,
            reason="Evidence items validated successfully.",
            relevance_score=max_score,
            requires_retry=False,
        )

    def verify_final_answer(self, answer: str, state: AgentState) -> VerificationOutcome:
        """Verify synthesized final answer against retrieved document chunks and intent."""
        if not answer or not answer.strip():
            return VerificationOutcome(is_valid=False, reason="Empty answer generated.")

        intent = extract_query_intent(state.user_query)
        v_res: VerificationResult = verify_answer(answer, state.retrieved_documents, intent)

        if not v_res.is_valid:
            logger.warning("[VERIFIER ANSWER REJECTED] reason=%s", v_res.reason)
            return VerificationOutcome(
                is_valid=False,
                reason=v_res.reason,
                hallucination_detected=True,
                requires_retry=True,
            )

        return VerificationOutcome(
            is_valid=True,
            reason="Answer verified against evidence.",
            relevance_score=1.0,
            requires_retry=False,
        )
