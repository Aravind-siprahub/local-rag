"""Task 7: Comprehensive regression test suite for RAG accuracy upgrade architecture."""
from __future__ import annotations

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.rag.query_understanding import extract_query_intent, AttributeCategory
from app.rag.intent_router import Route, classify
from app.rag.verifier import verify_answer, VerificationResult
from app.retrieval.ranking import RankedResult, _fallback_heuristic_rerank
from app.rag.service import _filter_relevant_chunks


def test_1_technology_question_intent_extraction():
    """Verify that technology questions extract entity and technology category."""
    query = "what frontend and backend are using talk to my data"
    intent = extract_query_intent(query)

    assert intent.entity == "Talk to My Data"
    assert intent.category == AttributeCategory.TECHNOLOGY
    assert "frontend" in intent.attributes
    assert "backend" in intent.attributes


def test_2_port_question_intent_extraction():
    """Verify that port questions extract configuration category and port attributes."""
    query = "what ports do frontend and backend use in sipraone"
    intent = extract_query_intent(query)

    assert intent.entity == "SipraOne"
    assert intent.category == AttributeCategory.CONFIGURATION
    assert "frontend port" in intent.attributes or "port" in intent.attributes


def test_3_attribute_isolation_reranking():
    """Verify technology query boosts framework chunks and penalizes pure port chunks."""
    tech_chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="Frontend: React with Vite. Backend: FastAPI.",
        document_id=uuid.uuid4(),
        similarity_score=0.75,
        rank=1,
        document_title="PRD_Talk_to_My_Data.docx",
    )
    port_chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="Frontend port: 4173. Backend port: 5000. PM2 process manager.",
        document_id=uuid.uuid4(),
        similarity_score=0.78,
        rank=2,
        document_title="Deployment_Guide.docx",
    )

    # 1. Tech query preference
    tech_scored = _fallback_heuristic_rerank("what frontend and backend are using talk to my data", [port_chunk, tech_chunk])
    assert tech_scored[0][1].chunk_id == tech_chunk.chunk_id

    # 2. Port query preference
    port_scored = _fallback_heuristic_rerank("what ports do frontend and backend use in sipraone", [tech_chunk, port_chunk])
    assert port_scored[0][1].chunk_id == port_chunk.chunk_id


def test_4_project_isolation_filter():
    """Verify post-reranking filter isolates tech query chunks from pure deployment port noise."""
    tech_chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="Frontend: React with Vite. Backend: FastAPI.",
        document_id=uuid.uuid4(),
        similarity_score=0.85,
        rank=1,
        document_title="PRD_Talk_to_My_Data.docx",
    )
    port_chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="AIRIS port 8000 and PM2 deployment.",
        document_id=uuid.uuid4(),
        similarity_score=0.40,
        rank=2,
        document_title="AIRIS_Deployment.docx",
    )

    filtered = _filter_relevant_chunks("what frontend and backend are using talk to my data", [tech_chunk, port_chunk])
    assert len(filtered) == 1
    assert filtered[0].chunk_id == tech_chunk.chunk_id


def test_5_answer_verification_rule_failure():
    """Verify that verifier flags pure port substitution for technology queries."""
    intent = extract_query_intent("what frontend and backend are using talk to my data")
    context_chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="Frontend: React with Vite. Backend: FastAPI.",
        document_id=uuid.uuid4(),
        similarity_score=0.85,
        rank=1,
    )

    # Bad answer: substituted ports
    bad_answer = "Port 4173 for frontend and Git for source-code."
    v_res_bad = verify_answer(bad_answer, [context_chunk], intent)
    assert not v_res_bad.is_valid
    assert "Substituted ports" in (v_res_bad.reason or "")

    # Good answer: framework names directly supported
    good_answer = "Frontend: React with Vite. Backend: FastAPI."
    v_res_good = verify_answer(good_answer, [context_chunk], intent)
    assert v_res_good.is_valid


def test_6_broken_english_normalization():
    """Verify that broken English query 'what fronted and backend are using talk to my data' normalizes properly."""
    query = "what fronted and backend are using talk to my data"
    intent = extract_query_intent(query)

    assert intent.entity == "Talk to My Data"
    assert "frontend" in intent.attributes
    assert "backend" in intent.attributes


def test_7_multi_document_conflict_preservation():
    """Verify conflict preservation rule formatting guidelines in SYSTEM_PROMPT."""
    from app.core.config import get_settings
    settings = get_settings()

    assert "multiple values" in settings.SYSTEM_PROMPT


def test_8_general_knowledge_routing():
    """Verify 'what is Python?' stays GENERAL_KNOWLEDGE route."""
    docs = ["PRD_Talk_to_My_Data.docx", "SipraOne_Deployment.docx"]
    route = classify("what is Python?", document_titles=docs)

    assert route == Route.GENERAL_KNOWLEDGE


def test_9_wait_other_documents_monologue_sanitization():
    """Verify that 'Wait, the other documents might be part of the same project...' monologue is sanitized."""
    from app.llm.sanitize import sanitize_response
    leaked = "Wait, the other documents might be part of the same project. Let me re-read the"
    sanitized = sanitize_response(leaked)
    assert sanitized == "" or "requested information is not found" in sanitized.lower()


def test_10_discrepancy_monologue_sanitization():
    """Verify that 'So there's a discrepancy between the two documents...' monologue is sanitized."""
    from app.llm.sanitize import sanitize_response
    leaked = "So there's a discrepancy between the two documents. The answer should state both values with '"
    sanitized = sanitize_response(leaked)
    assert sanitized == "" or "discrepancy" not in sanitized.lower()

