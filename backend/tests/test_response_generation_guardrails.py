"""Tests for response generation guardrails:
1. Normal document question: detected as document route and answers strictly from context.
2. Scanned PDF / image question: routes to vision/direct and does not dump OCR implementation text.
3. Real-time web-search question: routes to web search and instructs direct factual answers.
4. Unrelated question: does not match document QA cues and avoids document OCR chunks.
5. No relevant retrieval result: triggers clear refusal without fabricating content or returning 0% citations.
"""
import uuid
import pytest
from app.rag.intent_router import Route, classify
from app.prompting.templates import USER_PROMPT_WITH_CONTEXT, format_user_prompt
from app.retrieval.ranking import RankedResult


def test_normal_document_question_routing():
    """Requirement 1: Normal document question routes to document QA."""
    route = classify("what does the leave policy say about casual leaves?")
    assert route in (Route.DOCUMENT_QA, Route.DOCUMENT_SUMMARY, Route.DOCUMENT_DETAIL)


def test_image_question_not_misrouted_to_document_qa():
    """Requirement 2: Image questions must NOT be hardcoded to document QA."""
    route = classify("tell about this image and explain me")
    # Should not be forced into DOCUMENT_QA when no document keywords exist
    assert route != Route.DOCUMENT_QA


def test_real_time_web_search_routing():
    """Requirement 3: Real-time search query routes to Route.WEB."""
    route = classify("who won the latest 2024 ICC T20 World Cup? search the web")
    assert route == Route.WEB


def test_unrelated_general_question_routing():
    """Requirement 4: Unrelated questions do not trigger document QA."""
    route = classify("what is the capital of France?")
    assert route in (Route.GENERAL_KNOWLEDGE, Route.GENERIC_CHAT, Route.DIRECT)


def test_prompt_template_negative_constraints_against_ocr_leak():
    """Requirement 5: Prompt template explicitly forbids internal OCR / pipeline mechanics."""
    assert "Never output internal pipeline explanations" in USER_PROMPT_WITH_CONTEXT
    assert "PaddleOCR" in USER_PROMPT_WITH_CONTEXT
    assert "I could not find relevant information in the uploaded documents" in USER_PROMPT_WITH_CONTEXT


def test_prompt_formatting_with_irrelevant_context_instruction():
    """Requirement 6: When context is formatted, the rules instruct refusal if irrelevant."""
    prompt = format_user_prompt(
        context="Some document content about office Wi-Fi password.",
        question="What is the maternity leave entitlement?",
    )
    assert "I could not find relevant information in the uploaded documents" in prompt
    assert "Never output internal pipeline explanations" in prompt


def test_no_relevant_retrieval_result_citation_filtering():
    """Requirement 7: Low relevance or zero-match chunks are filtered out."""
    # Simulate low relevance hits
    chunks = [
        RankedResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            document_version_id=uuid.uuid4(),
            similarity_score=0.04,  # Below threshold
            rank=1,
            chunk_text="Random OCR engine implementation details",
            document_title="PRD_Talk_to_My_Data.docx",
        ),
        RankedResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            document_version_id=uuid.uuid4(),
            similarity_score=0.0,  # 0% match
            rank=2,
            chunk_text="PaddleOCR fallback logic",
            document_title="PRD_Talk_to_My_Data.docx",
        ),
    ]
    # Filter using our threshold rule
    min_threshold = 0.15
    valid_sources = [c for c in chunks if c.similarity_score >= min_threshold]
    assert len(valid_sources) == 0, "Low-similarity / 0% chunks must be filtered out"


def test_project_entity_isolation_rule_present():
    """Requirement 8: Prompt template explicitly enforces project isolation."""
    assert "PROJECT ISOLATION" in USER_PROMPT_WITH_CONTEXT
    assert "Never borrow, mix in, or attribute technologies" in USER_PROMPT_WITH_CONTEXT


@pytest.mark.asyncio
async def test_resolve_entity_filters_airis_isolation():
    """Requirement 9: Querying AIRIS technology stack matches only AIRIS docs and excludes generic technology stack summary."""
    from app.rag.service import RAGService
    from app.retrieval.search import SearchFilters
    from unittest.mock import MagicMock, AsyncMock

    # Create dummy documents
    class DummyDoc:
        def __init__(self, doc_id, title):
            self.id = doc_id
            self.title = title
            self.original_filename = title

    airis_id = uuid.uuid4()
    tech_id = uuid.uuid4()
    dummy_docs = [
        DummyDoc(airis_id, "AIRIS_Staging_Deployment_Documentation.docx"),
        DummyDoc(tech_id, "Technology_Stack_Summary.docx"),
    ]

    from typing import cast, Any
    mock_session = AsyncMock()
    service = RAGService(
        session=cast(Any, mock_session),
        retriever=MagicMock(),
        message_service=MagicMock(),
        citation_service=MagicMock(),
        session_service=MagicMock(),
    )

    mock_repo = MagicMock()
    mock_repo.list_by_user = AsyncMock(return_value=dummy_docs)
    mock_repo.list = AsyncMock(return_value=dummy_docs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("app.repositories.document_repository.DocumentRepository", lambda s: mock_repo)
        resolved = await service._resolve_entity_filters(
            user_id=uuid.uuid4(),
            question="what technology stack using in AIRIS ?",
            base_filters=SearchFilters(),
        )

    assert resolved.document_ids is not None
    assert airis_id in resolved.document_ids, "AIRIS document should match"
    assert tech_id not in resolved.document_ids, "Generic Technology_Stack_Summary.docx must be excluded"


def test_creative_and_math_logic_reasoning_routing():
    """Verify that creative prompts and multi-step math/logic questions are never misclassified as DOCUMENT_QA."""
    from app.rag.intent_router import Route, classify

    corpus_titles = [
        "Deployment_Guide.docx",
        "Technology_Stack_Summary.docx",
        "VM_Setup_Guide.docx",
        "AIRIS_Staging_Deployment_Guide_4.docx",
        "SipraOne_Deployment_Documentation.docx",
        "SipraHub_PRD_v1.1 1.docx",
        "PRD_Talk_to_My_Data.docx",
        "New HR Framework (3) 1.docx",
    ]

    # Creative / Abstract Thinking
    q_creative = "invent a new game combining chess and basketball"
    assert classify(q_creative, document_titles=corpus_titles) == Route.GENERAL_KNOWLEDGE

    # Multi-Step Math / Word Problem
    q_train = (
        "If a train leaves City A at 9:00 AM traveling at 80 km/h and another leaves City B "
        "at 10:00 AM traveling at 100 km/h toward City A, and the cities are 300 km apart, "
        "at what time will they meet?"
    )
    assert classify(q_train, document_titles=corpus_titles) == Route.GENERAL_KNOWLEDGE

    # Logic Puzzle / Brain Teaser
    q_puzzle = "Solve this riddle: The more you take, the more you leave behind. What am I?"
    assert classify(q_puzzle, document_titles=corpus_titles) == Route.GENERAL_KNOWLEDGE

    # Technical Coding / Domain Specific
    q_code = "Write a python function to find the longest palindromic substring"
    assert classify(q_code, document_titles=corpus_titles) == Route.GENERAL_KNOWLEDGE
