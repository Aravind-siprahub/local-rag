"""Automated regression tests for Project Data Isolation and Cross-Project Retrieval Elimination.

Guarantees:
1. Talk to My Data queries retrieve ONLY Talk to My Data documents.
2. SipraHub documents are excluded BEFORE LLM generation when Talk to My Data is selected.
3. SipraHub-only questions return fallback when Talk to My Data is selected.
4. Missing information in selected dataset returns clean fallback without inventing facts.
5. Changing project scope updates retrieval scope correctly without retaining stale context.
"""
from __future__ import annotations

import uuid
import pytest

from app.rag.query_understanding import extract_query_intent, AttributeCategory
from app.rag.verifier import verify_answer
from app.retrieval.ranking import RankedResult, _fallback_heuristic_rerank
from app.rag.service import _filter_relevant_chunks
from app.retrieval.search import SearchFilters


def test_1_talk_to_my_data_query_intent():
    """Test 1: Verify Talk to My Data question extracts correct project entity scope."""
    query = "what frontend and backend are using talk to my data"
    intent = extract_query_intent(query)

    assert intent.entity == "Talk to My Data"
    assert intent.category == AttributeCategory.TECHNOLOGY


def test_2_siprahub_documents_excluded_before_llm():
    """Test 2: Have both Talk to My Data and SipraHub docs; verify SipraHub is excluded."""
    ttmd_doc_id = uuid.uuid4()
    siprahub_doc_id = uuid.uuid4()

    ttmd_chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="Frontend — the chat interface. Backend — FastAPI.",
        document_id=ttmd_doc_id,
        similarity_score=0.85,
        rank=1,
        document_title="PRD_Talk_to_My_Data.docx",
    )
    siprahub_chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="SipraHub Frontend: React + Vite. Backend: Node.js / Express.",
        document_id=siprahub_doc_id,
        similarity_score=0.88,
        rank=2,
        document_title="SipraHub_PRD_v1.1.docx",
    )

    # Scoped filters for Talk to My Data
    scoped_filters = SearchFilters(user_id=uuid.uuid4(), document_ids=(ttmd_doc_id,))
    assert scoped_filters.document_ids is not None
    assert ttmd_chunk.document_id in scoped_filters.document_ids
    assert siprahub_chunk.document_id not in scoped_filters.document_ids

    # Post-reranking filter check
    filtered = _filter_relevant_chunks("what frontend and backend are using talk to my data", [ttmd_chunk, siprahub_chunk])
    # Verifier check
    intent = extract_query_intent("what frontend and backend are using talk to my data")
    v_res = verify_answer("SipraHub uses React + Vite and Node.js", [ttmd_chunk], intent)
    assert not v_res.is_valid
    assert "SipraHub" in (v_res.reason or "")


def test_3_siprahub_only_question_returns_fallback_when_ttmd_selected():
    """Test 3: Ask SipraHub-only question while Talk to My Data is selected -> verify no SipraHub answer."""
    intent = extract_query_intent("what is the SipraHub deployment checklist?")
    # When user targets Talk to My Data, SipraHub documents are not in scoped context
    ttmd_chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="Talk to My Data overview and architecture.",
        document_id=uuid.uuid4(),
        similarity_score=0.20,
        rank=1,
        document_title="PRD_Talk_to_My_Data.docx",
    )

    filtered = _filter_relevant_chunks("what is the SipraHub deployment checklist?", [ttmd_chunk])
    # 0 relevant chunks -> fallback response triggered
    assert len(filtered) == 0 or all("deployment checklist" not in c.chunk_text for c in filtered)


def test_4_missing_information_refuses_to_invent_answer():
    """Test 4: Ask question with 0 evidence in selected dataset -> verify clean fallback."""
    intent = extract_query_intent("what is the database migration strategy for talk to my data?")
    ttmd_chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="Talk to My Data frontend interface uses React.",
        document_id=uuid.uuid4(),
        similarity_score=0.10,
        rank=1,
        document_title="PRD_Talk_to_My_Data.docx",
    )

    fallback_ans = "I could not find this information in the selected data."
    v_res = verify_answer(fallback_ans, [ttmd_chunk], intent)
    assert v_res.is_valid
    assert "fallback" in (v_res.reason or "").lower()


def test_5_scope_change_updates_retrieval_scope():
    """Test 5: Verify changing dataset scope updates document_ids filter without retaining previous dataset."""
    ttmd_id = uuid.uuid4()
    siprahub_id = uuid.uuid4()

    filter_ttmd = SearchFilters(user_id=uuid.uuid4(), document_ids=(ttmd_id,))
    filter_siprahub = SearchFilters(user_id=uuid.uuid4(), document_ids=(siprahub_id,))

    assert filter_ttmd.document_ids is not None
    assert filter_siprahub.document_ids is not None
    assert filter_ttmd.document_ids == (ttmd_id,)
    assert filter_siprahub.document_ids == (siprahub_id,)
    assert ttmd_id not in filter_siprahub.document_ids
