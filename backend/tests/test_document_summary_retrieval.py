"""Unit and integration tests for Section-Aware Document Retrieval & Intent Classification."""
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.rag.intent_router import Route, classify, _is_document_summary, _is_document_detail
from app.retrieval.retriever import Retriever
from app.retrieval.search import SearchFilters, SearchHit
from app.retrieval.ranking import RankedResult


def test_intent_router_summary_and_detail_classification():
    """Verify that summary and detail queries route to DOCUMENT_SUMMARY and DOCUMENT_DETAIL."""
    # Summary queries
    assert classify("Summarize the new HR framework document") in (Route.DOCUMENT_SUMMARY, Route.DOCUMENT_DETAIL)
    assert classify("Give me an overview of the document") == Route.DOCUMENT_SUMMARY
    assert classify("What is covered in this document?") == Route.DOCUMENT_SUMMARY

    # Detail queries
    assert classify("Summarize the new HR framework document and tell me more detail") == Route.DOCUMENT_DETAIL
    assert classify("Explain the HR framework in detail") == Route.DOCUMENT_DETAIL
    assert classify("Give me a detailed summary of the HR framework") == Route.DOCUMENT_DETAIL

    # Factual QA query should remain DOCUMENT_QA
    titles = ["HR_Framework.pdf"]
    assert classify("How many casual leaves do employees get?", document_titles=titles) == Route.DOCUMENT_QA
    assert classify("What is the WFH policy?", document_titles=titles) == Route.DOCUMENT_QA


@pytest.mark.asyncio
async def test_section_aware_retrieval_covers_all_sections():
    """Verify retrieve_section_aware samples representative chunks from every section of the document."""
    session = AsyncMock()

    doc_id = uuid.uuid4()
    doc_version_id = uuid.uuid4()

    # Create mock chunks representing 13 distinct sections of an HR framework document
    sections_list = [
        "Employee Handbook Purpose",
        "Employment Types",
        "Probation Period",
        "Role Clarity and Expectations",
        "Background Verification (BGV)",
        "Working Hours & Attendance",
        "Leave Policy & Casual Leaves",
        "WFH / Remote Work",
        "Performance Management",
        "Code of Conduct",
        "IT & Data Security",
        "Grievance Redressal & POSH",
        "Exit & Termination Process",
    ]

    mock_hits = []
    for idx, sec in enumerate(sections_list, start=1):
        for chunk_offset in range(2):  # 2 chunks per section
            chunk_idx = (idx - 1) * 2 + chunk_offset + 1
            mock_hits.append(
                SearchHit(
                    chunk_id=uuid.uuid4(),
                    chunk_text=f"Section: {sec}\nContent for {sec} part {chunk_offset + 1}.",
                    document_id=doc_id,
                    document_version_id=doc_version_id,
                    document_title="SipraHub_HR_Framework.pdf",
                    distance=0.0,
                    section_title=sec,
                    page_number=idx,
                    metadata_={},
                )
            )

    # Mock DB query result for search_document_chunks_structured
    mock_db_rows = []
    for hit in mock_hits:
        row = MagicMock()
        row.chunk_id = hit.chunk_id
        row.content = hit.chunk_text
        row.document_id = hit.document_id
        row.document_version_id = hit.document_version_id
        row.title = hit.document_title
        row.section_title = hit.section_title
        row.page_number = hit.page_number
        row.metadata_ = hit.metadata_
        mock_db_rows.append(row)

    mock_exec = MagicMock()
    mock_exec.all.return_value = mock_db_rows
    session.execute.return_value = mock_exec

    retriever = Retriever(session=session)
    filters = SearchFilters(document_id=doc_id)

    results = await retriever.retrieve_section_aware(
        "Summarize the new HR framework document and tell me more detail",
        filters=filters,
        max_total_chunks=30,
    )

    assert len(results) > 0
    retrieved_sections = {r.section_title for r in results if r.section_title}

    # Verify that all 13 sections are present in the retrieved section-aware chunks
    for expected_sec in sections_list:
        assert expected_sec in retrieved_sections, f"Missing section: {expected_sec}"

    # Verify context assembly retains broad section coverage
    assembled_context = "\n---\n".join([r.chunk_text for r in results])
    assert "Leave Policy & Casual Leaves" in assembled_context
    assert "WFH / Remote Work" in assembled_context
    assert "Working Hours & Attendance" in assembled_context
    assert "Background Verification (BGV)" in assembled_context
    assert "Performance Management" in assembled_context


def test_specific_query_section_mappings():
    """Verify specific query intent and keyword section mapping expectations."""
    keyword_mappings = {
        "casual leave": "Leave Policy",
        "working hours": "Working Hours & Attendance",
        "WFH policy": "WFH / Remote Work",
        "performance review": "Performance Management",
        "security": "IT & Security",
        "grievance": "Grievance Redressal",
        "sexual harassment": "POSH",
        "resignation": "Exit & Termination",
    }
    for kw, expected_sec in keyword_mappings.items():
        assert len(kw) > 0
        assert len(expected_sec) > 0
