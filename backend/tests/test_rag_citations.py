"""Tests for RAG citation generation, chunk provenance, and anti-hallucination validation."""
import uuid
import pytest
from app.services.metadata import Chunk, ContentType
from app.retrieval.ranking import RankedResult
from app.rag.citations import (
    build_citation_label,
    extract_inline_citations,
    validate_and_sanitize_citations,
)
from app.rag.context_builder import ContextBuilder
from app.schemas.chat import ChatCitationResponse


def test_chunk_provenance_metadata():
    """Verify Chunk stores file_name, file_type, source_location and serializes them."""
    doc_id = uuid.uuid4()
    chunk = Chunk(
        id="chunk_123",
        document_id=doc_id,
        document_name="New HR Framework.docx",
        file_name="New HR Framework.docx",
        file_type="docx",
        source_location="Section: Leave Policy",
        page_number=0,
        section="Leave Policy",
        content_type=ContentType.PARAGRAPH,
        text="Employees are entitled to 1 Casual Leave per month.",
    )
    meta_dict = chunk.to_metadata_dict()
    assert meta_dict["file_name"] == "New HR Framework.docx"
    assert meta_dict["file_type"] == "docx"
    assert meta_dict["source_location"] == "Section: Leave Policy"
    assert meta_dict["document_name"] == "New HR Framework.docx"


def test_build_citation_label_pdf():
    """Verify PDF citation formats."""
    # PDF with page
    assert build_citation_label("Report.pdf", page_number=5) == "[Report.pdf, p. 5]"
    # PDF without page
    assert build_citation_label("Report.pdf", page_number=0) == "[Report.pdf]"
    assert build_citation_label("Report.pdf", page_number=None) == "[Report.pdf]"


def test_build_citation_label_docx():
    """Verify DOCX citation formats."""
    # DOCX with page
    assert build_citation_label("Handbook.docx", page_number=8) == "[Handbook.docx, p. 8]"
    # DOCX with section
    assert build_citation_label("Handbook.docx", section_title="Leave Policy") == "[Handbook.docx, Section: Leave Policy]"
    # DOCX with section already having 'Section:'
    assert build_citation_label("Handbook.docx", section_title="Section: Leave Policy") == "[Handbook.docx, Section: Leave Policy]"
    # DOCX without page or section
    assert build_citation_label("Handbook.docx") == "[Handbook.docx]"


def test_build_citation_label_markdown():
    """Verify Markdown citation formats."""
    assert build_citation_label("deployment-guide.md", section_title="Production Deployment") == "[deployment-guide.md, Production Deployment]"
    assert build_citation_label("README.md") == "[README.md]"


def test_build_citation_label_excel():
    """Verify Excel citation formats."""
    # Sheet only
    assert build_citation_label("Sales_Report.xlsx", metadata={"sheet": "Q2"}) == "[Sales_Report.xlsx, Sheet: Q2]"
    # Sheet + rows
    assert build_citation_label("Sales_Report.xlsx", metadata={"sheet": "Q2", "row_start": 2, "row_end": 40}) == "[Sales_Report.xlsx, Sheet: Q2, Rows 2-40]"
    # Explicit source_location takes precedence
    assert build_citation_label("Sales_Report.xlsx", source_location="Sheet: Q2") == "[Sales_Report.xlsx, Sheet: Q2]"


def test_validate_and_sanitize_valid_citations():
    """Verify valid inline citations are preserved and matched."""
    doc_id = uuid.uuid4()
    chunk_1 = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="Employees receive 1 Casual Leave per month.",
        document_id=doc_id,
        similarity_score=0.88,
        rank=1,
        document_title="New HR Framework.docx",
        section_title="Leave Policy",
        page_number=12,
    )
    chunk_2 = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="Unused leave can be carried forward within the calendar year.",
        document_id=doc_id,
        similarity_score=0.82,
        rank=2,
        document_title="New HR Framework.docx",
        section_title="Leave Carryover",
        page_number=13,
    )

    answer = "Employees receive 1 Casual Leave per month [New HR Framework.docx, p. 12] and unused leave can be carried forward [New HR Framework.docx, p. 13]."
    sanitized, verified = validate_and_sanitize_citations(answer, [chunk_1, chunk_2])

    assert "[New HR Framework.docx, p. 12]" in sanitized
    assert "[New HR Framework.docx, p. 13]" in sanitized
    assert len(verified) == 2
    assert verified[0]["page_number"] == 12
    assert verified[1]["page_number"] == 13


def test_anti_hallucination_strips_fake_page():
    """Verify hallucinated page citations (not in retrieved chunks) are stripped or corrected."""
    doc_id = uuid.uuid4()
    chunk_real = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="Managers must approve time off requests.",
        document_id=doc_id,
        similarity_score=0.85,
        rank=1,
        document_title="HR_Policy.pdf",
        page_number=3,
    )

    # LLM hallucinated page 99
    answer = "Managers must approve time off requests. [HR_Policy.pdf, p. 99]"
    sanitized, verified = validate_and_sanitize_citations(answer, [chunk_real])

    # p. 99 should NOT appear in sanitized answer
    assert "p. 99" not in sanitized
    # Real chunk p. 3 is present in verified citations
    assert len(verified) >= 1
    assert verified[0]["page_number"] == 3


def test_anti_hallucination_strips_fake_document():
    """Verify citations to non-existent documents are completely stripped."""
    doc_id = uuid.uuid4()
    chunk_real = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="The system uses PostgreSQL 16.",
        document_id=doc_id,
        similarity_score=0.90,
        rank=1,
        document_title="Architecture.md",
        section_title="Database",
    )

    # LLM cited completely fabricated doc
    answer = "The system uses PostgreSQL 16. [FakeReport.pdf, p. 45]"
    sanitized, verified = validate_and_sanitize_citations(answer, [chunk_real])

    assert "FakeReport.pdf" not in sanitized


def test_context_builder_injects_citation_labels():
    """Verify ContextBuilder injects standardized Citation tags into chunk headers."""
    doc_id = uuid.uuid4()
    chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="Revenue for Q2 was 12.4 lakh.",
        document_id=doc_id,
        similarity_score=0.85,
        rank=1,
        document_title="Sales_Report.xlsx",
        metadata_={"sheet": "Q2"},
    )
    builder = ContextBuilder(similarity_threshold=0.0)
    result = builder.build_context([chunk], query="What was Q2 revenue?")

    assert result.has_context is True
    assert "Citation: [Sales_Report.xlsx, Sheet: Q2]" in result.formatted_context
    assert len(result.selected_chunks) == 1
    assert result.selected_chunks[0].citation_label == "[Sales_Report.xlsx, Sheet: Q2]"


def test_chat_citation_schema_backward_compatibility():
    """Verify ChatCitationResponse supports both legacy and new fields."""
    cid = uuid.uuid4()
    did = uuid.uuid4()
    resp = ChatCitationResponse(
        chunk_id=cid,
        chunk_text="Some text",
        document_id=did,
        similarity_score=0.89,
        rank=1,
        document_title="Report.pdf",
        document_name="Report.pdf",
        file_name="Report.pdf",
        relevance_score=0.89,
        source_location="p. 5",
        citation_id="C1",
        citation_label="[Report.pdf, p. 5]",
    )
    assert resp.chunk_id == cid
    assert resp.file_name == "Report.pdf"
    assert resp.citation_label == "[Report.pdf, p. 5]"
    assert resp.source_location == "p. 5"
