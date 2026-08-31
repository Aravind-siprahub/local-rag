"""Real RAG Retrieval & Chunking Integration Test Suite.

Validates end-to-end retrieval, semantic chunking, list merging, query intent,
and attribute detection for HR Framework documents and SipraHub queries.
"""
import uuid
import pytest
from app.services.metadata import BlockType, DocumentBlock, ParsedDocument
from app.services.parser import DocumentParser
from app.services.chunker import SemanticChunker
from app.rag.query_understanding import extract_query_intent, AttributeCategory
from app.rag.attribute_detector import detect_requested_attributes, RequestedAttribute
from app.retrieval.search import SearchFilters


def test_docx_list_merging_preserves_core_values_chunk():
    """Verify that consecutive bullet points in a DOCX are merged into a single chunk containing all 5 core values."""
    blocks = [
        DocumentBlock(block_type=BlockType.HEADING, text="Our Core Values", level=1),
        DocumentBlock(block_type=BlockType.LIST, text="- Integrity – Do the right thing"),
        DocumentBlock(block_type=BlockType.LIST, text="- Accountability – Own your work"),
        DocumentBlock(block_type=BlockType.LIST, text="- Collaboration – Work as a team"),
        DocumentBlock(block_type=BlockType.LIST, text="- Excellence – Strive for quality"),
        DocumentBlock(block_type=BlockType.LIST, text="- Respect – Value people and ideas"),
    ]

    parser = DocumentParser()
    merged = parser._merge_consecutive_lists(blocks)

    # Heading block + 1 merged list block
    assert len(merged) == 2
    assert merged[0].block_type == BlockType.HEADING
    assert merged[1].block_type == BlockType.LIST

    parsed_doc = ParsedDocument(
        document_id=uuid.uuid4(),
        document_name="New HR Framework (3) 1.docx",
        language="en",
        page_count=1,
        parser_used="python-docx",
        blocks=merged,
    )

    chunker = SemanticChunker()
    chunks = chunker.chunk_document(parsed_doc)

    assert len(chunks) == 1
    chunk_text = chunks[0].text
    assert "Integrity" in chunk_text
    assert "Accountability" in chunk_text
    assert "Collaboration" in chunk_text
    assert "Excellence" in chunk_text
    assert "Respect" in chunk_text
    assert "Our Core Values" in chunk_text or chunks[0].section == "Our Core Values" or chunks[0].breadcrumb == "Our Core Values"


def test_docx_list_merging_preserves_purpose_chunk():
    """Verify that handbook purpose bullet points are merged into a single chunk."""
    blocks = [
        DocumentBlock(block_type=BlockType.HEADING, text="Purpose of Handbook", level=1),
        DocumentBlock(block_type=BlockType.LIST, text="- Provide clarity on company policies and processes"),
        DocumentBlock(block_type=BlockType.LIST, text="- Define employee roles, responsibilities, and expectations"),
        DocumentBlock(block_type=BlockType.LIST, text="- Ensure consistency in operations"),
        DocumentBlock(block_type=BlockType.LIST, text="- Support a positive and productive work environment"),
    ]

    parser = DocumentParser()
    merged = parser._merge_consecutive_lists(blocks)

    assert len(merged) == 2
    assert merged[0].block_type == BlockType.HEADING
    assert merged[1].block_type == BlockType.LIST

    parsed_doc = ParsedDocument(
        document_id=uuid.uuid4(),
        document_name="New HR Framework (3) 1.docx",
        language="en",
        page_count=1,
        parser_used="python-docx",
        blocks=merged,
    )

    chunker = SemanticChunker()
    chunks = chunker.chunk_document(parsed_doc)

    assert len(chunks) == 1
    chunk_text = chunks[0].text
    assert "clarity on company policies" in chunk_text
    assert "Define employee roles" in chunk_text
    assert "consistency in operations" in chunk_text
    assert "positive and productive" in chunk_text


def test_hr_framework_query_intent_classification():
    """Verify that asking about 'HR & Compliance Framework' is NOT misclassified as software tech stack."""
    q_purpose = "What is the purpose of the HR & Compliance Framework?"
    intent = extract_query_intent(q_purpose)

    assert intent.category == AttributeCategory.GENERAL
    assert intent.normalized_query == q_purpose
    assert "What frontend and backend technologies" not in intent.normalized_query

    q_values = "What are SipraHub's core values?"
    intent_values = extract_query_intent(q_values)
    assert intent_values.category == AttributeCategory.GENERAL
    assert intent_values.normalized_query == q_values


def test_hr_framework_attribute_detection():
    """Verify that 'HR & Compliance Framework' query does NOT trigger FRAMEWORK_TECH_STACK attribute."""
    q_purpose = "What is the purpose of the HR & Compliance Framework?"
    attrs = detect_requested_attributes(q_purpose)

    assert RequestedAttribute.FRAMEWORK_TECH_STACK not in attrs
    assert RequestedAttribute.PORT_NETWORKING not in attrs
    assert RequestedAttribute.GENERAL_ATTRIBUTE in attrs
