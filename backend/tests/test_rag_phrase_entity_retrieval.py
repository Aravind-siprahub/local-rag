import pytest
import asyncio
import re
from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_version import DocumentVersion
from app.rag.query_normalizer import normalize_query
from app.rag.intent_router import classify, Route
from app.rag.service import RAGService
from app.services.chat_session_service import ChatSessionService
from app.services.chat_message_service import ChatMessageService
from app.retrieval.retriever import Retriever
from app.retrieval.search import SearchFilters, search_fulltext, search_similar
from app.llm.sanitize import sanitize_response

@pytest.mark.asyncio
async def test_possessive_and_generic_core_values_queries():
    """Verify that possessives ('SipraHub's', 'Microsoft's', 'Acme's') and generic phrases retrieve Core Values correctly."""
    async with AsyncSessionLocal() as session:
        retriever = Retriever(session=session)
        doc_stmt = select(Document).where(Document.deleted_at.is_(None))
        docs = (await session.execute(doc_stmt)).scalars().all()
        if not docs:
            pytest.skip("No documents in DB for integration test.")

        user_id = docs[0].user_id

        queries = [
            "What are SipraHub's core values?",
            "What are SipraHub core values?",
            "What are the core values of SipraHub?",
            "Tell me SipraHub's values.",
            "Which values does SipraHub follow?",
            "What are Microsoft's core values?",
            "What are Acme's core values?",
            "What are the project's core values?",
        ]

        for q in queries:
            norm_res = normalize_query(q)
            assert norm_res.normalized_query != "", f"Query normalization failed for '{q}'"
            
            # Full text search test
            fts_hits = await search_fulltext(session, query_text=q, filters=SearchFilters(user_id=user_id), top_k=10)
            
            # Retriever hybrid search
            chunks = await retriever.retrieve(q, filters=SearchFilters(user_id=user_id), top_k=5, similarity_threshold=0.10)
            assert len(chunks) > 0, f"Retriever returned 0 chunks for query: '{q}'"
            
            # Ensure Core Values / Integrity / Purpose is present in top chunk
            top_content = " ".join([c.chunk_text for c in chunks])
            assert any(term in top_content.lower() for term in ["integrity", "accountability", "core values", "collaboration", "framework", "purpose"]), \
                f"Core Values chunk not retrieved in top hits for query '{q}'"

@pytest.mark.asyncio
async def test_hr_framework_phrase_queries():
    """Verify other HR phrase queries retrieve accurate answers."""
    async with AsyncSessionLocal() as session:
        retriever = Retriever(session=session)
        doc_stmt = select(Document).where(Document.deleted_at.is_(None))
        docs = (await session.execute(doc_stmt)).scalars().all()
        if not docs:
            pytest.skip("No documents in DB for integration test.")

        user_id = docs[0].user_id

        phrase_tests = [
            ("What is the purpose of the HR & Compliance Framework?", ["purpose", "compliance", "framework", "guideline", "environment"]),
            ("What are the standard working hours at SipraHub?", ["working hours", "9:30", "6:30", "hours", "monday"]),
            ("How many hours must employees complete per day?", ["productive", "8 hours", "hours", "9 hours", "break"]),
            ("How much Casual Leave is provided?", ["casual leave", "1 day", "month", "leave"]),
        ]

        for q, expected_keywords in phrase_tests:
            chunks = await retriever.retrieve(q, filters=SearchFilters(user_id=user_id), top_k=5, similarity_threshold=0.10)
            assert len(chunks) > 0, f"Retriever returned 0 chunks for query: '{q}'"
            top_content = " ".join([c.chunk_text for c in chunks]).lower()
            assert any(kw in top_content for kw in expected_keywords), f"Query '{q}' failed to retrieve relevant section. Got: {top_content[:150]}"

@pytest.mark.asyncio
async def test_sanitizer_preserves_valid_answer_sentences():
    """Verify that sanitizer does not erase LLM responses starting with 'The document excerpts state...'."""
    sample_llm_answers = [
        "The document excerpts state that SipraHub's core values are Integrity, Accountability, Collaboration, Excellence, and Respect.",
        "Based on the provided documents, the standard working hours are 9:30 AM to 6:30 PM, Monday through Friday.",
        "The uploaded document states that employees are provided 1 day of Casual Leave per month.",
    ]

    for raw_ans in sample_llm_answers:
        sanitized = sanitize_response(raw_ans)
        assert sanitized != "", f"Sanitizer incorrectly erased valid answer: '{raw_ans}'"
        assert any(term in sanitized for term in ["Integrity", "working hours", "Casual Leave", "9:30", "1 day", "SipraHub"]), \
            f"Sanitizer damaged valid answer. Result: '{sanitized}'"
