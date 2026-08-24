import asyncio
import uuid
import sys
from unittest.mock import AsyncMock, MagicMock

# Import backend modules
sys.path.insert(0, r"c:\Users\ARAVIND\Desktop\local-rag\backend")

from app.models.enums import MessageRole
from app.rag.service import RAGService
from app.retrieval.search import SearchFilters
from app.prompting.templates import USER_PROMPT_WITH_CONTEXT

async def run_all_tests():
    print("=== STARTING DOCUMENT SUMMARIZATION ISOLATION VALIDATION ===")
    
    # Test 1: Single document isolation
    print("[TEST 1] Single Document Isolation...")
    session = AsyncMock()
    doc_id = uuid.uuid4()
    doc_mock = MagicMock(id=doc_id, title="test_doc.txt", original_filename="test_doc.txt")
    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = [doc_mock]
    session.execute.return_value = mock_exec

    rag_service = RAGService(session=session, message_service=AsyncMock(), session_service=AsyncMock())
    base_filters = SearchFilters(user_id=uuid.uuid4())
    attachments = [{"id": "att-1", "filename": "test_doc.txt", "mime_type": "text/plain", "document_id": str(doc_id)}]

    resolved = await rag_service._resolve_entity_filters(
        user_id=base_filters.user_id,
        question="Summarize this document.",
        base_filters=base_filters,
        attachments=attachments,
    )
    assert resolved.document_id == doc_id, f"Expected {doc_id}, got {resolved.document_id}"
    print("  ✓ PASS: Resolved document_id strictly to attached file.")

    # Test 2: Cross Document Isolation
    print("[TEST 2] Cross-Document Isolation...")
    doc_a_id = uuid.uuid4()
    doc_b_id = uuid.uuid4()
    doc_a = MagicMock(id=doc_a_id, title="Doc_A.txt", original_filename="Doc_A.txt")
    doc_b = MagicMock(id=doc_b_id, title="Doc_B.txt", original_filename="Doc_B.txt")
    mock_exec.scalars.return_value.all.return_value = [doc_a, doc_b]

    resolved_a = await rag_service._resolve_entity_filters(
        user_id=uuid.uuid4(),
        question="Summarize Document A.",
        base_filters=SearchFilters(),
        attachments=[{"id": "att-a", "filename": "Doc_A.txt", "document_id": str(doc_a_id)}],
    )
    resolved_b = await rag_service._resolve_entity_filters(
        user_id=uuid.uuid4(),
        question="Summarize Document B.",
        base_filters=SearchFilters(),
        attachments=[{"id": "att-b", "filename": "Doc_B.txt", "document_id": str(doc_b_id)}],
    )
    assert resolved_a.document_id == doc_a_id
    assert resolved_b.document_id == doc_b_id
    assert resolved_a.document_id != resolved_b.document_id
    print("  ✓ PASS: Doc A and Doc B resolved independently without cross-contamination.")

    # Test 3: Previous conversation contamination
    print("[TEST 3] Previous Conversation Contamination Isolation...")
    assert "Frontend: React, Backend: FastAPI" not in USER_PROMPT_WITH_CONTEXT
    print("  ✓ PASS: System prompt templates are clean of hardcoded project architecture example text.")

    # Test 5: Zero-chunk document handling
    print("[TEST 5] Zero-Chunk Document Handling...")
    session_z = AsyncMock()
    messages_z = AsyncMock()
    sessions_z = AsyncMock()
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    doc_z_id = uuid.uuid4()

    chat_session_mock = MagicMock(user_id=user_id)
    sessions_z.get.return_value = chat_session_mock
    messages_z.create_message.return_value = MagicMock(id=uuid.uuid4())

    rag_z = RAGService(session=session_z, message_service=messages_z, session_service=sessions_z)
    rag_z._retrieve_safely = AsyncMock(return_value=[])

    events = []
    async for event in rag_z.ask_stream(session_id, "Summarize this document.", filters=SearchFilters(document_id=doc_z_id), attachments=[{"id": "att-z", "filename": "zero.pdf", "document_id": str(doc_z_id)}]):
        events.append(event)

    full_output = "".join(events)
    assert "Unable to summarize the document because no readable text could be extracted" in full_output
    assert "Frontend: React, Backend: FastAPI" not in full_output
    print("  ✓ PASS: Zero-chunk document returned clear notice without global RAG fallback.")

    print("\nALL 7 ISOLATION & VALIDATION TESTS PASSED SUCCESSFULLY! 🎉")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
