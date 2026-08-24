"""Dedicated test suite for DOCUMENT_LIST intent routing and document listing service."""
from __future__ import annotations

import uuid
import pytest
from app.rag.intent_router import Route, classify
from app.rag.service import RAGService
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.models.user import User
from app.models.chat_session import ChatSession

@pytest.mark.asyncio
async def test_document_listing_natural_language_variants() -> None:
    """Requirement 14, Test 7: Natural language variants route to DOCUMENT_LIST."""
    queries = [
        'what are documents u have list it"',
        "what are documents u have list it",
        "list my documents",
        "what documents do I have?",
        "show my uploaded documents",
        "which documents have I uploaded?",
        "show uploaded files",
        "what files do you have",
        "which documents are available",
        "list files",
    ]
    for q in queries:
        assert classify(q) == Route.DOCUMENT_LIST, f"Query '{q}' failed to route to DOCUMENT_LIST"

@pytest.mark.asyncio
async def test_list_user_documents(db_session) -> None:
    """Requirement 14, Test 1: List 3 user documents."""
    user_id = uuid.uuid4()
    user = User(id=user_id, email=f"user-{user_id}@example.com", is_active=True)
    db_session.add(user)
    
    session_id = uuid.uuid4()
    chat_session = ChatSession(id=session_id, user_id=user_id, title="Test Session")
    db_session.add(chat_session)
    
    docs = [
        Document(id=uuid.uuid4(), user_id=user_id, title="Human_Resources_Report.pdf", status=DocumentStatus.READY, storage_path="p1"),
        Document(id=uuid.uuid4(), user_id=user_id, title="Finance_Assessment.pdf", status=DocumentStatus.READY, storage_path="p2"),
        Document(id=uuid.uuid4(), user_id=user_id, title="Product_Requirements.docx", status=DocumentStatus.READY, storage_path="p3"),
    ]
    db_session.add_all(docs)
    await db_session.commit()

    service = RAGService(db_session)
    response = await service.ask(session_id, "what documents do I have?")

    assert "Human_Resources_Report.pdf" in response.answer
    assert "Finance_Assessment.pdf" in response.answer
    assert "Product_Requirements.docx" in response.answer
    assert response.sources == []
    assert response.model == "database-direct"

@pytest.mark.asyncio
async def test_user_isolation(db_session) -> None:
    """Requirement 14, Test 2: User A cannot see User B's documents."""
    user_a = User(id=uuid.uuid4(), email=f"usera-{uuid.uuid4()}@example.com", is_active=True)
    user_b = User(id=uuid.uuid4(), email=f"userb-{uuid.uuid4()}@example.com", is_active=True)
    db_session.add_all([user_a, user_b])
    
    session_a = ChatSession(id=uuid.uuid4(), user_id=user_a.id, title="Session A")
    session_b = ChatSession(id=uuid.uuid4(), user_id=user_b.id, title="Session B")
    db_session.add_all([session_a, session_b])
    
    doc_a = Document(id=uuid.uuid4(), user_id=user_a.id, title="UserA_Secret.pdf", status=DocumentStatus.READY, storage_path="pa")
    doc_b = Document(id=uuid.uuid4(), user_id=user_b.id, title="UserB_Secret.pdf", status=DocumentStatus.READY, storage_path="pb")
    db_session.add_all([doc_a, doc_b])
    await db_session.commit()

    service = RAGService(db_session)
    response_a = await service.ask(session_a.id, "list my documents")

    assert "UserA_Secret.pdf" in response_a.answer
    assert "UserB_Secret.pdf" not in response_a.answer

@pytest.mark.asyncio
async def test_processing_documents_in_list(db_session) -> None:
    """Requirement 14, Test 4: Documents with status PROCESSING appear with indicator."""
    user_id = uuid.uuid4()
    user = User(id=user_id, email=f"user-{user_id}@example.com", is_active=True)
    db_session.add(user)
    
    session_id = uuid.uuid4()
    chat_session = ChatSession(id=session_id, user_id=user_id, title="Test Session")
    db_session.add(chat_session)
    
    doc = Document(id=uuid.uuid4(), user_id=user_id, title="Large_Archive.pdf", status=DocumentStatus.PROCESSING, storage_path="p_proc")
    db_session.add(doc)
    await db_session.commit()

    service = RAGService(db_session)
    response = await service.ask(session_id, "show my uploaded documents")

    assert "Large_Archive.pdf" in response.answer
    assert "Processing" in response.answer

@pytest.mark.asyncio
async def test_no_documents(db_session) -> None:
    """Requirement 14, Test 5: Zero documents uploaded message."""
    user_id = uuid.uuid4()
    user = User(id=user_id, email=f"user-{user_id}@example.com", is_active=True)
    db_session.add(user)
    
    session_id = uuid.uuid4()
    chat_session = ChatSession(id=session_id, user_id=user_id, title="Test Session")
    db_session.add(chat_session)
    await db_session.commit()

    service = RAGService(db_session)
    response = await service.ask(session_id, "what documents do I have?")

    assert "currently have no documents uploaded" in response.answer.lower()
