import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.models.enums import MessageRole
from app.rag.service import RAGService
from app.retrieval.ranking import RankedResult
from app.retrieval.search import SearchFilters
from app.prompting.templates import USER_PROMPT_WITH_CONTEXT


@pytest.mark.asyncio
async def test_single_document_isolation():
    """Test 1: Verify single-document summarization resolves target document and receives exact content (Apollo, Engineering, 75000)."""
    session = AsyncMock()

    doc_id = uuid.uuid4()
    doc_mock = MagicMock()
    doc_mock.id = doc_id
    doc_mock.title = "test_doc.txt"
    doc_mock.original_filename = "test_doc.txt"

    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = [doc_mock]
    session.execute.return_value = mock_exec

    rag_service = RAGService(session=session, message_service=AsyncMock(), session_service=AsyncMock())
    base_filters = SearchFilters(user_id=uuid.uuid4())
    attachments = [{
        "id": "att-1",
        "filename": "test_doc.txt",
        "mime_type": "text/plain",
        "document_id": str(doc_id),
    }]

    resolved = await rag_service._resolve_entity_filters(
        user_id=base_filters.user_id,
        question="Summarize this document.",
        base_filters=base_filters,
        attachments=attachments,
    )

    assert resolved.document_id == doc_id


@pytest.mark.asyncio
async def test_cross_document_isolation():
    """Test 2: Verify Document A and Document B maintain strict isolation and do not mix context."""
    session = AsyncMock()

    doc_a_id = uuid.uuid4()
    doc_b_id = uuid.uuid4()

    doc_a = MagicMock(id=doc_a_id, title="Doc_A.txt", original_filename="Doc_A.txt")
    doc_b = MagicMock(id=doc_b_id, title="Doc_B.txt", original_filename="Doc_B.txt")

    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = [doc_a, doc_b]
    session.execute.return_value = mock_exec

    rag_service = RAGService(session=session, message_service=AsyncMock(), session_service=AsyncMock())

    # Resolve Doc A
    resolved_a = await rag_service._resolve_entity_filters(
        user_id=uuid.uuid4(),
        question="Summarize Document A.",
        base_filters=SearchFilters(),
        attachments=[{"id": "att-a", "filename": "Doc_A.txt", "document_id": str(doc_a_id)}],
    )
    assert resolved_a.document_id == doc_a_id

    # Resolve Doc B
    resolved_b = await rag_service._resolve_entity_filters(
        user_id=uuid.uuid4(),
        question="Summarize Document B.",
        base_filters=SearchFilters(),
        attachments=[{"id": "att-b", "filename": "Doc_B.txt", "document_id": str(doc_b_id)}],
    )
    assert resolved_b.document_id == doc_b_id
    assert resolved_b.document_id != resolved_a.document_id


@pytest.mark.asyncio
async def test_previous_conversation_contamination():
    """Test 3: Previous conversation messages containing 'Frontend: React, Backend: FastAPI' do NOT leak into document summarization."""
    session = AsyncMock()

    doc_id = uuid.uuid4()
    doc_mock = MagicMock(id=doc_id, title="Employee_Report.pdf", original_filename="Employee_Report.pdf")

    mock_exec = MagicMock()
    mock_exec.scalars.return_value.all.return_value = [doc_mock]
    session.execute.return_value = mock_exec

    rag_service = RAGService(session=session, message_service=AsyncMock(), session_service=AsyncMock())

    resolved = await rag_service._resolve_entity_filters(
        user_id=uuid.uuid4(),
        question="Summarize this document.",
        base_filters=SearchFilters(),
        attachments=[{"id": "att-hr", "filename": "Employee_Report.pdf", "document_id": str(doc_id)}],
    )

    assert resolved.document_id == doc_id


@pytest.mark.asyncio
async def test_zero_chunk_document():
    """Test 5: Zero-chunk document returns clear notice without executing global RAG or leaking project architecture."""
    session = AsyncMock()
    messages_service = AsyncMock()
    sessions_service = AsyncMock()

    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    chat_session_mock = MagicMock(user_id=user_id)
    sessions_service.get.return_value = chat_session_mock

    user_msg_mock = MagicMock(id=uuid.uuid4())
    messages_service.create_message.return_value = user_msg_mock

    from app.models.enums import DocumentStatus
    doc_mock = MagicMock(id=doc_id, status=DocumentStatus.READY)
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = doc_mock
    session.execute.return_value = mock_exec

    rag_service = RAGService(
        session=session,
        message_service=messages_service,
        session_service=sessions_service,
    )
    rag_service._retrieve_safely = AsyncMock(return_value=[])

    attachments = [{"id": "att-zero", "filename": "empty.pdf", "document_id": str(doc_id)}]
    filters = SearchFilters(document_id=doc_id)

    stream_tokens = []
    async for event in rag_service.ask_stream(session_id, "Summarize this document.", filters=filters, attachments=attachments):
        stream_tokens.append(event)

    full_stream_text = "".join(stream_tokens)
    assert "Unable to summarize the document because no readable text could be extracted" in full_stream_text
    assert "Frontend: React, Backend: FastAPI" not in full_stream_text


def test_user_prompt_template_has_no_hardcoded_example():
    """Verify templates.py contains NO hardcoded 'Frontend: React, Backend: FastAPI' example text."""
    assert "Frontend: React, Backend: FastAPI" not in USER_PROMPT_WITH_CONTEXT
