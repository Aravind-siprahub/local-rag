import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.response import LLMResponse, TokenUsage
from app.rag.service import RAGService
from app.retrieval.ranking import RankedResult
from app.tools.web_search import StubWebSearchProvider

@pytest.mark.asyncio
async def test_combines_frontend_backend_info():
    """Verify that frontend and backend details from separate chunks are combined and passed correctly without being dropped."""
    session = AsyncMock(spec=AsyncSession)
    
    # Mock routing hints (document titles)
    mock_result = MagicMock()
    mock_doc = MagicMock()
    mock_doc.title = "PRD_Talk_to_My_Data.docx"
    mock_result.scalars.return_value.all.return_value = [mock_doc]
    session.execute.return_value = mock_result
    
    retriever = AsyncMock()
    
    frontend_chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="Frontend — the chat interface; also renders citations, source snippets, and (for data questions) the generated SQL and any resulting...",
        document_id=uuid.uuid4(),
        document_version_id=uuid.uuid4(),
        document_title="PRD_Talk_to_My_Data.docx",
        similarity_score=0.85,
        rank=1,
        section_title="System Architecture Overview",
        page_number=5,
    )
    
    backend_chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="Backend — FastAPI handles API requests, orchestrates RAG, and manages PostgreSQL connections.",
        document_id=uuid.uuid4(),
        document_version_id=uuid.uuid4(),
        document_title="PRD_Talk_to_My_Data.docx",
        similarity_score=0.84,
        rank=2,
        section_title="System Architecture Overview",
        page_number=5,
    )
    
    # 10 duplicate noise chunks to simulate pre-deduplication array 
    noise_chunks = [
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text=f"Noise text {i}...",
            document_id=uuid.uuid4(),
            document_version_id=uuid.uuid4(),
            document_title="PRD_Talk_to_My_Data.docx",
            similarity_score=0.6,
            rank=i+3,
            section_title="Other",
            page_number=5,
        ) for i in range(10)
    ]
    
    # Send a mix of frontend, backend, and duplicates
    # Retriever returns retrieved_chunks to the service
    retriever.retrieve.return_value = [frontend_chunk, backend_chunk] + noise_chunks

    llm_client = AsyncMock()
    llm_client.model = "qwen3:4b"
    llm_client.generate.return_value = LLMResponse(
        answer="Talk to My Data uses a chat interface for the frontend and FastAPI for the backend.",
        model_name="qwen3:4b",
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    )

    messages = AsyncMock()
    messages.create_message.side_effect = [MagicMock(id=uuid.uuid4()), MagicMock(id=uuid.uuid4())]
    messages.list_by_session.return_value = []

    sessions = AsyncMock()
    sessions.get.return_value = MagicMock(user_id=uuid.uuid4())

    rag = RAGService(
        session,
        retriever=retriever,
        llm_client=llm_client,
        message_service=messages,
        session_service=sessions,
        web_search=StubWebSearchProvider(),
    )

    response = await rag.ask(uuid.uuid4(), "What frontend and backend are used in Talk to My Data?")
    
    # Assert LLM was called with both frontend and backend chunks in the prompt
    assert llm_client.generate.call_count == 1
    system_prompt, user_prompt = llm_client.generate.call_args[0]
    
    assert "Frontend \u2014 the chat interface" in user_prompt
    assert "Backend \u2014 FastAPI" in user_prompt
    
    # Make sure noise didn't truncate them away
    assert "not specified" not in response.answer.lower()
