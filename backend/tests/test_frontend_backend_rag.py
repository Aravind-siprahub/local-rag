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
    llm_client.model = "qwen3:8b"
    llm_client.generate.return_value = LLMResponse(
        answer="Talk to My Data uses a chat interface for the frontend and FastAPI for the backend.",
        model_name="qwen3:8b",
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


def test_architecture_chunk_ranks_above_deployment_chunk():
    """Verify that framework architecture chunks (React, FastAPI) rank above deployment port chunks (port 4173, PM2) for tech-stack queries."""
    from app.retrieval.ranking import RankedResult, _fallback_heuristic_rerank

    architecture_chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="Frontend: React with Vite. Backend: FastAPI.",
        document_id=uuid.uuid4(),
        similarity_score=0.75,
        rank=1,
        document_title="PRD_Talk_to_My_Data.docx",
        section_title="Architecture",
    )

    deployment_chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="The deployment document specifies that the frontend runs on port 4173 and backend on port 5000. Also, the system uses PM2 for process management.",
        document_id=uuid.uuid4(),
        similarity_score=0.78,
        rank=2,
        document_title="AIRIS_Staging_Deployment_Guide_Combined.docx",
        section_title="Deployment",
    )

    query = "what frontend and backend are using talk to my data"
    scored = _fallback_heuristic_rerank(query, [deployment_chunk, architecture_chunk])

    top_chunk = scored[0][1]
    assert top_chunk.chunk_id == architecture_chunk.chunk_id
    assert "React with Vite" in top_chunk.chunk_text
    assert "FastAPI" in top_chunk.chunk_text


def test_deployment_chunks_cannot_contaminate_architecture_query():
    """Verify post-reranking relevance filter excludes unrelated deployment port chunks from architecture prompts."""
    from app.rag.service import _filter_relevant_chunks
    from app.retrieval.ranking import RankedResult

    architecture_chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="Frontend: React with Vite. Backend: FastAPI.",
        document_id=uuid.uuid4(),
        similarity_score=1.1920,
        rank=1,
        document_title="PRD_Talk_to_My_Data.docx",
        section_title="Architecture",
    )

    deployment_chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="The deployment document specifies that the frontend runs on port 4173 and backend on port 5000. Also, the system uses PM2 for process management.",
        document_id=uuid.uuid4(),
        similarity_score=0.3420,
        rank=2,
        document_title="AIRIS_Staging_Deployment_Guide_Combined.docx",
        section_title="Deployment",
    )

    query = "what frontend and backend are using talk to my data"
    filtered = _filter_relevant_chunks(query, [architecture_chunk, deployment_chunk])

    assert len(filtered) == 1
    assert filtered[0].chunk_id == architecture_chunk.chunk_id
    assert "port 4173" not in filtered[0].chunk_text
    assert "PM2" not in filtered[0].chunk_text


def test_attribute_specific_no_cross_contamination():
    """Verify that tech stack queries and port queries isolate their requested attributes without cross-contaminating."""
    from app.rag.attribute_detector import detect_requested_attributes, RequestedAttribute
    from app.rag.service import _filter_relevant_chunks
    from app.retrieval.ranking import RankedResult, _fallback_heuristic_rerank

    architecture_chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="Frontend: React with Vite. Backend: FastAPI.",
        document_id=uuid.uuid4(),
        similarity_score=0.75,
        rank=1,
        document_title="PRD_Talk_to_My_Data.docx",
        section_title="Architecture",
    )

    port_chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="Frontend port: 4173. Backend port: 5000.",
        document_id=uuid.uuid4(),
        similarity_score=0.75,
        rank=2,
        document_title="AIRIS_Staging_Deployment_Guide_Combined.docx",
        section_title="Deployment",
    )

    # Question 1: Tech stack requested
    tech_query = "what frontend and backend are using talk to my data"
    tech_attrs = detect_requested_attributes(tech_query)
    assert RequestedAttribute.FRAMEWORK_TECH_STACK in tech_attrs
    assert RequestedAttribute.PORT_NETWORKING not in tech_attrs

    tech_filtered = _filter_relevant_chunks(tech_query, [architecture_chunk, port_chunk])
    assert len(tech_filtered) == 1
    assert tech_filtered[0].chunk_id == architecture_chunk.chunk_id

    # Question 2: Port requested
    port_query = "what ports does frontend/backend use"
    port_attrs = detect_requested_attributes(port_query)
    assert RequestedAttribute.PORT_NETWORKING in port_attrs
    assert RequestedAttribute.FRAMEWORK_TECH_STACK not in port_attrs

    port_filtered = _filter_relevant_chunks(port_query, [architecture_chunk, port_chunk])
    assert len(port_filtered) == 1
    assert port_filtered[0].chunk_id == port_chunk.chunk_id


@pytest.mark.asyncio
async def test_talk_to_my_data_vs_sipraone_project_entity_scoping():
    """Verify that 'talk to my data' query matches Talk-to-My-Data docs while 'sipraone' query matches SipraOne docs."""
    from app.models.document import Document
    from app.rag.service import RAGService
    from app.retrieval.search import SearchFilters

    ttmd_doc = MagicMock(spec=Document)
    ttmd_doc.id = uuid.uuid4()
    ttmd_doc.title = "PRD_Talk_to_My_Data.docx"

    sipra_doc = MagicMock(spec=Document)
    sipra_doc.id = uuid.uuid4()
    sipra_doc.title = "SipraOne_Frontend_Deployment_Guide.docx"

    session = AsyncMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [ttmd_doc, sipra_doc]
    retriever = AsyncMock()
    retriever.retrieve.return_value = []

    messages_mock = AsyncMock()
    messages_mock.create_message.return_value = MagicMock(id=uuid.uuid4())
    sessions_mock = AsyncMock()
    sessions_mock.get.return_value = MagicMock(user_id=uuid.uuid4())

    llm_mock = AsyncMock()
    llm_mock.generate.return_value = LLMResponse(
        answer="Talk to My Data uses React and FastAPI.",
        model_name="test-model",
    )

    rag = RAGService(
        session,
        retriever=retriever,
        llm_client=llm_mock,
        message_service=messages_mock,
        session_service=sessions_mock,
    )

    # 1. Talk to My Data query
    await rag.ask(uuid.uuid4(), "what frontend and backend are using talk to my data")
    ttmd_filters = retriever.retrieve.call_args[1]["filters"]
    assert ttmd_filters.document_ids == (ttmd_doc.id,)

    # 2. SipraOne query
    retriever.retrieve.reset_mock()
    await rag.ask(uuid.uuid4(), "what frontend and backend are use sipraone")
    sipra_filters = retriever.retrieve.call_args[1]["filters"]
    assert sipra_filters.document_ids == (sipra_doc.id,)



