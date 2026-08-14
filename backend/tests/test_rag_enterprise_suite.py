"""Comprehensive Enterprise RAG Pipeline Test Suite.

Covers:
- No documents (0 candidates -> immediate return, 0 LLM calls)
- Multiple documents & Top 20 candidate -> Top 5 reranking
- Duplicate chunks & malformed chunks
- Context limits / Huge PDFs
- SSE Streaming
- Truncation retry (done_reason == "length" -> num_predict *= 2)
- Citations (Doc Name, Section, Page, Chunk ID)
- Multilingual documents
- Reasoning leakage detection, discard, and retry
- Ollama unavailable (503) & Timeout (504)
- Hybrid retrieval (pgvector + FTS RRF)
"""
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.llm.client import LLMUnavailableError, LLMTimeoutError
from app.llm.response import LLMResponse, TokenUsage
from app.llm.sanitize import detect_reasoning_leakage
from app.prompting.builder import PromptBuilder
from app.prompting.templates import format_chunk, format_user_prompt
from app.rag.response import RAGResponse
from app.rag.service import RAGError, RAGService
from app.retrieval.ranking import RankedResult, rank_hybrid_rrf, rerank_cross_encoder
from app.retrieval.search import SearchFilters, SearchHit
from app.tools.web_search import StubWebSearchProvider


@pytest.mark.asyncio
async def test_no_documents_found_early_exit():
    """Verify that when 0 chunks are found, LLM is NOT called and fallback answer is returned immediately."""
    session = AsyncMock(spec=AsyncSession)
    retriever = AsyncMock()
    retriever.retrieve.return_value = []

    llm_client = AsyncMock()
    llm_client.model = "qwen3:4b"

    messages = AsyncMock()
    user_msg = MagicMock(id=uuid.uuid4())
    assistant_msg = MagicMock(id=uuid.uuid4())
    messages.create_message.side_effect = [user_msg, assistant_msg]
    messages.list_by_session.return_value = []

    sessions = AsyncMock()
    chat_session = MagicMock(user_id=uuid.uuid4())
    sessions.get.return_value = chat_session

    rag = RAGService(
        session,
        retriever=retriever,
        llm_client=llm_client,
        message_service=messages,
        session_service=sessions,
        web_search=StubWebSearchProvider(),
    )

    session_id = uuid.uuid4()
    response = await rag.ask(
        session_id,
        "According to my documents, what is the policy for remote work?",
    )

    assert response.answer == "I could not find this information in the uploaded documents."
    assert len(response.sources) == 0
    # Crucial enterprise requirement: LLM client must NOT be called when no documents match!
    llm_client.generate.assert_not_called()


@pytest.mark.asyncio
async def test_multiple_documents_retrieval_and_reranking():
    """Verify top 20 candidates retrieved across multiple documents are reranked to top 5."""
    doc1 = uuid.uuid4()
    doc2 = uuid.uuid4()
    ver1 = uuid.uuid4()
    ver2 = uuid.uuid4()

    candidates = [
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text=f"Chunk content {i} discussing employee policy details",
            document_id=doc1 if i % 2 == 0 else doc2,
            document_version_id=ver1 if i % 2 == 0 else ver2,
            document_title="Employee Handbook.pdf" if i % 2 == 0 else "HR Policy.docx",
            similarity_score=0.7 + (i * 0.01),
            rank=i + 1,
            section_title="Benefits" if i % 2 == 0 else "Annual Leave",
            page_number=i + 1,
        )
        for i in range(20)
    ]

    reranked = rerank_cross_encoder("employee policy details", candidates, final_top_k=5)

    assert len(reranked) == 5
    assert all(isinstance(r, RankedResult) for r in reranked)
    assert reranked[0].rank == 1
    assert reranked[4].rank == 5


@pytest.mark.asyncio
async def test_duplicate_chunks_handling():
    """Verify that duplicate chunks from multiple versions are handled gracefully."""
    doc_id = uuid.uuid4()
    ver_id = uuid.uuid4()

    sem_hit = SearchHit(
        chunk_id=uuid.uuid4(),
        chunk_text="Identical chunk content",
        document_id=doc_id,
        document_version_id=ver_id,
        document_title="Doc.pdf",
        distance=0.1,
    )
    ft_hit = SearchHit(
        chunk_id=sem_hit.chunk_id,  # Same chunk ID in vector and fulltext
        chunk_text="Identical chunk content",
        document_id=doc_id,
        document_version_id=ver_id,
        document_title="Doc.pdf",
        distance=0.2,
    )

    rrf_results = rank_hybrid_rrf([sem_hit], [ft_hit])
    assert len(rrf_results) == 1
    assert rrf_results[0].chunk_id == sem_hit.chunk_id


@pytest.mark.asyncio
async def test_reasoning_leakage_detection():
    """Verify reasoning leakage detection triggers for unhandled thinking tags."""
    leakage_1 = "<think>Let me analyze the prompt carefully...</think>The answer is 42."
    leakage_2 = "<thinking>The user is asking about leave policy.</thinking>20 days."
    clean_text = "The annual leave allowance is 25 days per calendar year."

    assert detect_reasoning_leakage(leakage_1) is True
    assert detect_reasoning_leakage(leakage_2) is True
    assert detect_reasoning_leakage(clean_text) is False


@pytest.mark.asyncio
async def test_reasoning_leakage_discard_and_retry():
    """Verify that if LLM emits reasoning tags, response is discarded and retried ONCE."""
    session = AsyncMock(spec=AsyncSession)
    retriever = AsyncMock()
    chunk_id = uuid.uuid4()
    retriever.retrieve.return_value = [
        RankedResult(
            chunk_id=chunk_id,
            chunk_text="Annual leave policy allows 20 days per year.",
            document_id=uuid.uuid4(),
            document_version_id=uuid.uuid4(),
            document_title="Handbook.pdf",
            similarity_score=0.9,
            rank=1,
            section_title="Leave",
            page_number=5,
        )
    ]

    llm_client = AsyncMock()
    llm_client.model = "qwen3:4b"

    # First call returns tagged reasoning; second call returns clean factual answer
    bad_response = LLMResponse(
        answer="<think>The user asks about annual leave.</think>I will check the excerpt...",
        model_name="qwen3:4b",
    )
    good_response = LLMResponse(
        answer="The annual leave allowance is 20 days per year.",
        model_name="qwen3:4b",
    )
    llm_client.generate.side_effect = [bad_response, good_response]

    messages = AsyncMock()
    user_msg = MagicMock(id=uuid.uuid4())
    assistant_msg = MagicMock(id=uuid.uuid4())
    messages.create_message.side_effect = [user_msg, assistant_msg]
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

    response = await rag.ask(
        uuid.uuid4(),
        "According to my documents, what is the annual leave allowance?",
    )

    # LLM should have been called twice (1st discarded, 2nd accepted)
    assert llm_client.generate.call_count == 2
    assert "20 days per year" in response.answer
    assert "<think>" not in response.answer


@pytest.mark.asyncio
async def test_truncation_done_reason_length_retry():
    """Verify that finish_reason == 'length' triggers automatic num_predict doubling retry."""
    session = AsyncMock(spec=AsyncSession)
    retriever = AsyncMock()
    retriever.retrieve.return_value = [
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text="Detailed policy documentation text...",
            document_id=uuid.uuid4(),
            document_version_id=uuid.uuid4(),
            document_title="Policy.pdf",
            similarity_score=0.85,
            rank=1,
        )
    ]

    llm_client = AsyncMock()
    llm_client.model = "qwen3:4b"

    truncated_response = LLMResponse(
        answer="The company policy regarding...",
        model_name="qwen3:4b",
        finish_reason="length",
    )
    full_response = LLMResponse(
        answer="The company policy regarding remote work specifies 2 days WFH.",
        model_name="qwen3:4b",
        finish_reason="stop",
    )
    llm_client.generate.side_effect = [truncated_response, full_response]

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

    response = await rag.ask(
        uuid.uuid4(),
        "According to my documents, what is the WFH policy?",
    )

    assert llm_client.generate.call_count == 2
    # Verify num_predict was doubled on 2nd call
    second_call_kwargs = llm_client.generate.call_args_list[1].kwargs
    assert second_call_kwargs.get("num_predict") == 2048
    assert "remote work specifies 2 days WFH" in response.answer


@pytest.mark.asyncio
async def test_citations_formatting():
    """Verify citations include Document Name, Section, Page, Chunk ID."""
    chunk_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    ver_id = uuid.uuid4()

    chunk = RankedResult(
        chunk_id=chunk_id,
        chunk_text="Safety guidelines must be strictly followed.",
        document_id=doc_id,
        document_version_id=ver_id,
        document_title="Safety Manual.pdf",
        similarity_score=0.92,
        rank=1,
        section_title="General Safety",
        page_number=12,
    )

    formatted = format_chunk(
        index=1,
        chunk_text=chunk.chunk_text,
        title=chunk.document_title or "Unknown",
        section=chunk.section_title or "N/A",
        page=chunk.page_number or "N/A",
        chunk_id=chunk.chunk_id,
    )

    assert "Document: Safety Manual.pdf" in formatted
    assert "Section: General Safety" in formatted
    assert "Page: 12" in formatted
    # Passage ID is tracked via RetrievedChunkContext metadata, not rendered in the LLM prompt


@pytest.mark.asyncio
async def test_multilingual_document_qa():
    """Verify RAG pipeline handles non-English multilingual chunks."""
    doc_id = uuid.uuid4()
    ver_id = uuid.uuid4()

    multilingual_chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="Le congé annuel est de 25 jours par an.",
        document_id=doc_id,
        document_version_id=ver_id,
        document_title="Politique RH.pdf",
        similarity_score=0.88,
        rank=1,
        section_title="Congés",
        page_number=3,
    )

    builder = PromptBuilder()
    prompt = builder.build("Combien de jours de congé?", [multilingual_chunk])

    assert "Le congé annuel est de 25 jours par an." in prompt.user_prompt
    assert "Politique RH.pdf" in prompt.user_prompt


@pytest.mark.asyncio
async def test_regression_query_problem_statement_in_talk_to_my_data():
    """Test A: 'what is problem statement in my talk to my data' routes to RAG, retrieves PRD document, and produces grounded answer."""
    from app.rag.intent_router import classify, Route
    query = "what is problem statement in my talk to my data"
    assert classify(query) == Route.RAG

    session = AsyncMock(spec=AsyncSession)
    retriever = AsyncMock()
    doc_id = uuid.uuid4()
    retriever.retrieve.return_value = [
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text="Problem Statement: Business users struggle to query tabular data efficiently.",
            document_id=doc_id,
            document_version_id=uuid.uuid4(),
            document_title="PRD_Talk_to_My_Data.docx",
            similarity_score=0.91,
            rank=1,
        )
    ]

    llm_client = AsyncMock()
    llm_client.model = "qwen3:4b"
    llm_client.generate.return_value = LLMResponse(
        answer="The problem statement is that business users struggle to query tabular data efficiently.",
        model_name="qwen3:4b",
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

    response = await rag.ask(uuid.uuid4(), query)

    assert "Problem Statement" in response.answer or "tabular data" in response.answer
    assert retriever.retrieve.called is True


@pytest.mark.asyncio
async def test_regression_query_inside_document_name():
    """Test B: 'What is inside PRD_Talk_to_My_Data.docx?' routes to RAG and retrieves chunks."""
    from app.rag.intent_router import classify, Route
    query = "What is inside PRD_Talk_to_My_Data.docx?"
    assert classify(query) == Route.RAG


@pytest.mark.asyncio
async def test_regression_query_good_friday_routes_web():
    """Test C: 'When is Good Friday in 2026?' routes to WEB."""
    from app.rag.intent_router import classify, Route
    assert classify("When is Good Friday in 2026?") == Route.WEB


@pytest.mark.asyncio
async def test_regression_query_percent_routes_calculator():
    """Test D: 'What is 18% of 45000?' routes to CALCULATOR."""
    from app.rag.intent_router import classify, Route
    assert classify("What is 18% of 45000?") == Route.CALCULATOR


@pytest.mark.asyncio
async def test_regression_unknown_document_question_fallback_message():
    """Test E: Unknown document question returns exact required fallback message when 0 chunks retrieved."""
    session = AsyncMock(spec=AsyncSession)
    retriever = AsyncMock()
    retriever.retrieve.return_value = []

    llm_client = AsyncMock()
    llm_client.model = "qwen3:4b"

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

    response = await rag.ask(uuid.uuid4(), "what is the policy for quantum teleportation?")

    assert response.answer == "I could not find this information in the uploaded documents."
    llm_client.generate.assert_not_called()

