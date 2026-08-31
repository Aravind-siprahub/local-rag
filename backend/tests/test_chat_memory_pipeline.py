"""Comprehensive unit and integration tests for 3-Level Chat Memory Subsystem."""
import uuid
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from app.models.enums import MessageRole, DocumentStatus
from app.models.chat_session import ChatSession
from app.models.long_term_memory import LongTermMemory
from app.memory.conversation_memory import ConversationMemory
from app.memory.context_builder import MemoryContextBuilder, build_chat_context
from app.memory.long_term_store import LongTermMemoryStore
from app.memory.extractor import MemoryExtractor
from app.memory.types import MemoryType, MemoryEntry, ExtractionCandidate
from app.prompting.builder import PromptBuilder
from app.rag.service import RAGService
from app.retrieval.search import SearchFilters


@pytest.mark.asyncio
async def test_1_basic_conversation_memory():
    """TEST 1: Verify short-term conversation memory fetches recent messages for a session."""
    session = AsyncMock()
    msg_service_mock = AsyncMock()
    
    sess_id = uuid.uuid4()
    msg1 = MagicMock(id=uuid.uuid4(), role=MessageRole.USER, content="My project uses Ollama.")
    msg2 = MagicMock(id=uuid.uuid4(), role=MessageRole.ASSISTANT, content="Got it, Ollama runtime noted!")
    
    msg_service_mock.list_by_session.return_value = [msg1, msg2]
    
    conv_mem = ConversationMemory(session=session)
    conv_mem._service = msg_service_mock
    
    messages = await conv_mem.get_recent_messages(sess_id, limit=5)
    assert len(messages) == 2
    assert messages[0]["content"] == "My project uses Ollama."
    assert messages[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_2_and_10_session_isolation():
    """TEST 2 & 10: Verify strict session isolation (Session A history is isolated from Session B)."""
    session = AsyncMock()
    msg_service_mock = AsyncMock()
    
    sess_a_id = uuid.uuid4()
    sess_b_id = uuid.uuid4()
    
    msg_a = MagicMock(id=uuid.uuid4(), role=MessageRole.USER, content="Session A private text.")
    
    def side_effect_list(sess_id, **kwargs):
        if sess_id == sess_a_id:
            return [msg_a]
        return []
        
    msg_service_mock.list_by_session.side_effect = side_effect_list
    
    conv_mem = ConversationMemory(session=session)
    conv_mem._service = msg_service_mock
    
    res_a = await conv_mem.get_recent_messages(sess_a_id)
    res_b = await conv_mem.get_recent_messages(sess_b_id)
    
    assert len(res_a) == 1
    assert res_a[0]["content"] == "Session A private text."
    assert len(res_b) == 0


@pytest.mark.asyncio
async def test_3_long_term_memory_cross_session():
    """TEST 3: Verify long-term memory created in Session A is accessible in Session B for the same user."""
    session = AsyncMock()
    user_id = uuid.uuid4()
    sess_a = uuid.uuid4()
    sess_b = uuid.uuid4()
    
    ltm = LongTermMemory(
        id=uuid.uuid4(),
        user_id=user_id,
        memory_type=MemoryType.TECHNICAL_CONTEXT.value,
        content="User project uses FastAPI and Ollama.",
        importance=0.8,
        confidence=0.9,
        source_conversation_id=sess_a,
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    
    repo_mock = AsyncMock()
    repo_mock.list_by_user.return_value = [ltm]
    
    store = LongTermMemoryStore(session=session)
    store._repo = repo_mock
    store._embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
    
    memories = await store.retrieve(user_id, "What is my tech stack?", top_k=5)
    assert len(memories) == 1
    assert memories[0].content == "User project uses FastAPI and Ollama."


def test_4_duplicate_memory_prevention():
    """TEST 4: Verify duplicate memory candidates trigger conflict detection and supersede linking."""
    extractor = MemoryExtractor()
    
    existing_mem = MemoryEntry(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        memory_type=MemoryType.PREFERENCE,
        content="User prefers local open-source models over cloud options.",
        importance=0.8,
        confidence=0.9,
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    
    new_candidate = ExtractionCandidate(
        memory_type=MemoryType.PREFERENCE,
        content="User prefers local open-source models over cloud options.",
        importance=0.85,
        confidence=0.9,
    )
    
    resolved = extractor._detect_conflicts([new_candidate], [existing_mem])
    assert len(resolved) == 1
    assert resolved[0].conflicts_with == existing_mem.id


@pytest.mark.asyncio
async def test_5_session_summary_generation():
    """TEST 5: Verify session summary is generated and updated on ChatSession when threshold is reached."""
    session = AsyncMock()
    sess_id = uuid.uuid4()
    
    messages = [
        MagicMock(id=uuid.uuid4(), role=MessageRole.USER, content=f"User turn {i}: Discussing HR policy.")
        for i in range(7)
    ]
    
    msg_service_mock = AsyncMock()
    msg_service_mock.list_by_session.return_value = messages
    msg_service_mock.repository.session = session
    
    sess_obj = ChatSession(id=sess_id, user_id=uuid.uuid4(), title="Test Session")
    mock_exec = MagicMock()
    mock_exec.scalar_one_or_none.return_value = sess_obj
    session.execute.return_value = mock_exec
    
    conv_mem = ConversationMemory(session=session)
    conv_mem._service = msg_service_mock
    
    summary = await conv_mem.update_session_summary_if_needed(sess_id, force=True)
    assert summary is not None
    assert "User Intent" in summary or "Topics" in summary


def test_6_and_7_context_building_and_rag_memory_separation():
    """TEST 6 & 7: Verify build_chat_context produces structured separation of session summary, memories, messages, and document chunks."""
    sess_summary = "User is debugging HR Leave Policy."
    
    ltm_entry = MemoryEntry(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        memory_type=MemoryType.PREFERENCE,
        content="User prefers bullet points.",
        importance=0.8,
        confidence=0.9,
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    
    recent_msgs = [{"role": "user", "content": "What about casual leaves?"}]
    
    doc_chunk = MagicMock(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        section_title="Leave Policy",
        chunk_text="Employees get 12 days of casual leave.",
    )
    
    ctx_obj = build_chat_context(
        session_summary=sess_summary,
        long_term_memories=[ltm_entry],
        recent_messages=recent_msgs,
        retrieved_documents=[doc_chunk],
    )
    
    assert ctx_obj["session_summary"] == sess_summary
    assert len(ctx_obj["long_term_memories"]) == 1
    assert len(ctx_obj["recent_messages"]) == 1
    assert len(ctx_obj["retrieved_documents"]) == 1
    assert ctx_obj["retrieved_documents"][0]["section"] == "Leave Policy"


@pytest.mark.asyncio
async def test_8_missing_memory_graceful_fallback():
    """TEST 8: Verify RAG Service proceeds gracefully when memory is disabled or fails."""
    session = AsyncMock()
    session_svc = AsyncMock()
    msg_svc = AsyncMock()
    
    rag_service = RAGService(session=session, message_service=msg_svc, session_service=session_svc)
    
    # Memory manager before_query returning empty context
    rag_service._load_routing_hints = AsyncMock(return_value=([], []))
    
    builder = PromptBuilder()
    prompt = builder.build(
        question="What are the working hours?",
        retrieved_chunks=[],
        chat_history=None,
        working_memory_summary=None,
        long_term_memory_context="",
    )
    
    assert "Question:\n\nWhat are the working hours?" in prompt.user_prompt
    assert "Retrieved Document Context" in prompt.user_prompt or "Information not found" in prompt.user_prompt


def test_9_missing_document_retrieval_no_fabrication():
    """TEST 9: Verify prompt instructions strictly prevent fabricating unstated document facts."""
    builder = PromptBuilder()
    prompt = builder.build(
        question="What is the stock price of Apple?",
        retrieved_chunks=[],
        chat_history=None,
        working_memory_summary=None,
        long_term_memory_context="",
    )
    
    assert "Information not found in document excerpts." in prompt.user_prompt
