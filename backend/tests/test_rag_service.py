"""Unit tests for `app.rag.service`."""
import uuid
from dataclasses import dataclass

import pytest

from app.models.enums import MessageRole
from app.llm.response import LLMResponse, TokenUsage
from app.prompting.builder import PromptBuilder
from app.rag.service import RAGError, RAGService
from app.retrieval.ranking import RankedResult
from app.retrieval.retriever import RetrievalError
from app.retrieval.search import SearchFilters


@dataclass
class _FakeChatSession:
    id: uuid.UUID
    user_id: uuid.UUID


@dataclass
class _FakeMessage:
    id: uuid.UUID
    session_id: uuid.UUID
    role: MessageRole
    content: str
    model_used: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None
    generation_time_ms: int | None = None


class FakeChatSessionService:
    def __init__(self, session: _FakeChatSession) -> None:
        self.session = session

    async def get(self, session_id: uuid.UUID) -> _FakeChatSession:
        if session_id != self.session.id:
            raise KeyError(session_id)
        return self.session


class FakeChatMessageService:
    def __init__(self) -> None:
        self.created: list[dict] = []

    async def create_message(self, **kwargs) -> _FakeMessage:
        self.created.append(kwargs)
        return _FakeMessage(
            id=uuid.uuid4(),
            session_id=kwargs["session_id"],
            role=kwargs["role"],
            content=kwargs["content"],
            model_used=kwargs.get("model_used"),
            prompt_tokens=kwargs.get("prompt_tokens"),
            completion_tokens=kwargs.get("completion_tokens"),
            latency_ms=kwargs.get("latency_ms"),
            generation_time_ms=kwargs.get("generation_time_ms"),
        )


class FakeCitationService:
    def __init__(self) -> None:
        self.created: list[tuple[uuid.UUID, list]] = []

    async def create_citations_for_message(self, message_id: uuid.UUID, citations: list) -> list:
        self.created.append((message_id, citations))
        return citations


class FakeRetriever:
    def __init__(self, results: list[RankedResult] | None = None, *, fail: bool = False) -> None:
        self.results = results or []
        self.fail = fail
        self.calls: list[dict] = []

    async def retrieve(self, question: str, **kwargs) -> list[RankedResult]:
        self.calls.append({"question": question, **kwargs})
        if self.fail:
            raise RetrievalError("simulated retrieval failure")
        return self.results

    async def close(self) -> None:
        return None


class FakeLLMClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls.append((system_prompt, user_prompt))
        return LLMResponse(
            answer="The revenue grew 12%.",
            model_name="test-chat-model",
            token_usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
            finish_reason="stop",
        )

    async def close(self) -> None:
        return None


def _ranked(text: str, rank: int) -> RankedResult:
    return RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text=text,
        document_id=uuid.uuid4(),
        document_version_id=uuid.uuid4(),
        similarity_score=0.92,
        rank=rank,
    )


def _make_service(
    *,
    retriever: FakeRetriever | None = None,
    retrieval_results: list[RankedResult] | None = None,
    user_id: uuid.UUID | None = None,
) -> tuple[RAGService, _FakeChatSession, FakeChatMessageService, FakeCitationService, FakeRetriever, FakeLLMClient]:
    session = _FakeChatSession(id=uuid.uuid4(), user_id=user_id or uuid.uuid4())
    messages = FakeChatMessageService()
    citations = FakeCitationService()
    fake_retriever = retriever or FakeRetriever(retrieval_results or [_ranked("Revenue up 12%.", 1)])
    llm = FakeLLMClient()
    service = RAGService(
        session=None,
        retriever=fake_retriever,
        prompt_builder=PromptBuilder(),
        llm_client=llm,
        message_service=messages,
        citation_service=citations,
        session_service=FakeChatSessionService(session),
    )
    return service, session, messages, citations, fake_retriever, llm


class TestRAGService:
    @pytest.mark.asyncio
    async def test_end_to_end_flow(self) -> None:
        service, session, messages, citations, retriever, llm = _make_service()

        response = await service.ask(session.id, "What happened to revenue?")

        assert response.answer == "The revenue grew 12%."
        assert response.model == "test-chat-model"
        assert response.processing_time_ms >= 0
        assert response.token_usage is not None
        assert response.token_usage.prompt_tokens == 50
        assert response.token_usage.completion_tokens == 20
        assert len(response.sources) == 1
        assert len(messages.created) == 2
        assert messages.created[0]["role"] == MessageRole.USER
        assert messages.created[0]["content"] == "What happened to revenue?"
        assert messages.created[1]["role"] == MessageRole.ASSISTANT
        assert messages.created[1]["model_used"] == "test-chat-model"
        assert messages.created[1]["prompt_tokens"] == 50
        assert messages.created[1]["completion_tokens"] == 20
        assert len(citations.created) == 1
        assert citations.created[0][1][0]["rank"] == 1
        assert len(llm.calls) == 1
        assert retriever.calls[0]["filters"].user_id == session.user_id

    @pytest.mark.asyncio
    async def test_retrieval_failure_continues_without_sources(self) -> None:
        retriever = FakeRetriever(fail=True)
        service, session, messages, citations, _, llm = _make_service(retriever=retriever)

        response = await service.ask(session.id, "Any data?")

        assert response.answer == "The revenue grew 12%."
        assert response.sources == []
        assert citations.created == []
        assert len(messages.created) == 2
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_passes_document_filter_to_retriever(self) -> None:
        service, session, _, _, retriever, _ = _make_service()
        document_id = uuid.uuid4()
        filters = SearchFilters(document_id=document_id)

        await service.ask(session.id, "Scoped question?", filters=filters)

        assert retriever.calls[0]["filters"].document_id == document_id
        assert retriever.calls[0]["filters"].user_id == session.user_id

    @pytest.mark.asyncio
    async def test_rejects_empty_question(self) -> None:
        service, session, messages, _, _, _ = _make_service()

        with pytest.raises(RAGError, match="empty"):
            await service.ask(session.id, "   ")

        assert messages.created == []
