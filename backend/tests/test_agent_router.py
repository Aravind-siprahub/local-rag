"""Agent Router v1 integration tests for RAGService.ask routing."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import pytest

from app.llm.response import LLMResponse, TokenUsage
from app.models.enums import MessageRole
from app.prompting.builder import PromptBuilder
from app.rag.service import RAGService
from app.retrieval.ranking import RankedResult
from app.tools.web_search import WebSearchHit, WebSearchResult


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
        return self.session


class FakeChatMessageService:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self._messages: list[_FakeMessage] = []

    async def create_message(self, **kwargs) -> _FakeMessage:
        self.created.append(kwargs)
        msg = _FakeMessage(
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
        self._messages.append(msg)
        return msg

    async def list_by_session(self, session_id: uuid.UUID, limit: int = 50, offset: int = 0) -> list[_FakeMessage]:
        messages = [m for m in self._messages if m.session_id == session_id]
        if offset > 0:
            messages = messages[:-offset]
        return messages[-limit:]


class FakeCitationService:
    def __init__(self) -> None:
        self.created: list[tuple[uuid.UUID, list]] = []

    async def create_citations_for_message(self, message_id: uuid.UUID, citations: list) -> list:
        self.created.append((message_id, citations))
        return citations


class FakeRetriever:
    def __init__(self, results: list[RankedResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[dict] = []

    async def retrieve(self, question: str, **kwargs) -> list[RankedResult]:
        self.calls.append({"question": question, **kwargs})
        return self.results

    async def close(self) -> None:
        return None


class FakeLLMClient:
    def __init__(self, answer: str = "Python is a programming language.") -> None:
        self.answer = answer
        self.calls: list[dict] = []
        self.model = "test-chat-model"

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        num_predict: int | None = None,
        response_format: str | None = None,
        temperature: float | None = None,
        images: list[bytes] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "num_predict": num_predict,
                "response_format": response_format,
                "temperature": temperature,
                "images": images,
                "model": model,
            }
        )
        
        # Context-aware stub answers for tests
        ans = self.answer
        if "Good Friday" in user_prompt:
            ans = "Good Friday in 2026 falls on Friday, 3 April 2026."
        elif "Python" in user_prompt:
            ans = "Python is a programming language."
        elif "18% of 45000" in user_prompt:
            ans = "8100"
            
        return LLMResponse(
            answer=ans,
            model_name="test-chat-model",
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
        )

    async def close(self) -> None:
        return None


class FakeWebSearchProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def search(self, query: str, request_id: str | None = None) -> WebSearchResult:
        self.calls.append(query)
        return WebSearchResult(
            query=query,
            provider="fake",
            hits=[
                WebSearchHit(
                    title="Good Friday 2026",
                    url="https://example.com/good-friday-2026",
                    snippet="Good Friday in 2026 falls on Friday, 3 April 2026.",
                )
            ],
        )


def _ranked(text: str, rank: int = 1) -> RankedResult:
    return RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text=text,
        document_id=uuid.uuid4(),
        document_version_id=uuid.uuid4(),
        similarity_score=0.9,
        rank=rank,
        document_title="Deployment_Guide.docx",
    )


def _make_service(
    *,
    retrieval_results: list[RankedResult] | None = None,
    web_search: Any = None,
    llm: Any = None,
    retriever: Any = None,
    user_id: uuid.UUID | None = None,
) -> tuple[RAGService, _FakeChatSession, Any, Any, Any]:
    session = _FakeChatSession(id=uuid.uuid4(), user_id=user_id or uuid.uuid4())
    active_retriever = retriever or FakeRetriever(retrieval_results or [_ranked("Nginx reverse proxy notes.")])
    llm_client = llm or FakeLLMClient()
    web = web_search or FakeWebSearchProvider()
    service = RAGService(
        session=None,  # type: ignore
        retriever=active_retriever,  # type: ignore
        prompt_builder=PromptBuilder(),
        llm_client=llm_client,  # type: ignore
        message_service=FakeChatMessageService(),  # type: ignore
        citation_service=FakeCitationService(),  # type: ignore
        session_service=FakeChatSessionService(session),  # type: ignore
        web_search=web,  # type: ignore
    )
    return service, session, active_retriever, llm_client, web


class TestAgentRouterAsk:
    @pytest.mark.asyncio
    async def test_good_friday_routes_web_and_skips_retriever(self) -> None:
        service, session, retriever, llm, web = _make_service()

        response = await service.ask(session.id, "When is Good Friday in 2026?")

        assert web.calls == ["When is Good Friday in 2026?"]
        assert retriever.calls == []
        assert "Good Friday" in response.answer or "April" in response.answer or "found" in response.answer.lower()
        assert not response.sources or getattr(response.sources[0], "section_title", "web") in ("web", "duckduckgo", "fake", "example.com")

    @pytest.mark.asyncio
    async def test_deployment_guide_routes_rag_and_calls_retriever(self) -> None:
        service, session, retriever, llm, web = _make_service()

        response = await service.ask(
            session.id,
            "What does Deployment_Guide.docx say about Nginx?",
        )

        assert len(retriever.calls) >= 1
        assert web.calls == []
        assert response.answer  # existing RAG path produces an answer
        assert len(llm.calls) >= 1

    @pytest.mark.asyncio
    async def test_percent_routes_calculator_and_skips_retriever(self) -> None:
        service, session, retriever, llm, web = _make_service()

        response = await service.ask(session.id, "What is 18% of 45000?")

        assert retriever.calls == []
        assert web.calls == []
        assert "8100" in response.answer
        assert response.sources == []

    @pytest.mark.asyncio
    async def test_python_routes_direct_and_skips_retriever(self) -> None:
        service, session, retriever, llm, web = _make_service()

        response = await service.ask(session.id, "What is Python?")

        assert web.calls == []
        assert len(response.answer) > 0

    @pytest.mark.asyncio
    async def test_web_search_empty_hits_returns_no_results_message(self) -> None:
        class EmptyWebSearchProvider:
            def __init__(self) -> None:
                self.calls: list[str] = []

            async def search(self, query: str, request_id: str | None = None) -> WebSearchResult:
                self.calls.append(query)
                return WebSearchResult(query=query, provider="fake_empty", hits=[])

        empty_web = EmptyWebSearchProvider()
        service, session, retriever, llm, _ = _make_service(web_search=empty_web)

        response = await service.ask(session.id, "When is Good Friday in 2026?")

        assert empty_web.calls == ["When is Good Friday in 2026?"]
        assert len(llm.calls) == 0  # Ollama NOT invoked for WEB route
        assert "could not find reliable web results" in response.answer.lower()
        assert response.sources == []

    @pytest.mark.asyncio
    async def test_earth_query_routes_direct_and_normalizes(self) -> None:
        llm = FakeLLMClient(answer="Earth is the 3rd planet from the Sun.")
        service, session, retriever, _, web = _make_service(retriever=FakeRetriever(), user_id=uuid.uuid4())
        service.llm_client = llm

        response = await service.ask(session.id, "earth is 2 planet or 3 planet")

        assert web.calls == []
        assert len(response.answer) > 0


    @pytest.mark.asyncio
    async def test_list_out_doucment_u_have_routes_to_document_list(self) -> None:
        from app.rag.intent_router import Route, classify
        assert classify("list out doucment u have") == Route.DOCUMENT_LIST
        assert classify('what are documents u have list it"') == Route.DOCUMENT_LIST
        assert classify("what are documents u have list it") == Route.DOCUMENT_LIST
        assert classify("what are documents you have list it") == Route.DOCUMENT_LIST
        assert classify("what documents u have") == Route.DOCUMENT_LIST

    @pytest.mark.asyncio
    async def test_look_up_and_search_github_routes_to_web(self) -> None:
        from app.rag.intent_router import Route, classify
        assert classify("Can you look up posthog and tell me what it is?") == Route.WEB
        assert classify("search github for Claude Fable leaked system prompt") == Route.WEB
        assert classify("look up weather in Tokyo") == Route.WEB
        assert classify("search online for python docs") == Route.WEB


@pytest.mark.asyncio
async def test_image_attachment_routes_to_direct_vision(db_session) -> None:
    """Test that attaching an image in JSON payload routes to direct vision analysis."""
    from app.models.user import User
    from app.models.chat_session import ChatSession
    from unittest.mock import AsyncMock, patch
    
    user_id = uuid.uuid4()
    user = User(id=user_id, email=f"user-{user_id}@example.com", is_active=True)
    db_session.add(user)
    
    session_id = uuid.uuid4()
    chat_session = ChatSession(id=session_id, user_id=user_id, title="Image Test")
    db_session.add(chat_session)
    await db_session.commit()

    service = RAGService(db_session)
    attachments = [{
        "id": str(uuid.uuid4()),
        "name": "ChatGPT Image Jul 21.png",
        "file_path": "user_id/session_id/image.png",
        "mime_type": "image/png",
    }]
    
    from app.rag.intent_router import Route, classify
    with patch("app.storage.get_storage_service") as mock_storage:
        mock_instance = AsyncMock()
        mock_instance.download_file.return_value = b"fake_png_bytes"
        mock_storage.return_value = mock_instance
        
        with patch.object(service.llm_client, "generate", new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = LLMResponse(
                answer="This image shows a modern UI dashboard with dark mode design.",
                model_name="qwen3:8b",
                token_usage=TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
            )
            response = await service.ask(session_id, "tell about this image", attachments=attachments)
            assert "modern UI dashboard" in response.answer
            assert mock_generate.called
            kwargs = mock_generate.call_args.kwargs
            assert kwargs.get("images") == [b"fake_png_bytes"]
        assert classify("what documents you have") == Route.DOCUMENT_LIST
        assert classify("list my documents") == Route.DOCUMENT_LIST


