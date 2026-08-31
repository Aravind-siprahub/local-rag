"""Unit tests for `app.rag.service`."""
import uuid
from dataclasses import dataclass
from typing import Any

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
        msgs = [m for m in self._messages if m.session_id == session_id]
        if offset:
            msgs = msgs[offset:]
        return msgs[:limit]


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
    def __init__(self, vision_support: bool = True) -> None:
        self.calls: list[tuple[str, str, list[bytes] | None, str | None]] = []
        self.model = "test-chat-model"
        self._vision_support = vision_support

    async def supports_vision(self, model: str | None = None) -> bool:
        return self._vision_support

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        num_predict: int | None = None,
        images: list[bytes] | None = None,
        response_format: str | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        self.calls.append((system_prompt, user_prompt, images, model))
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
        document_title="Revenue Report.docx",
    )


def _make_service(
    *,
    retriever: FakeRetriever | None = None,
    retrieval_results: list[RankedResult] | None = None,
    user_id: uuid.UUID | None = None,
    llm: FakeLLMClient | None = None,
) -> tuple[RAGService, _FakeChatSession, FakeChatMessageService, FakeCitationService, FakeRetriever, FakeLLMClient]:
    session = _FakeChatSession(id=uuid.uuid4(), user_id=user_id or uuid.uuid4())
    messages = FakeChatMessageService()
    citations = FakeCitationService()
    fake_retriever = retriever or FakeRetriever(retrieval_results or [_ranked("Revenue up 12%.", 1)])
    llm_client = llm or FakeLLMClient()
    from app.tools.web_search import StubWebSearchProvider

    service = RAGService(
        session=None,  # type: ignore
        retriever=fake_retriever,  # type: ignore
        prompt_builder=PromptBuilder(),
        llm_client=llm_client,  # type: ignore
        message_service=messages,  # type: ignore
        citation_service=citations,  # type: ignore
        session_service=FakeChatSessionService(session),  # type: ignore
        web_search=StubWebSearchProvider(),  # type: ignore
    )
    return service, session, messages, citations, fake_retriever, llm_client


class TestRAGService:
    @pytest.mark.asyncio
    async def test_end_to_end_flow(self) -> None:
        service, session, messages, citations, retriever, llm = _make_service()

        response = await service.ask(
            session.id,
            "According to my documents, what happened to revenue?",
        )

        assert response.answer.startswith("The revenue grew 12%.")
        assert response.model == "test-chat-model"
        assert response.processing_time_ms >= 0
        assert response.token_usage is not None
        assert response.token_usage.prompt_tokens == 50
        assert response.token_usage.completion_tokens == 20
        assert len(response.sources) == 1
        assert len(messages.created) == 2
        assert messages.created[0]["role"] == MessageRole.USER
        assert messages.created[0]["content"] == "According to my documents, what happened to revenue?"
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

        try:
            res = await service.ask(session.id, "According to my documents, any data?")
            assert len(res.sources) == 0
        except (RetrievalError, RAGError):
            pass

        assert citations.created == []
        assert len(llm.calls) == 0

    @pytest.mark.asyncio
    async def test_passes_document_filter_to_retriever(self) -> None:
        service, session, _, _, retriever, _ = _make_service()
        document_id = uuid.uuid4()
        filters = SearchFilters(document_id=document_id)

        await service.ask(
            session.id,
            "According to my documents, scoped question?",
            filters=filters,
        )

        assert retriever.calls[0]["filters"].document_id == document_id
        assert retriever.calls[0]["filters"].user_id == session.user_id

    @pytest.mark.asyncio
    async def test_rejects_empty_question(self) -> None:
        service, session, messages, _, _, _ = _make_service()

        with pytest.raises(RAGError, match="empty"):
            await service.ask(session.id, "   ")

        assert messages.created == []

    @pytest.mark.asyncio
    async def test_rejects_image_upload_if_vision_unsupported(self) -> None:
        # Mock LLM client with vision support disabled
        llm = FakeLLMClient(vision_support=False)
        service, session, messages, _, _, _ = _make_service(llm=llm)

        with pytest.raises(RAGError, match="does not support vision"):
            await service.ask(
                session.id,
                "What is in this image?",
                image=b"fakeimagebytes",
            )

        assert messages.created == []

    @pytest.mark.asyncio
    async def test_accepts_image_if_vision_supported_and_saves_attachments(self) -> None:
        llm = FakeLLMClient(vision_support=True)
        service, session, messages, _, _, _ = _make_service(llm=llm)

        image_bytes = b"fakeimagebytes"
        response = await service.ask(
            session.id,
            "What is in this image?",
            image=image_bytes,
            image_name="test_image.png",
            image_mime="image/png",
        )

        assert response.answer.startswith("The revenue grew 12%.")
        assert len(messages.created) == 2
        user_msg = messages.created[0]
        assert user_msg["role"] == MessageRole.USER
        assert user_msg["content"] == "What is in this image?"
        
        # Check attachments metadata was created
        assert user_msg["attachments"] is not None
        assert len(user_msg["attachments"]) == 1
        att = user_msg["attachments"][0]
        assert att["filename"] == "test_image.png"
        assert att["mime_type"] == "image/png"
        assert att["size"] == len(image_bytes)

        # Check images argument and vision model override were passed to LLM client generate call
        assert len(llm.calls) == 1
        assert llm.calls[0][2] == [image_bytes]
        from app.core.config import get_settings
        assert llm.calls[0][3] == get_settings().ollama_vision_model

    @pytest.mark.asyncio
    async def test_empty_question_defaults_to_describe_image(self) -> None:
        llm = FakeLLMClient(vision_support=True)
        service, session, messages, _, _, _ = _make_service(llm=llm)

        response = await service.ask(
            session.id,
            "",
            image=b"fakeimagebytes",
        )

        assert response.answer.startswith("The revenue grew 12%.")
        assert len(messages.created) == 2
        user_msg = messages.created[0]
        assert user_msg["content"] == "Describe this image."
        assert len(user_msg["attachments"]) == 1

    @pytest.mark.asyncio
    async def test_downloads_image_from_storage_path(self, monkeypatch: Any) -> None:
        llm = FakeLLMClient(vision_support=True)
        service, session, messages, _, _, _ = _make_service(llm=llm)

        download_calls: list[str] = []

        async def mock_download_file(self_storage: Any, storage_path: str) -> bytes:
            download_calls.append(storage_path)
            return b"downloadedbytes"

        monkeypatch.setattr("app.storage.supabase_storage_service.SupabaseStorageService.download_file", mock_download_file)
        monkeypatch.setattr("app.storage.s3_storage_service.S3StorageService.download_file", mock_download_file)

        response = await service.ask(
            session.id,
            "What is in this image?",
            image_storage_path="user1/session1/img.png",
            image_name="test_image.png",
            image_mime="image/png",
            image_size=1234,
        )

        assert response.answer.startswith("The revenue grew 12%.")
        assert len(messages.created) == 2

        user_msg = messages.created[0]
        assert user_msg["attachments"] is not None
        assert len(user_msg["attachments"]) == 1
        att = user_msg["attachments"][0]
        assert att["storage_path"] == "user1/session1/img.png"
        assert att["filename"] == "test_image.png"
        assert att["size"] == 1234

        assert download_calls == ["user1/session1/img.png"]

        assert len(llm.calls) == 1
        assert llm.calls[0][2] == [b"downloadedbytes"]
