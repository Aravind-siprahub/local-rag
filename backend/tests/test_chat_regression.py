"""Regression tests for /api/chat flow and reasoning-sanitization failure paths."""
from __future__ import annotations

import uuid
import typing
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_rag_service
from app.llm.client import LLMAPIError, LLMClientError, LLMModelError, LLMTimeoutError
from app.llm.ollama_client import OllamaLLMClient, _parse_chat_response
from app.llm.sanitize import sanitize_response
from app.main import app
from app.rag.service import RAGService


class TestRAGServiceInit:
    def test_rag_service_imports_citation_service(self) -> None:
        """Regression: missing CitationService import caused HTTP 500 on POST /api/chat."""
        session = MagicMock()
        service = RAGService(session)
        assert service.citations is not None


class TestSanitizeEdgeCases:
    def test_none_and_missing_content_safe(self) -> None:
        assert sanitize_response(None) == ""

    def test_non_string_returns_empty(self) -> None:
        assert sanitize_response([]) == ""  # type: ignore[arg-type]


class TestOllamaParseEdgeCases:
    def test_missing_message_raises_api_error(self) -> None:
        with pytest.raises(LLMAPIError, match="message"):
            _parse_chat_response({"model": "test"}, fallback_model="test")

    def test_null_content_uses_fallback_answer(self) -> None:
        result = _parse_chat_response(
            {"model": "test", "message": {"role": "assistant", "content": None}},
            fallback_model="test",
        )
        assert result.answer == "Information not found in document excerpts."

    def test_empty_content_uses_fallback_answer(self) -> None:
        result = _parse_chat_response(
            {"model": "test", "message": {"role": "assistant", "content": "  "}},
            fallback_model="test",
        )
        assert result.answer == "Information not found in document excerpts."

    def test_thinking_field_is_never_used_as_answer(self) -> None:
        result = _parse_chat_response(
            {
                "model": "qwen3:8b",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "thinking": "Okay, let me reason about this...",
                },
            },
            fallback_model="qwen3:8b",
        )
        assert "Okay" not in result.answer
        assert result.answer == "Information not found in document excerpts."

    def test_non_dict_response_raises(self) -> None:
        with pytest.raises(LLMAPIError, match="JSON object"):
            _parse_chat_response([], fallback_model="test")  # type: ignore[arg-type]


class TestChatAPIErrorResponses:
    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(app)

    @pytest.fixture(autouse=True)
    def setup_overrides(self) -> typing.Generator[None, None, None]:
        from app.api.dependencies import get_current_user, get_chat_session_service
        from app.models.user import User
        
        user_id = uuid.uuid4()
        user = User(
            id=user_id, email="test@example.com", hashed_password="pwd", is_active=True
        )

        class FakeChatSession:
            def __init__(self, id, user_id):
                self.id = id
                self.user_id = user_id
                self.title = "Test chat"
                self.archived_at = None

        class FakeChatSessionService:
            def __init__(self):
                self.session = MagicMock()

            async def get(self, session_id):
                return FakeChatSession(session_id, user_id)

        app.dependency_overrides.clear()
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_chat_session_service] = lambda: FakeChatSessionService()
        yield
        app.dependency_overrides.clear()

    def test_llm_model_error_returns_json_not_generic_500(self, client: TestClient) -> None:
        class FailingRAG:
            async def ask(self, *args, **kwargs):
                raise LLMModelError("model 'bad' not found")

            async def close(self):
                return None

        app.dependency_overrides[get_rag_service] = lambda: FailingRAG()

        response = client.post(
            "/chat",
            json={"session_id": str(uuid.uuid4()), "question": "Hello?"},
        )

        assert response.status_code == 404
        body = response.json()
        assert body["error"] == "LLMModelError"
        assert "not found" in body["message"]

    def test_llm_timeout_returns_504_json(self, client: TestClient) -> None:
        class FailingRAG:
            async def ask(self, *args, **kwargs):
                raise LLMTimeoutError("Ollama request timed out after 300s.")

            async def close(self):
                return None

        app.dependency_overrides[get_rag_service] = lambda: FailingRAG()

        response = client.post(
            "/chat",
            json={"session_id": str(uuid.uuid4()), "question": "Hello?"},
        )

        assert response.status_code == 504
        assert response.json()["error"] == "LLMTimeoutError"

    def test_llm_api_error_returns_502_json(self, client: TestClient) -> None:
        class FailingRAG:
            async def ask(self, *args, **kwargs):
                raise LLMAPIError("Ollama returned HTTP 500")

            async def close(self):
                return None

        app.dependency_overrides[get_rag_service] = lambda: FailingRAG()

        response = client.post(
            "/chat",
            json={"session_id": str(uuid.uuid4()), "question": "Hello?"},
        )

        assert response.status_code == 502
        assert response.json()["error"] == "LLMAPIError"

    def test_llm_client_error_returns_400_json(self, client: TestClient) -> None:
        class FailingRAG:
            async def ask(self, *args, **kwargs):
                raise LLMClientError("user_prompt must not be empty.")

            async def close(self):
                return None

        app.dependency_overrides[get_rag_service] = lambda: FailingRAG()

        response = client.post(
            "/chat",
            json={"session_id": str(uuid.uuid4()), "question": "Hello?"},
        )

        assert response.status_code == 400
        assert response.json()["error"] == "LLMClientError"


@pytest.mark.asyncio
async def test_llama_model_does_not_send_think_param() -> None:
    capture: dict = {}

    def handler(request):
        import json
        capture["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"model": "llama3.2", "message": {"role": "assistant", "content": "Hi"}, "done": True},
        )

    import httpx

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaLLMClient(
            base_url="http://ollama.test",
            model="llama3.2",
            max_retries=0,
            client=http_client,
        )
        await client.generate("System.", "Question?")
        await client.close()

    assert "think" not in capture["payload"]
