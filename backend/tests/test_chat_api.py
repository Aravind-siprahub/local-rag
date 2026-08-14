"""API tests for `app/api/endpoints/chat.py`."""
import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_chat_message_service, get_chat_session_service, get_rag_service
from app.main import app
from app.models.enums import MessageRole
from app.rag.response import RAGResponse, RAGTokenUsage, SourceCitation
from app.rag.service import RAGError


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


from typing import Generator

@pytest.fixture(autouse=True)
def clear_overrides() -> Generator[None, None, None]:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


class FakeRAGService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def ask(self, session_id: uuid.UUID, question: str, **kwargs) -> RAGResponse:
        self.calls.append({"session_id": session_id, "question": question, **kwargs})
        chunk_id = uuid.uuid4()
        return RAGResponse(
            answer="Revenue grew 12%.",
            sources=[
                SourceCitation(
                    chunk_id=chunk_id,
                    chunk_text="Revenue grew 12% in Q3.",
                    document_id=uuid.uuid4(),
                    document_version_id=uuid.uuid4(),
                    similarity_score=0.91,
                    rank=1,
                )
            ],
            token_usage=RAGTokenUsage(prompt_tokens=40, completion_tokens=15, total_tokens=55),
            model="test-chat-model",
            processing_time_ms=250,
            user_message_id=uuid.uuid4(),
            assistant_message_id=uuid.uuid4(),
        )

    async def close(self) -> None:
        return None


class FakeChatSession:
    def __init__(self, session_id: uuid.UUID, user_id: uuid.UUID) -> None:
        self.id = session_id
        self.user_id = user_id
        self.title = "Test chat"
        self.is_archived = False
        self.last_message_at = datetime.now(timezone.utc)
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.deleted_at = None


class FakeChatSessionService:
    def __init__(self, session: FakeChatSession) -> None:
        self.session = session

    async def get(self, session_id: uuid.UUID) -> FakeChatSession:
        if session_id != self.session.id:
            raise KeyError(session_id)
        return self.session


class FakeChatMessage:
    def __init__(self, session_id: uuid.UUID) -> None:
        self.id = uuid.uuid4()
        self.session_id = session_id
        self.role = MessageRole.USER
        self.content = "What is revenue?"
        self.model_used = None
        self.prompt_tokens = None
        self.completion_tokens = None
        self.total_tokens = None
        self.latency_ms = None
        self.generation_time_ms = None
        self.error_message = None
        self.created_at = datetime.now(timezone.utc)


class FakeChatMessageService:
    def __init__(self, session_id: uuid.UUID) -> None:
        self.session_id = session_id
        self.messages = [FakeChatMessage(session_id)]

    async def list_by_session(self, session_id: uuid.UUID, **kwargs) -> list[FakeChatMessage]:
        if session_id != self.session_id:
            return []
        return self.messages


def _setup_overrides(session_id: uuid.UUID, user_id: uuid.UUID) -> FakeRAGService:
    from app.api.dependencies import get_current_user
    from app.models.user import User
    fake_rag = FakeRAGService()
    app.dependency_overrides[get_rag_service] = lambda: fake_rag
    app.dependency_overrides[get_chat_session_service] = lambda: FakeChatSessionService(
        FakeChatSession(session_id, user_id)
    )
    app.dependency_overrides[get_chat_message_service] = lambda: FakeChatMessageService(session_id)
    app.dependency_overrides[get_current_user] = lambda: User(
        id=user_id, email="test@example.com", hashed_password="pwd", is_active=True
    )
    return fake_rag


class TestChatAPI:
    def test_post_chat_returns_rag_response(self, client: TestClient) -> None:
        session_id = uuid.uuid4()
        document_id = uuid.uuid4()
        fake_rag = _setup_overrides(session_id, uuid.uuid4())

        response = client.post(
            "/chat",
            json={
                "session_id": str(session_id),
                "question": "What is revenue?",
                "document_id": str(document_id),
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Revenue grew 12%."
        assert body["model"] == "test-chat-model"
        assert body["processing_time_ms"] == 250
        assert body["token_usage"]["prompt_tokens"] == 40
        assert body["token_usage"]["completion_tokens"] == 15
        assert len(body["citations"]) == 1
        assert body["citations"][0]["rank"] == 1
        assert fake_rag.calls[0]["filters"].document_id == document_id

    def test_post_chat_rejects_blank_question(self, client: TestClient) -> None:
        session_id = uuid.uuid4()
        _setup_overrides(session_id, uuid.uuid4())

        response = client.post(
            "/chat",
            json={"session_id": str(session_id), "question": "   "},
        )

        assert response.status_code == 422

    def test_post_chat_maps_rag_error_to_422(self, client: TestClient) -> None:
        session_id = uuid.uuid4()

        class FailingRAG:
            async def ask(self, *args, **kwargs):
                raise RAGError("Question must not be empty.")

            async def close(self):
                return None

        user_id = uuid.uuid4()
        _setup_overrides(session_id, user_id)

        app.dependency_overrides[get_rag_service] = lambda: FailingRAG()

        response = client.post(
            "/chat",
            json={"session_id": str(session_id), "question": "valid?"},
        )

        assert response.status_code == 400

    def test_get_chat_session(self, client: TestClient) -> None:
        session_id = uuid.uuid4()
        user_id = uuid.uuid4()
        _setup_overrides(session_id, user_id)

        response = client.get(f"/chat/sessions/{session_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == str(session_id)
        assert body["user_id"] == str(user_id)
        assert body["title"] == "Test chat"

    def test_get_chat_session_messages(self, client: TestClient) -> None:
        session_id = uuid.uuid4()
        _setup_overrides(session_id, uuid.uuid4())

        response = client.get(f"/chat/sessions/{session_id}/messages")

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert len(body["items"]) == 1
        assert body["items"][0]["content"] == "What is revenue?"
        assert body["items"][0]["role"] == "user"

    def test_openapi_lists_chat_paths(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        paths = schema["paths"]

        assert "/chat" in paths
        assert "post" in paths["/chat"]
        assert "/chat/sessions/{session_id}" in paths
        assert "get" in paths["/chat/sessions/{session_id}"]
        assert "/chat/sessions/{session_id}/messages" in paths
        assert "get" in paths["/chat/sessions/{session_id}/messages"]

        post_schema = paths["/chat"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
        assert "$ref" in post_schema
        assert post_schema["$ref"].endswith("ChatResponse")

    def test_post_chat_multipart_form_success(self, client: TestClient) -> None:
        session_id = uuid.uuid4()
        fake_rag = _setup_overrides(session_id, uuid.uuid4())
        
        # 1x1 transparent PNG bytes
        png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc`00\x00\x00\x00\x00\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
        
        response = client.post(
            "/chat",
            data={
                "session_id": str(session_id),
                "question": "What is in this report screenshot?",
            },
            files={
                "file": ("screenshot.png", png_bytes, "image/png")
            }
        )
        
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Revenue grew 12%."
        assert len(fake_rag.calls) == 1
        assert fake_rag.calls[0]["question"] == "What is in this report screenshot?"
        assert fake_rag.calls[0]["image"] == png_bytes
        assert fake_rag.calls[0]["image_name"] == "screenshot.png"
        assert fake_rag.calls[0]["image_mime"] == "image/png"

    def test_post_chat_multipart_form_invalid_format(self, client: TestClient) -> None:
        session_id = uuid.uuid4()
        _setup_overrides(session_id, uuid.uuid4())
        
        response = client.post(
            "/chat",
            data={
                "session_id": str(session_id),
                "question": "What is in this text file?",
            },
            files={
                "file": ("notes.txt", b"plain text", "text/plain")
            }
        )
        
        assert response.status_code == 400
        assert "Unsupported file extension" in response.json()["detail"]
