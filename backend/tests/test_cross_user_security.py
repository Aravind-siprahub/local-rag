"""Automated cross-user access and JWT authentication security test suite.

Proves:
1. No authentication -> 401 Unauthorized
2. Invalid token -> 401 Unauthorized
3. Expired token -> 401 Unauthorized
4. Valid user token -> authenticates correct user identity
5. User A cannot access User B's document -> 403 Forbidden
6. User A cannot retrieve User B's document through RAG (0 hits)
7. User A cannot access User B's chat sessions or messages -> 403 Forbidden
8. User A cannot access User B's citations -> 403 Forbidden
9. User A cannot access User B's execution traces
10. Supplying another user's ID in X-User-Id header cannot bypass authentication
11. Supplying another user's ID in user_id query parameter cannot bypass authentication
"""
from __future__ import annotations

import uuid
from datetime import timedelta
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.security import create_access_token, hash_password, verify_password
from app.core.config import Settings
from app.main import app
from app.models.enums import MessageRole, UserRole
from app.repositories.chat_message_repository import ChatMessageRepository
from app.repositories.chat_session_repository import ChatSessionRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.user_repository import UserRepository
from app.retrieval.retriever import Retriever
from app.retrieval.search import SearchFilters
from app.services.trace_service import TraceStore


def test_password_hashing_properties() -> None:
    """Password hash is salted PBKDF2-HMAC-SHA256, not plaintext, different from original, and unique per hash."""
    plain = "SuperSecretPassword123!"
    hashed1 = hash_password(plain)
    hashed2 = hash_password(plain)

    # Never store plaintext passwords
    assert plain not in hashed1
    assert hashed1.startswith("$pbkdf2-sha256$100000$")
    assert hashed1 != plain

    # Random salt produces different hashes for identical passwords
    assert hashed1 != hashed2

    # Verification
    assert verify_password(plain, hashed1) is True
    assert verify_password(plain, hashed2) is True
    assert verify_password("WrongPassword123!", hashed1) is False
    assert verify_password("", hashed1) is False


@pytest.mark.asyncio
async def test_successful_login_with_correct_password(db_session) -> None:
    """Correct email and password returns 200 OK and a valid JWT access token."""
    user_repo = UserRepository(db_session)
    password = "MySecurePassword2026!"
    user = await user_repo.create(
        email="login_success@example.com",
        hashed_password=hash_password(password),
        role=UserRole.MEMBER,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/api/auth/login",
            json={"email": user.email, "password": password},
        )
        assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
        data = res.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["email"] == user.email


@pytest.mark.asyncio
async def test_login_rejected_with_incorrect_password(db_session) -> None:
    """Incorrect password returns 401 Unauthorized."""
    user_repo = UserRepository(db_session)
    user = await user_repo.create(
        email="login_fail@example.com",
        hashed_password=hash_password("RightPassword123!"),
        role=UserRole.MEMBER,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/api/auth/login",
            json={"email": user.email, "password": "WrongPassword123!"},
        )
        assert res.status_code == 401
        assert "Authentication failed" in res.json().get("detail", "")


@pytest.mark.asyncio
async def test_login_rejected_with_empty_password(db_session) -> None:
    """Empty password returns 401 Unauthorized."""
    user_repo = UserRepository(db_session)
    user = await user_repo.create(
        email="empty_pass@example.com",
        hashed_password=hash_password("ValidPassword123!"),
        role=UserRole.MEMBER,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/api/auth/login",
            json={"email": user.email, "password": ""},
        )
        assert res.status_code == 401
        assert "Password is required" in res.json().get("detail", "")


def test_jwt_secret_production_validation() -> None:
    """Settings validator rejects weak or missing secrets in production environment."""
    from pydantic import ValidationError

    with pytest.raises((ValueError, ValidationError)):
        Settings(
            ENVIRONMENT="production",
            JWT_SECRET_KEY="secret",  # weak/short secret in production fails fast
            DATABASE_URL="postgresql+psycopg://user:pass@localhost:5432/db",
        )


@pytest.mark.asyncio
async def test_unauthenticated_request_returns_401(db_session) -> None:
    """Request without Authorization Bearer header returns 401 Unauthorized."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/documents")
        assert res.status_code == 401
        assert "Authentication required" in res.json().get("detail", "")


@pytest.mark.asyncio
async def test_invalid_token_returns_401(db_session) -> None:
    """Request with malformed or tampered JWT returns 401 Unauthorized."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get(
            "/api/documents",
            headers={"Authorization": "Bearer invalid.tampered.token"},
        )
        assert res.status_code == 401
        assert "Authentication failed" in res.json().get("detail", "")


@pytest.mark.asyncio
async def test_expired_token_returns_401(db_session) -> None:
    """Request with an expired JWT token returns 401 Unauthorized."""
    user_repo = UserRepository(db_session)
    user = await user_repo.create(email="exp_user@example.com", hashed_password="pwd", role=UserRole.MEMBER)

    # Expired token (-1 hour)
    expired_token = create_access_token(user.id, expires_delta=timedelta(hours=-1))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get(
            "/api/documents",
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert res.status_code == 401
        assert "expired" in res.json().get("detail", "").lower()


@pytest.mark.asyncio
async def test_valid_token_authenticates_correct_user(db_session) -> None:
    """Valid JWT token authenticates as the correct user identity."""
    user_repo = UserRepository(db_session)
    user = await user_repo.create(email="valid_user@example.com", hashed_password="pwd", role=UserRole.MEMBER)

    token = create_access_token(user.id)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 200
        assert res.json().get("id") == str(user.id)
        assert res.json().get("email") == "valid_user@example.com"


@pytest.mark.asyncio
async def test_x_user_id_header_cannot_bypass_authentication(db_session) -> None:
    """Supplying X-User-Id header without valid Bearer token returns 401 Unauthorized."""
    user_repo = UserRepository(db_session)
    user_b = await user_repo.create(email="target_b@example.com", hashed_password="pwd", role=UserRole.MEMBER)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Client tries to impersonate User B via X-User-Id header
        res = await client.get(
            "/api/documents",
            headers={"X-User-Id": str(user_b.id)},
        )
        assert res.status_code == 401, f"Expected 401, got {res.status_code}"


@pytest.mark.asyncio
async def test_query_parameter_user_id_cannot_bypass_authentication(db_session) -> None:
    """Supplying ?user_id= without valid Bearer token returns 401 Unauthorized."""
    user_repo = UserRepository(db_session)
    user_b = await user_repo.create(email="target_query_b@example.com", hashed_password="pwd", role=UserRole.MEMBER)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get(f"/api/documents?user_id={user_b.id}")
        assert res.status_code == 401, f"Expected 401, got {res.status_code}"


@pytest.mark.asyncio
async def test_user_a_cannot_read_user_b_document(db_session) -> None:
    """User A using User A's token cannot read User B's document (returns 403)."""
    user_repo = UserRepository(db_session)
    user_a = await user_repo.create(email="usera_doc_jwt@example.com", hashed_password="pwd", role=UserRole.MEMBER)
    user_b = await user_repo.create(email="userb_doc_jwt@example.com", hashed_password="pwd", role=UserRole.MEMBER)

    token_a = create_access_token(user_a.id)

    doc_repo = DocumentRepository(db_session)
    doc_b = await doc_repo.create(user_id=user_b.id, title="User B Confidential.pdf", status="ready")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # User A tries to GET User B's document
        res = await client.get(
            f"/api/documents/{doc_b.id}",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert res.status_code == 403, f"Expected 403 Forbidden, got {res.status_code}: {res.text}"

        # User A attempts to list documents passing user_id=User_B_ID
        res_list = await client.get(
            f"/api/documents?user_id={user_b.id}",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert res_list.status_code == 403, f"Expected 403 Forbidden for cross-user document list, got {res_list.status_code}"


@pytest.mark.asyncio
async def test_user_a_cannot_retrieve_user_b_document_through_rag(db_session) -> None:
    """Retriever scoped with SearchFilters(user_id=User_A) produces 0 hits for User B's documents."""
    user_repo = UserRepository(db_session)
    user_a = await user_repo.create(email="usera_rag_jwt@example.com", hashed_password="pwd", role=UserRole.MEMBER)
    user_b = await user_repo.create(email="userb_rag_jwt@example.com", hashed_password="pwd", role=UserRole.MEMBER)

    doc_repo = DocumentRepository(db_session)
    doc_b = await doc_repo.create(user_id=user_b.id, title="Secret Project B", status="ready")

    retriever = Retriever(db_session)
    filters_a = SearchFilters(user_id=user_a.id)

    # Retrieval executed for User A
    hits = await retriever.retrieve("Secret Project B", filters=filters_a)
    doc_ids_retrieved = [h.document_id for h in hits]

    assert doc_b.id not in doc_ids_retrieved, "User A retrieved User B's document via RAG search!"


@pytest.mark.asyncio
async def test_user_a_cannot_read_user_b_chat_session(db_session) -> None:
    """User A using User A's token cannot read User B's chat sessions or messages (returns 403)."""
    user_repo = UserRepository(db_session)
    user_a = await user_repo.create(email="usera_chat_jwt@example.com", hashed_password="pwd", role=UserRole.MEMBER)
    user_b = await user_repo.create(email="userb_chat_jwt@example.com", hashed_password="pwd", role=UserRole.MEMBER)

    token_a = create_access_token(user_a.id)

    session_repo = ChatSessionRepository(db_session)
    session_b = await session_repo.create(user_id=user_b.id, title="User B Private Chat")

    msg_repo = ChatMessageRepository(db_session)
    msg_b = await msg_repo.create(session_id=session_b.id, role=MessageRole.USER, content="User B private query")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # User A tries to GET User B's chat session
        res_sess = await client.get(
            f"/api/chat-sessions/{session_b.id}",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert res_sess.status_code == 403, f"Expected 403 Forbidden for chat session, got {res_sess.status_code}"

        # User A tries to GET User B's chat messages
        res_msgs = await client.get(
            f"/api/chat-messages?session_id={session_b.id}",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert res_msgs.status_code == 403, f"Expected 403 Forbidden for chat messages, got {res_msgs.status_code}"


@pytest.mark.asyncio
async def test_user_a_cannot_read_user_b_citations(db_session) -> None:
    """User A cannot read citations belonging to User B's session messages (returns 403)."""
    user_repo = UserRepository(db_session)
    user_a = await user_repo.create(email="usera_cite_jwt@example.com", hashed_password="pwd", role=UserRole.MEMBER)
    user_b = await user_repo.create(email="userb_cite_jwt@example.com", hashed_password="pwd", role=UserRole.MEMBER)

    token_a = create_access_token(user_a.id)

    session_repo = ChatSessionRepository(db_session)
    session_b = await session_repo.create(user_id=user_b.id, title="User B Session")

    msg_repo = ChatMessageRepository(db_session)
    msg_b = await msg_repo.create(session_id=session_b.id, role=MessageRole.ASSISTANT, content="Answer for User B")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get(
            f"/api/chat-messages/{msg_b.id}",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert res.status_code == 403, f"Expected 403 Forbidden for User B message/citation, got {res.status_code}"


@pytest.mark.asyncio
async def test_user_a_cannot_read_user_b_traces(db_session) -> None:
    """TraceStore enforces user scoping so User A cannot retrieve User B's RAG traces."""
    user_repo = UserRepository(db_session)
    user_a = await user_repo.create(email="usera_trace_jwt@example.com", hashed_password="pwd", role=UserRole.MEMBER)
    user_b = await user_repo.create(email="userb_trace_jwt@example.com", hashed_password="pwd", role=UserRole.MEMBER)

    trace_store = TraceStore(db_session)
    req_id_b = f"req-{uuid.uuid4()}"
    await trace_store.save_trace_safely(
        request_id=req_id_b,
        user_id=user_b.id,
        original_query="Secret user B query",
    )

    # User A tries to retrieve User B's trace
    trace_for_a = await trace_store.get_by_request_id_for_user(req_id_b, user_a.id)
    assert trace_for_a is None, "User A was able to read User B's RAG trace!"

    # User B retrieves their own trace
    trace_for_b = await trace_store.get_by_request_id_for_user(req_id_b, user_b.id)
    assert trace_for_b is not None
    assert trace_for_b.request_id == req_id_b
