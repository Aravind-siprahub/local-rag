"""Regression tests for the signup / login auth flow.

Covers:
- POST /users: valid payload -> 201, invalid password -> 422
- POST /auth/login: valid creds -> access_token, wrong pw -> 401
- GET /users: unauthenticated -> 401, authenticated -> 200
- GET /documents: unauthenticated -> 401, authenticated -> 200
"""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user, get_document_service, get_user_service
from app.api.security import create_access_token, hash_password
from app.main import app
from app.models.enums import UserRole
from app.models.user import User
from app.services.user_service import UserService

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

VALID_EMAIL    = "signup_reg@example.com"
VALID_PASSWORD = "Secure@Pass123"   # >=8, upper, lower, digit, special -> passes PasswordPolicy
VALID_NAME     = "Test Signup User"


def _make_user(
    email: str = VALID_EMAIL,
    password: str = VALID_PASSWORD,
    role: UserRole = UserRole.MEMBER,
) -> User:
    """Build an in-memory User ORM object with a real PBKDF2 hashed password."""
    now = datetime.now(timezone.utc)
    return User(
        id=uuid.uuid4(),
        email=email,
        hashed_password=hash_password(password),
        full_name=VALID_NAME,
        role=role,
        is_active=True,
        is_verified=False,
        deleted_at=None,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def clear_overrides():
    """Ensure dependency overrides never leak between tests."""
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /users  --  signup
# ---------------------------------------------------------------------------

class TestSignup:
    def test_valid_payload_returns_201(self, client: TestClient) -> None:
        """Full valid payload (email, password, full_name, role) -> 201."""
        mock_svc = AsyncMock(spec=UserService)
        mock_svc.create_user.return_value = _make_user()
        app.dependency_overrides[get_user_service] = lambda: mock_svc

        resp = client.post(
            "/users",
            json={
                "email": VALID_EMAIL,
                "password": VALID_PASSWORD,
                "full_name": VALID_NAME,
                "role": "member",
            },
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["email"] == VALID_EMAIL

    def test_password_missing_special_char_returns_422(self, client: TestClient) -> None:
        """Password without a special character fails PasswordPolicy -> 422."""
        resp = client.post(
            "/users",
            json={
                "email": VALID_EMAIL,
                "password": "NoSpecial1",   # missing !@#$ etc.
                "full_name": VALID_NAME,
            },
        )
        assert resp.status_code == 422

    def test_password_too_short_returns_422(self, client: TestClient) -> None:
        """Password shorter than 8 chars -> 422."""
        resp = client.post(
            "/users",
            json={"email": VALID_EMAIL, "password": "Ab1!", "full_name": VALID_NAME},
        )
        assert resp.status_code == 422

    def test_missing_email_returns_422(self, client: TestClient) -> None:
        """Omitting email entirely -> 422."""
        resp = client.post(
            "/users",
            json={"password": VALID_PASSWORD, "full_name": VALID_NAME},
        )
        assert resp.status_code == 422

    def test_invalid_email_format_returns_422(self, client: TestClient) -> None:
        """Malformed email -> 422."""
        resp = client.post(
            "/users",
            json={"email": "not-an-email", "password": VALID_PASSWORD, "full_name": VALID_NAME},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /auth/login
# ---------------------------------------------------------------------------

def _mock_repo(user):
    repo = AsyncMock()
    repo.get_by_email.return_value = user
    return repo


class TestLogin:
    def test_valid_credentials_return_token(self, client: TestClient) -> None:
        """Correct email + password -> 200 with access_token and bearer type."""
        user = _make_user()
        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_repo(user)):
            resp = client.post(
                "/auth/login",
                json={"email": VALID_EMAIL, "password": VALID_PASSWORD},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"
        assert body["user"]["email"] == VALID_EMAIL

    def test_wrong_password_returns_401(self, client: TestClient) -> None:
        """Correct email but wrong password -> 401."""
        user = _make_user()
        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_repo(user)):
            resp = client.post(
                "/auth/login",
                json={"email": VALID_EMAIL, "password": "WrongPass@1"},
            )
        assert resp.status_code == 401

    def test_nonexistent_user_returns_401(self, client: TestClient) -> None:
        """Unknown email -> 401 (not 404)."""
        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_repo(None)):
            resp = client.post(
                "/auth/login",
                json={"email": "ghost@example.com", "password": VALID_PASSWORD},
            )
        assert resp.status_code == 401

    def test_missing_password_returns_401(self, client: TestClient) -> None:
        """No password field -> 401 (backend explicitly rejects empty password)."""
        resp = client.post("/auth/login", json={"email": VALID_EMAIL})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /users  --  auth enforcement
# ---------------------------------------------------------------------------

class TestUsersAuth:
    def test_no_token_returns_401(self, client: TestClient) -> None:
        """GET /users without Authorization header -> 401."""
        resp = client.get("/users")
        assert resp.status_code == 401

    def test_invalid_token_returns_401(self, client: TestClient) -> None:
        """GET /users with a fake non-JWT token -> 401."""
        resp = client.get("/users", headers={"Authorization": "Bearer demo-access-token"})
        assert resp.status_code == 401

    def test_valid_token_returns_200(self, client: TestClient) -> None:
        """GET /users with a real signed JWT + mocked service -> 200."""
        user = _make_user()
        token = create_access_token(user.id)

        mock_svc = AsyncMock(spec=UserService)
        mock_svc.list.return_value = [user]
        mock_svc.count.return_value = 1

        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_user_service] = lambda: mock_svc

        resp = client.get("/users", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "items" in body
        assert body["total"] == 1


# ---------------------------------------------------------------------------
# GET /documents  --  auth enforcement
# ---------------------------------------------------------------------------

class TestDocumentsAuth:
    def test_no_token_returns_401(self, client: TestClient) -> None:
        """GET /documents without Authorization header -> 401."""
        resp = client.get("/documents")
        assert resp.status_code == 401

    def test_invalid_token_returns_401(self, client: TestClient) -> None:
        """GET /documents with a fake token -> 401."""
        resp = client.get("/documents", headers={"Authorization": "Bearer demo-access-token"})
        assert resp.status_code == 401

    def test_valid_token_returns_200(self, client: TestClient) -> None:
        """GET /documents with a real JWT + mocked service -> 200."""
        user = _make_user()
        token = create_access_token(user.id)

        mock_doc_svc = AsyncMock()
        mock_doc_svc.list_by_user.return_value = []

        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_document_service] = lambda: mock_doc_svc

        resp = client.get("/documents", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "items" in body
        assert body["total"] == 0
