"""Automated test suite for password reset / forgot password flow.

Tests:
1. Forgot password request for existing user -> sends email, returns 200 generic message
2. Forgot password request for non-existent user -> returns 200 generic message without error
3. Forgot password within 60s cooldown -> returns 429
4. Reset password with wrong OTP -> returns 400 Bad Request
5. Reset password with expired OTP -> returns 400 Bad Request
6. Reset password with weak password -> returns 422 Unprocessable Entity
7. Reset password with correct OTP & valid password -> returns 200, updates password, allows direct login
"""
import uuid
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.security import hash_password, verify_password
from app.db.session import get_db
from app.main import app
from app.models.enums import UserRole
from app.models.user import User
from app.services.email_service import hash_verification_otp

TEST_EMAIL = "reset_user@example.com"
OLD_PASSWORD = "OldPassword123!"
NEW_PASSWORD = "NewPassword456!"
TEST_OTP = "889900"


def _make_test_user(
    email: str = TEST_EMAIL,
    password: str = OLD_PASSWORD,
    is_verified: bool = True,
    otp: str | None = TEST_OTP,
    expires_in_minutes: int = 10,
    attempts: int = 0,
    last_otp_sent_at: datetime | None = None,
) -> User:
    now = datetime.now(timezone.utc)
    otp_hash = hash_verification_otp(otp) if otp else None
    exp_at = now + timedelta(minutes=expires_in_minutes) if expires_in_minutes else None

    return User(
        id=uuid.uuid4(),
        email=email,
        hashed_password=hash_password(password),
        full_name="Reset Flow User",
        role=UserRole.MEMBER,
        is_active=True,
        is_verified=is_verified,
        verification_otp_hash=otp_hash,
        verification_expires_at=exp_at,
        verification_attempts=attempts,
        last_otp_sent_at=last_otp_sent_at or now,
        is_2fa_enabled=False,
        deleted_at=None,
        created_at=now,
        updated_at=now,
    )


def _mock_user_repo(user: User | None = None):
    repo = AsyncMock()
    repo.get_by_email.return_value = user
    return repo


@pytest.fixture
def mock_db():
    session = AsyncMock()
    session.add = lambda x: None
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def client(mock_db) -> Generator[TestClient, None, None]:
    app.dependency_overrides[get_db] = lambda: mock_db
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def clear_overrides() -> Generator[None, None, None]:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


class TestForgotPassword:
    def test_forgot_password_sends_email_to_existing_user(self, client: TestClient, mock_db) -> None:
        """Existing user requesting reset receives email and generic success response."""
        app.dependency_overrides[get_db] = lambda: mock_db
        user = _make_test_user(last_otp_sent_at=datetime.now(timezone.utc) - timedelta(seconds=350))

        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_user_repo(user)), \
             patch("app.api.endpoints.auth.send_password_reset_email", new_callable=AsyncMock) as mock_send:

            resp = client.post("/auth/forgot-password", json={"email": TEST_EMAIL})

            assert resp.status_code == 200, resp.text
            assert resp.json()["status"] == "sent"
            assert "6-digit" in resp.json()["message"]
            mock_send.assert_awaited_once()

    def test_forgot_password_non_existent_user_returns_generic_200(self, client: TestClient, mock_db) -> None:
        """Non-existent email also returns generic 200 to prevent enumeration, but does not send email."""
        app.dependency_overrides[get_db] = lambda: mock_db

        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_user_repo(None)), \
             patch("app.api.endpoints.auth.send_password_reset_email", new_callable=AsyncMock) as mock_send:

            resp = client.post("/auth/forgot-password", json={"email": "nonexistent@example.com"})

            assert resp.status_code == 200, resp.text
            assert resp.json()["status"] == "sent"
            mock_send.assert_not_awaited()

    def test_forgot_password_within_cooldown_returns_429(self, client: TestClient, mock_db) -> None:
        """Requesting password reset within 60s cooldown returns 429 Too Many Requests."""
        app.dependency_overrides[get_db] = lambda: mock_db
        user = _make_test_user(last_otp_sent_at=datetime.now(timezone.utc) - timedelta(seconds=10))

        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_user_repo(user)):
            resp = client.post("/auth/forgot-password", json={"email": TEST_EMAIL})

            assert resp.status_code == 429, resp.text
            assert "Please wait" in resp.json()["detail"]


class TestResetPassword:
    def test_reset_password_wrong_otp_rejected(self, client: TestClient, mock_db) -> None:
        """Wrong OTP returns 400 Bad Request."""
        app.dependency_overrides[get_db] = lambda: mock_db
        user = _make_test_user(otp=TEST_OTP)

        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_user_repo(user)):
            resp = client.post(
                "/auth/reset-password",
                json={"email": TEST_EMAIL, "code": "000000", "new_password": NEW_PASSWORD},
            )

            assert resp.status_code == 400, resp.text
            assert "Invalid reset code" in resp.json()["detail"]

    def test_reset_password_expired_otp_rejected(self, client: TestClient, mock_db) -> None:
        """Expired OTP returns 400 Bad Request."""
        app.dependency_overrides[get_db] = lambda: mock_db
        user = _make_test_user(otp=TEST_OTP, expires_in_minutes=-5)

        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_user_repo(user)):
            resp = client.post(
                "/auth/reset-password",
                json={"email": TEST_EMAIL, "code": TEST_OTP, "new_password": NEW_PASSWORD},
            )

            assert resp.status_code == 400, resp.text
            assert "expired" in resp.json()["detail"].lower()

    def test_reset_password_weak_password_rejected(self, client: TestClient, mock_db) -> None:
        """Weak password fails validation and returns 422."""
        app.dependency_overrides[get_db] = lambda: mock_db
        user = _make_test_user(otp=TEST_OTP)

        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_user_repo(user)):
            resp = client.post(
                "/auth/reset-password",
                json={"email": TEST_EMAIL, "code": TEST_OTP, "new_password": "weak"},
            )

            assert resp.status_code == 422, resp.text

    def test_reset_password_successful_updates_password(self, client: TestClient, mock_db) -> None:
        """Valid OTP and strong password resets the password and allows login."""
        app.dependency_overrides[get_db] = lambda: mock_db
        user = _make_test_user(otp=TEST_OTP)

        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_user_repo(user)):
            resp = client.post(
                "/auth/reset-password",
                json={"email": TEST_EMAIL, "code": TEST_OTP, "new_password": NEW_PASSWORD},
            )

            assert resp.status_code == 200, resp.text
            assert resp.json()["status"] == "success"
            assert user.verification_otp_hash is None
            assert verify_password(NEW_PASSWORD, user.hashed_password)
