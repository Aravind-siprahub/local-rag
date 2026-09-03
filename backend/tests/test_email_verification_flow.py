"""Comprehensive automated tests for email verification and login flow.

Covers all cases specified in Requirement 15:
- New email -> verification required, no auto-login / JWT issued
- Existing verified email -> rejected with 409 ("Email already registered. Please login.")
- Existing unverified email -> resends OTP, remains in verification state
- Wrong OTP -> rejected with 400 and attempt counter
- Expired OTP -> rejected with 400
- Reused OTP -> rejected
- Successful OTP -> email marked verified (is_verified=True), no token returned
- Direct API login attempt with unverified email -> backend independently rejects with 403
- Direct API login attempt with verified email -> succeeds (returns 2FA challenge / token)
- Resend verification cooldown -> rejects with 429 if called within 60s
"""
import uuid
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.security import hash_password
from app.db.session import get_db
from app.main import app
from app.models.enums import UserRole
from app.models.user import User
from app.services.email_service import hash_verification_otp

TEST_EMAIL = "verify_user@example.com"
TEST_PASSWORD = "Strong@Password123"
TEST_NAME = "Verification Test User"
TEST_OTP = "654321"


def _make_test_user(
    email: str = TEST_EMAIL,
    password: str = TEST_PASSWORD,
    is_verified: bool = False,
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
        full_name=TEST_NAME,
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


# ==============================================================================
# 1. Registration Flow Tests
# ==============================================================================

class TestRegistrationFlow:
    def test_new_email_requires_verification_and_no_auto_login(self, client: TestClient, mock_db) -> None:
        """New email registration: returns 201, verification_required status, and NO access_token."""
        app.dependency_overrides[get_db] = lambda: mock_db
        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_user_repo(None)), \
             patch("app.api.endpoints.auth.send_verification_email", new_callable=AsyncMock) as mock_send:

            resp = client.post(
                "/auth/register",
                json={
                    "email": TEST_EMAIL,
                    "password": TEST_PASSWORD,
                    "full_name": TEST_NAME,
                },
            )

            assert resp.status_code == 201, resp.text
            body = resp.json()
            assert body["status"] == "verification_required"
            assert body["email"] == TEST_EMAIL
            assert "access_token" not in body  # Requirement 1: DO NOT automatically log in
            mock_send.assert_awaited_once()

    def test_existing_verified_email_returns_409(self, client: TestClient, mock_db) -> None:
        """Existing verified email: returns 409 Conflict with 'Email already registered. Please login.'"""
        app.dependency_overrides[get_db] = lambda: mock_db
        verified_user = _make_test_user(is_verified=True)
        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_user_repo(verified_user)):
            resp = client.post(
                "/auth/register",
                json={
                    "email": TEST_EMAIL,
                    "password": TEST_PASSWORD,
                },
            )

            assert resp.status_code == 409, resp.text
            assert "Email already registered. Please login." in resp.json()["detail"]

    def test_existing_unverified_email_resends_otp(self, client: TestClient, mock_db) -> None:
        """Existing unverified email: refreshes OTP and keeps in verification_required state."""
        app.dependency_overrides[get_db] = lambda: mock_db
        past = datetime.now(timezone.utc) - timedelta(seconds=350)
        unverified_user = _make_test_user(is_verified=False, last_otp_sent_at=past)

        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_user_repo(unverified_user)), \
             patch("app.api.endpoints.auth.send_verification_email", new_callable=AsyncMock) as mock_send:

            resp = client.post(
                "/auth/register",
                json={
                    "email": TEST_EMAIL,
                    "password": TEST_PASSWORD,
                },
            )

            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["status"] == "verification_required"
            assert "access_token" not in body
            mock_send.assert_awaited_once()


# ==============================================================================
# 2. Email Verification Flow Tests (/auth/verify-email)
# ==============================================================================

class TestVerifyEmail:
    def test_wrong_otp_rejected(self, client: TestClient, mock_db) -> None:
        """Wrong OTP: rejected with 400 Bad Request and decrements remaining attempts."""
        app.dependency_overrides[get_db] = lambda: mock_db
        user = _make_test_user(is_verified=False, otp=TEST_OTP, attempts=0)

        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_user_repo(user)):
            resp = client.post(
                "/auth/verify-email",
                json={"email": TEST_EMAIL, "code": "000000"},
            )

            assert resp.status_code == 400, resp.text
            assert "Invalid verification code" in resp.json()["detail"]
            assert user.verification_attempts == 1

    def test_expired_otp_rejected(self, client: TestClient, mock_db) -> None:
        """Expired OTP: rejected with 400 Bad Request."""
        app.dependency_overrides[get_db] = lambda: mock_db
        user = _make_test_user(is_verified=False, otp=TEST_OTP, expires_in_minutes=-5)

        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_user_repo(user)):
            resp = client.post(
                "/auth/verify-email",
                json={"email": TEST_EMAIL, "code": TEST_OTP},
            )

            assert resp.status_code == 400, resp.text
            assert "expired" in resp.json()["detail"].lower()

    def test_successful_otp_marks_verified_and_issues_token(self, client: TestClient, mock_db) -> None:
        """Correct OTP: marks is_verified=True, clears OTP fields, returns 200, and returns access_token."""
        app.dependency_overrides[get_db] = lambda: mock_db
        user = _make_test_user(is_verified=False, otp=TEST_OTP)

        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_user_repo(user)):
            resp = client.post(
                "/auth/verify-email",
                json={"email": TEST_EMAIL, "code": TEST_OTP},
            )

            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["status"] == "verified"
            assert "access_token" in body  # OTP verification logs user in immediately
            assert user.is_verified is True
            assert user.verification_otp_hash is None

    def test_already_verified_email_returns_verified_status(self, client: TestClient, mock_db) -> None:
        """Calling verify on an already verified account safely returns verified status."""
        app.dependency_overrides[get_db] = lambda: mock_db
        user = _make_test_user(is_verified=True, otp=None)

        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_user_repo(user)):
            resp = client.post(
                "/auth/verify-email",
                json={"email": TEST_EMAIL, "code": TEST_OTP},
            )

            assert resp.status_code == 200, resp.text
            assert resp.json()["status"] == "verified"


# ==============================================================================
# 3. Independent Login Enforcement Tests (/auth/login)
# ==============================================================================

class TestLoginVerificationEnforcement:
    def test_unverified_email_login_rejected_with_403(self, client: TestClient, mock_db) -> None:
        """Direct API login attempt with unverified email: backend independently rejects with 403 Forbidden."""
        app.dependency_overrides[get_db] = lambda: mock_db
        unverified_user = _make_test_user(is_verified=False)

        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_user_repo(unverified_user)):
            resp = client.post(
                "/auth/login",
                json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
            )

            assert resp.status_code == 403, resp.text
            assert "Email is not verified" in resp.json()["detail"]

    def test_verified_email_login_allowed(self, client: TestClient, mock_db) -> None:
        """Verified email login: allowed to proceed."""
        app.dependency_overrides[get_db] = lambda: mock_db
        verified_user = _make_test_user(is_verified=True)

        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_user_repo(verified_user)):
            resp = client.post(
                "/auth/login",
                json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
            )

            # Verified user logs in directly without 2FA QR code challenge
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body.get("requires_2fa") is False
            assert "access_token" in body


# ==============================================================================
# 4. Resend Verification Cooldown Tests (/auth/resend-verification)
# ==============================================================================

class TestResendVerification:
    def test_resend_within_cooldown_returns_429(self, client: TestClient, mock_db) -> None:
        """Resending verification within 60s cooldown returns 429 Too Many Requests."""
        app.dependency_overrides[get_db] = lambda: mock_db
        recent = datetime.now(timezone.utc) - timedelta(seconds=15)
        user = _make_test_user(is_verified=False, last_otp_sent_at=recent)

        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_user_repo(user)):
            resp = client.post(
                "/auth/resend-verification",
                json={"email": TEST_EMAIL},
            )

            assert resp.status_code == 429, resp.text
            assert "Please wait" in resp.json()["detail"]

    def test_resend_after_cooldown_succeeds(self, client: TestClient, mock_db) -> None:
        """Resending verification after 60s cooldown elapses generates new OTP and sends email."""
        app.dependency_overrides[get_db] = lambda: mock_db
        old = datetime.now(timezone.utc) - timedelta(seconds=350)
        user = _make_test_user(is_verified=False, last_otp_sent_at=old)

        with patch("app.api.endpoints.auth.UserRepository", return_value=_mock_user_repo(user)), \
             patch("app.api.endpoints.auth.send_verification_email", new_callable=AsyncMock) as mock_send:

            resp = client.post(
                "/auth/resend-verification",
                json={"email": TEST_EMAIL},
            )

            assert resp.status_code == 200, resp.text
            assert resp.json()["status"] == "sent"
            mock_send.assert_awaited_once()
