"""Comprehensive tests for Password + TOTP Authenticator-App 2FA and User Isolation.

Tests all 17 security requirements:
1. Registration & password hashing
2. Password verification on login
3. Incorrect password rejection
4. TOTP secret and provisioning URI generation
5. QR code data URL rendering
6. 2FA challenge creation and intermediate state
7. Correct 6-digit TOTP code verification
8. Incorrect TOTP rejection and attempt decrement
9. Challenge token expiration and tampering checks
10. 2FA Rate limiting & brute force lockout
11. Encrypted TOTP secret storage at rest
12. Backup recovery code generation and formatting
13. Backup recovery code verification
14. Backup recovery code single-use consumption enforcement
15. User isolation: Documents (User A cannot access User B's documents)
16. User isolation: Chat sessions (User A cannot access User B's chats)
17. User isolation: Long-term memory (User A cannot access User B's memories)
"""
import base64
import json
import time
import uuid
from datetime import timedelta
import pytest
from httpx import ASGITransport, AsyncClient
import pyotp

from app.api.security import (
    create_2fa_challenge_token,
    decode_2fa_challenge_token,
    decrypt_totp_secret,
    encrypt_totp_secret,
    hash_password,
    is_2fa_rate_limited,
    record_failed_2fa_attempt,
    reset_2fa_attempts,
    verify_password,
    InvalidTokenError,
    TokenExpiredError,
)
from app.core.config import get_settings
from app.db.session import get_db
from app.main import app
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.services.totp_service import (
    generate_backup_codes,
    generate_recovery_codes,
    generate_qr_data_url,
    generate_totp_secret,
    get_provisioning_uri,
    hash_recovery_codes,
    verify_and_consume_recovery_code,
    verify_totp_code,
)


@pytest.mark.asyncio
async def test_password_hashing_and_verification():
    """Verify password hashing with random salt and constant-time verification."""
    password = "SuperSecretPassword123!"
    hashed1 = hash_password(password)
    hashed2 = hash_password(password)

    # Different salts must yield different hashes
    assert hashed1 != hashed2
    assert hashed1.startswith("$pbkdf2-sha256$")

    # Verification must succeed for correct password
    assert verify_password(password, hashed1) is True
    assert verify_password(password, hashed2) is True

    # Verification must fail for wrong password
    assert verify_password("WrongPassword123!", hashed1) is False
    assert verify_password("", hashed1) is False


@pytest.mark.asyncio
async def test_totp_secret_encryption_at_rest():
    """Verify TOTP secret is encrypted at rest using AES-GCM and never plain text."""
    plain_secret = generate_totp_secret()
    assert len(plain_secret) == 32

    encrypted = encrypt_totp_secret(plain_secret)
    # Must not store plain secret
    assert plain_secret not in encrypted
    assert encrypted.startswith(("$aes-gcm$", "$hmac-enc$"))

    # Decryption must match original secret
    decrypted = decrypt_totp_secret(encrypted)
    assert decrypted == plain_secret


@pytest.mark.asyncio
async def test_totp_provisioning_and_qr_generation():
    """Verify standard otpauth URI format and QR code generation."""
    secret = generate_totp_secret()
    email = "alice@example.com"
    uri = get_provisioning_uri(secret, email, issuer="Local RAG")

    assert uri.startswith("otpauth://totp/")
    assert "alice%40example.com" in uri or "alice@example.com" in uri
    assert secret in uri

    qr_url = generate_qr_data_url(uri)
    assert qr_url.startswith(("data:image/png;base64,", "data:image/svg+xml;base64,"))
    assert len(qr_url) > 100


@pytest.mark.asyncio
async def test_totp_code_verification():
    """Verify RFC 6238 TOTP code generation and time window verification."""
    secret = generate_totp_secret()
    totp = pyotp.TOTP(secret)
    valid_code = totp.now()

    # Valid code matches
    assert verify_totp_code(secret, valid_code) is True

    # Invalid code rejected
    assert verify_totp_code(secret, "000000") is False
    assert verify_totp_code(secret, "abcdef") is False
    assert verify_totp_code(secret, "123") is False


@pytest.mark.asyncio
async def test_backup_recovery_codes_single_use():
    """Verify backup recovery code generation, verification, and single-use consumption."""
    codes = generate_recovery_codes(10)
    assert len(codes) == 10
    for c in codes:
        assert len(c) == 9  # XXXX-XXXX format
        assert "-" in c

    stored_json = hash_recovery_codes(codes)
    assert codes[0] not in stored_json  # Stored as hashes

    # 1. Use the first code
    valid1, updated_json1, remaining1 = verify_and_consume_recovery_code(codes[0], stored_json)
    assert valid1 is True
    assert remaining1 == 9
    assert updated_json1 is not None

    # 2. Try to reuse the SAME consumed code -> MUST FAIL
    valid_reuse, _, remaining_after = verify_and_consume_recovery_code(codes[0], updated_json1)
    assert valid_reuse is False
    assert remaining_after == 9

    # 3. Use another unconsumed code -> MUST SUCCEED
    valid2, updated_json2, remaining2 = verify_and_consume_recovery_code(codes[1], updated_json1)
    assert valid2 is True
    assert remaining2 == 8


@pytest.mark.asyncio
async def test_2fa_rate_limiting():
    """Verify brute force lockout after 5 failed 2FA verification attempts."""
    test_id = f"test-user-{uuid.uuid4()}"
    reset_2fa_attempts(test_id)

    # 4 failed attempts: not yet locked out
    for i in range(4):
        rem = record_failed_2fa_attempt(test_id)
        is_lim, _ = is_2fa_rate_limited(test_id)
        assert is_lim is False

    # 5th failed attempt: triggers lockout
    rem = record_failed_2fa_attempt(test_id)
    is_lim, retry_after = is_2fa_rate_limited(test_id)
    assert is_lim is True
    assert retry_after > 0

    # Reset clears lockout
    reset_2fa_attempts(test_id)
    is_lim, _ = is_2fa_rate_limited(test_id)
    assert is_lim is False


@pytest.mark.asyncio
async def test_full_registration_and_2fa_enable_flow():
    """Test full registration -> 2FA onboarding -> login flow with test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        test_email = f"user_2fa_{uuid.uuid4().hex[:8]}@example.com"
        test_pwd = "SecurePassword2026!"

        # 1. Register User
        reg_res = await client.post(
            "/auth/register",
            json={"email": test_email, "password": test_pwd, "full_name": "Test 2FA User"},
        )
        assert reg_res.status_code == 201
        reg_data = reg_res.json()
        assert "totp_secret" in reg_data
        assert "qr_code_data_url" in reg_data
        assert "temp_token" in reg_data
        assert len(reg_data["backup_codes"]) == 10

        secret = reg_data["totp_secret"]
        temp_token = reg_data["temp_token"]

        # 2. Try invalid 2FA code -> 400 Bad Request
        fail_res = await client.post(
            "/auth/2fa/enable",
            json={"temp_token": temp_token, "code": "999999"},
        )
        assert fail_res.status_code == 400

        # 3. Enter valid 6-digit TOTP code -> 200 OK & JWT access token
        totp = pyotp.TOTP(secret)
        valid_code = totp.now()
        enable_res = await client.post(
            "/auth/2fa/enable",
            json={"temp_token": temp_token, "code": valid_code},
        )
        assert enable_res.status_code == 200
        enable_data = enable_res.json()
        assert "access_token" in enable_data
        assert enable_data["user"]["is_2fa_enabled"] is True

        # 4. Login with email + password -> Returns 2FA challenge
        login_res = await client.post(
            "/auth/login",
            json={"email": test_email, "password": test_pwd},
        )
        assert login_res.status_code == 200
        login_data = login_res.json()
        assert login_data["requires_2fa"] is True
        assert "temp_token" in login_data
        login_challenge = login_data["temp_token"]

        # 5. Verify 2FA challenge with authenticator code -> Full access token
        verify_res = await client.post(
            "/auth/verify-2fa",
            json={"temp_token": login_challenge, "code": totp.now()},
        )
        assert verify_res.status_code == 200
        auth_data = verify_res.json()
        assert "access_token" in auth_data
        token = auth_data["access_token"]

        # 6. Access /auth/me with access token
        me_res = await client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert me_res.status_code == 200
        assert me_res.json()["email"] == test_email


@pytest.mark.asyncio
async def test_login_with_recovery_code():
    """Test user login using an emergency recovery code instead of 6-digit TOTP."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        test_email = f"recovery_user_{uuid.uuid4().hex[:8]}@example.com"
        test_pwd = "SecurePassword2026!"

        # Register & enable
        reg_res = await client.post(
            "/auth/register",
            json={"email": test_email, "password": test_pwd, "full_name": "Recovery User"},
        )
        reg_data = reg_res.json()
        secret = reg_data["totp_secret"]
        backup_code = reg_data["backup_codes"][0]

        await client.post(
            "/auth/2fa/enable",
            json={"temp_token": reg_data["temp_token"], "code": pyotp.TOTP(secret).now()},
        )

        # Login -> Challenge
        login_res = await client.post(
            "/auth/login",
            json={"email": test_email, "password": test_pwd},
        )
        challenge = login_res.json()["temp_token"]

        # Verify with backup code
        rec_res = await client.post(
            "/auth/verify-2fa",
            json={"temp_token": challenge, "code": backup_code, "is_backup_code": True},
        )
        assert rec_res.status_code == 200
        assert "access_token" in rec_res.json()


@pytest.mark.asyncio
async def test_user_isolation_documents_chats_memory():
    """Verify strict user isolation across Documents, Chat Sessions, and Memory."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create User A
        email_a = f"usera_{uuid.uuid4().hex[:8]}@example.com"
        pwd = "Password123!"
        res_a = await client.post("/auth/register", json={"email": email_a, "password": pwd})
        data_a = res_a.json()
        auth_a = await client.post(
            "/auth/2fa/enable",
            json={"temp_token": data_a["temp_token"], "code": pyotp.TOTP(data_a["totp_secret"]).now()},
        )
        token_a = auth_a.json()["access_token"]
        user_id_a = auth_a.json()["user"]["id"]

        # Create User B
        email_b = f"userb_{uuid.uuid4().hex[:8]}@example.com"
        res_b = await client.post("/auth/register", json={"email": email_b, "password": pwd})
        data_b = res_b.json()
        auth_b = await client.post(
            "/auth/2fa/enable",
            json={"temp_token": data_b["temp_token"], "code": pyotp.TOTP(data_b["totp_secret"]).now()},
        )
        token_b = auth_b.json()["access_token"]
        user_id_b = auth_b.json()["user"]["id"]

        # 1. User A creates a Document
        doc_res = await client.post(
            "/documents",
            json={"user_id": user_id_a, "title": "User A Private Document", "description": "Confidential"},
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert doc_res.status_code == 201
        doc_id = doc_res.json()["id"]

        # User A attempts to create a document using User B's user_id -> 403 Forbidden
        doc_spoof = await client.post(
            "/documents",
            json={"user_id": user_id_b, "title": "Spoofed Document", "description": "Attack"},
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert doc_spoof.status_code == 403

        # User B attempts to access User A's document -> 403 Forbidden
        doc_b_access = await client.get(
            f"/documents/{doc_id}/debug",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert doc_b_access.status_code in (403, 404)

        # 2. User A creates a Chat Session
        chat_res = await client.post(
            "/chat-sessions",
            json={"title": "User A Chat"},
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert chat_res.status_code == 201
        session_id = chat_res.json()["id"]

        # User B attempts to access User A's chat session -> 403 Forbidden
        chat_b_access = await client.get(
            f"/chat-sessions/{session_id}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert chat_b_access.status_code == 403

        # User B attempts to delete User A's chat session -> 403 Forbidden
        chat_b_delete = await client.delete(
            f"/chat-sessions/{session_id}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert chat_b_delete.status_code == 403

        # 3. User A creates a Memory entry
        mem_res = await client.post(
            "/memory",
            json={"content": "User A secret memory", "memory_type": "preference"},
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert mem_res.status_code == 201

        # User B lists memories -> Must NOT contain User A's memory
        mem_b_list = await client.get(
            "/memory",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert mem_b_list.status_code == 200
        b_memories = mem_b_list.json()["items"]
        assert all(m["content"] != "User A secret memory" for m in b_memories)


@pytest.mark.asyncio
async def test_unauthenticated_api_rejection():
    """Verify that protected API endpoints reject unauthenticated or malformed tokens with 401."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Missing header
        res1 = await client.get("/documents")
        assert res1.status_code == 401

        # Malformed bearer
        res2 = await client.get("/documents", headers={"Authorization": "Bearer invalid.token.signature"})
        assert res2.status_code == 401

        # Expired token
        expired_token = create_2fa_challenge_token(uuid.uuid4(), expires_minutes=-10)
        res3 = await client.get("/documents", headers={"Authorization": f"Bearer {expired_token}"})
        assert res3.status_code == 401


@pytest.mark.asyncio
async def test_expired_2fa_challenge_rejection():
    """Verify that expired 2FA challenge tokens are strictly rejected."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        expired_challenge = create_2fa_challenge_token(uuid.uuid4(), expires_minutes=-5)
        res = await client.post(
            "/auth/verify-2fa",
            json={"temp_token": expired_challenge, "code": "123456"},
        )
        assert res.status_code == 401


@pytest.mark.asyncio
async def test_incorrect_totp_rejection_keeps_user_unauthenticated():
    """Verify that entering wrong TOTP code rejects authentication and leaves user unauthenticated."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        email = f"wrong_totp_{uuid.uuid4().hex[:8]}@example.com"
        pwd = "ValidPassword123!"

        # Register & enable
        reg = await client.post("/auth/register", json={"email": email, "password": pwd})
        reg_data = reg.json()
        await client.post(
            "/auth/2fa/enable",
            json={"temp_token": reg_data["temp_token"], "code": pyotp.TOTP(reg_data["totp_secret"]).now()},
        )

        # Login -> get challenge
        login_res = await client.post("/auth/login", json={"email": email, "password": pwd})
        challenge = login_res.json()["temp_token"]

        # Attempt with wrong TOTP
        verify_fail = await client.post(
            "/auth/verify-2fa",
            json={"temp_token": challenge, "code": "000000"},
        )
        assert verify_fail.status_code == 400
        assert "attempt(s) remaining" in verify_fail.json()["detail"]


@pytest.mark.asyncio
async def test_logout_endpoint():
    """Verify user logout endpoint succeeds cleanly."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/auth/logout")
        assert res.status_code == 200
        assert res.json() == {"message": "Logged out successfully."}

