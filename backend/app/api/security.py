"""Security helper functions for JWT token handling, authentication, and ownership authorization."""
from __future__ import annotations

import base64
import hmac
import hashlib
import json
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException, status

from app.core.config import get_settings
from app.models.user import User


def hash_password(password: str) -> str:
    """Securely hash a password using PBKDF2-HMAC-SHA256 with 100,000 iterations and a random 16-byte salt."""
    if not password or not password.strip():
        raise ValueError("Password must not be empty.")
    salt = secrets.token_bytes(16)
    iterations = 100_000
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    salt_b64 = _b64url_encode(salt)
    hash_b64 = _b64url_encode(derived)
    return f"$pbkdf2-sha256${iterations}${salt_b64}${hash_b64}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain text password against a stored hashed password in constant time."""
    if not plain_password or not hashed_password:
        return False

    # Legacy fallback for early database seed accounts
    if hashed_password.startswith("INSECURE-sha256$"):
        parts = hashed_password.split("$", 1)
        if len(parts) == 2:
            computed_hex = hashlib.sha256(plain_password.encode("utf-8")).hexdigest()
            return hmac.compare_digest(computed_hex, parts[1])

    parts = hashed_password.split("$")
    if len(parts) != 5 or parts[1] != "pbkdf2-sha256":
        return False

    try:
        iterations = int(parts[2])
        salt = _b64url_decode(parts[3])
        expected_hash = _b64url_decode(parts[4])
        computed_hash = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(computed_hash, expected_hash)
    except Exception:
        return False


class TokenError(Exception):
    """Base exception for JWT processing errors."""
    pass


class TokenExpiredError(TokenError):
    """Raised when JWT token timestamp exp is in the past."""
    pass


class InvalidTokenError(TokenError):
    """Raised when JWT token is malformed, tampered with, or invalid."""
    pass


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _b64url_decode(data_str: str) -> bytes:
    padded = data_str + "=" * ((4 - len(data_str) % 4) % 4)
    return base64.urlsafe_b64decode(padded.encode("utf-8"))


def create_access_token(
    user_id: uuid.UUID | str,
    expires_delta: timedelta | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Create a cryptographically signed HS256 JWT access token for user_id."""
    settings = get_settings()
    now_dt = datetime.now(timezone.utc)
    now_ts = int(now_dt.timestamp())

    if expires_delta is not None:
        expire_ts = int((now_dt + expires_delta).timestamp())
    else:
        expire_ts = now_ts + (settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60)

    header = {"alg": settings.JWT_ALGORITHM, "typ": "JWT"}
    payload = {
        "sub": str(user_id),
        "iat": now_ts,
        "exp": expire_ts,
    }
    if extra_claims:
        payload.update(extra_claims)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    header_b64 = _b64url_encode(header_bytes)
    payload_b64 = _b64url_encode(payload_bytes)

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    secret_bytes = settings.JWT_SECRET_KEY.encode("utf-8")
    signature = hmac.new(secret_bytes, signing_input, hashlib.sha256).digest()
    signature_b64 = _b64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and cryptographically verify an HS256 JWT access token.
    
    Raises:
        InvalidTokenError: If malformed, header invalid, or signature comparison fails.
        TokenExpiredError: If exp claim is in the past.
    """
    settings = get_settings()
    parts = token.strip().split(".")
    if len(parts) != 3:
        raise InvalidTokenError("JWT token must consist of exactly 3 parts.")

    header_b64, payload_b64, signature_b64 = parts

    try:
        header_bytes = _b64url_decode(header_b64)
        header = json.loads(header_bytes.decode("utf-8"))
        if header.get("alg") != settings.JWT_ALGORITHM:
            raise InvalidTokenError(f"Unsupported JWT algorithm: {header.get('alg')}")
    except Exception as exc:
        raise InvalidTokenError(f"Invalid JWT header: {exc}") from exc

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    secret_bytes = settings.JWT_SECRET_KEY.encode("utf-8")
    expected_sig = hmac.new(secret_bytes, signing_input, hashlib.sha256).digest()

    try:
        actual_sig = _b64url_decode(signature_b64)
    except Exception as exc:
        raise InvalidTokenError(f"Invalid JWT signature encoding: {exc}") from exc

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise InvalidTokenError("JWT signature verification failed — token tampered with or invalid key.")

    try:
        payload_bytes = _b64url_decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:
        raise InvalidTokenError(f"Invalid JWT payload: {exc}") from exc

    if "sub" not in payload:
        raise InvalidTokenError("JWT payload missing required 'sub' subject claim.")

    exp = payload.get("exp")
    if exp is not None:
        now_ts = int(time.time())
        if now_ts > exp:
            raise TokenExpiredError("JWT token has expired.")

    return payload


# --- 2FA Secret Encryption & Decryption ----------------------------------------

def _get_encryption_key() -> bytes:
    """Derive a 256-bit encryption key from JWT_SECRET_KEY using SHA-256."""
    settings = get_settings()
    return hashlib.sha256(f"totp-enc-salt:{settings.JWT_SECRET_KEY}".encode("utf-8")).digest()


def encrypt_totp_secret(plain_secret: str) -> str:
    """Encrypt TOTP secret using AES-256-GCM authenticated encryption."""
    if not plain_secret or not plain_secret.strip():
        raise ValueError("Secret cannot be empty.")

    key = _get_encryption_key()
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        aesgcm = AESGCM(key)
        nonce = secrets.token_bytes(12)
        ciphertext = aesgcm.encrypt(nonce, plain_secret.strip().encode("utf-8"), None)
        return f"$aes-gcm${_b64url_encode(nonce)}${_b64url_encode(ciphertext)}"
    except ImportError:
        # Fallback: HMAC-SHA256 authenticated encryption
        iv = secrets.token_bytes(16)
        stream_key = hashlib.sha256(key + iv).digest()
        plain_bytes = plain_secret.strip().encode("utf-8")
        cipher_bytes = bytes(p ^ stream_key[i % len(stream_key)] for i, p in enumerate(plain_bytes))
        tag = hmac.new(key, iv + cipher_bytes, hashlib.sha256).digest()
        return f"$hmac-enc${_b64url_encode(iv)}${_b64url_encode(cipher_bytes)}${_b64url_encode(tag)}"


def decrypt_totp_secret(encrypted_secret: str) -> str:
    """Decrypt an encrypted TOTP secret using AES-256-GCM authenticated decryption."""
    if not encrypted_secret or not encrypted_secret.strip():
        raise ValueError("Encrypted secret cannot be empty.")

    parts = encrypted_secret.strip().split("$")
    key = _get_encryption_key()

    if len(parts) == 4 and parts[1] == "aes-gcm":
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = _b64url_decode(parts[2])
        ciphertext = _b64url_decode(parts[3])
        aesgcm = AESGCM(key)
        plain_bytes = aesgcm.decrypt(nonce, ciphertext, None)
        return plain_bytes.decode("utf-8")
    elif len(parts) == 5 and parts[1] == "hmac-enc":
        iv = _b64url_decode(parts[2])
        cipher_bytes = _b64url_decode(parts[3])
        tag = _b64url_decode(parts[4])
        expected_tag = hmac.new(key, iv + cipher_bytes, hashlib.sha256).digest()
        if not hmac.compare_digest(expected_tag, tag):
            raise InvalidTokenError("Encrypted TOTP secret MAC validation failed.")
        stream_key = hashlib.sha256(key + iv).digest()
        plain_bytes = bytes(c ^ stream_key[i % len(stream_key)] for i, c in enumerate(cipher_bytes))
        return plain_bytes.decode("utf-8")
    else:
        # Backward compatibility for plain base32 secret in development/testing
        if len(encrypted_secret) == 32 and encrypted_secret.isalnum():
            return encrypted_secret
        raise InvalidTokenError("Unknown or corrupted encrypted secret format.")


# --- 2FA Challenge Tokens -----------------------------------------------------

def create_2fa_challenge_token(
    user_id: uuid.UUID | str,
    is_setup: bool = False,
    expires_minutes: int = 5,
    extra_data: dict[str, Any] | None = None,
) -> str:
    """Create a short-lived (5-minute) signed challenge token for 2FA verification."""
    extra = {
        "token_type": "2fa_challenge",
        "is_setup": is_setup,
    }
    if extra_data:
        extra.update(extra_data)
    return create_access_token(
        user_id=user_id,
        expires_delta=timedelta(minutes=expires_minutes),
        extra_claims=extra,
    )


def decode_2fa_challenge_token(token: str) -> dict[str, Any]:
    """Decode and verify a 2FA challenge token, ensuring it is unexpired and of type '2fa_challenge'."""
    payload = decode_access_token(token)
    if payload.get("token_type") != "2fa_challenge":
        raise InvalidTokenError("Token is not a valid 2FA challenge token.")
    return payload


# --- 2FA Rate Limiting & Brute Force Protection -------------------------------

_2FA_ATTEMPTS: dict[str, list[float]] = {}
_MAX_2FA_ATTEMPTS = 5
_LOCKOUT_WINDOW_SECONDS = 300  # 5 minutes


def is_2fa_rate_limited(identifier: str) -> tuple[bool, int]:
    """Check if identifier (user_id or IP) is rate limited. Returns (is_limited, retry_after_seconds)."""
    now = time.time()
    history = _2FA_ATTEMPTS.get(identifier, [])
    # Keep only timestamps within window
    recent = [t for t in history if now - t < _LOCKOUT_WINDOW_SECONDS]
    _2FA_ATTEMPTS[identifier] = recent

    if len(recent) >= _MAX_2FA_ATTEMPTS:
        oldest = recent[0]
        retry_after = int(_LOCKOUT_WINDOW_SECONDS - (now - oldest))
        return True, max(1, retry_after)
    return False, 0


def record_failed_2fa_attempt(identifier: str) -> int:
    """Record a failed 2FA verification attempt. Returns number of remaining attempts."""
    now = time.time()
    history = _2FA_ATTEMPTS.setdefault(identifier, [])
    history.append(now)
    # clean old
    recent = [t for t in history if now - t < _LOCKOUT_WINDOW_SECONDS]
    _2FA_ATTEMPTS[identifier] = recent
    return max(0, _MAX_2FA_ATTEMPTS - len(recent))


def reset_2fa_attempts(identifier: str) -> None:
    """Clear failed attempts on successful verification."""
    _2FA_ATTEMPTS.pop(identifier, None)


def verify_ownership(
    resource_owner_id: uuid.UUID | str,
    current_user: User,
    resource_name: str = "resource",
) -> None:
    """Raise HTTP 403 Forbidden if current_user does not own the target resource."""
    owner_str = str(resource_owner_id)
    user_str = str(current_user.id)

    if owner_str != user_str:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: You do not have permission to access or modify this {resource_name}.",
        )

