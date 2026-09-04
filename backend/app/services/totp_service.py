"""TOTP Authenticator-App (RFC 6238) and Recovery Codes Service.

Provides cryptographically secure secret generation, QR code rendering,
constant-time TOTP verification, and single-use recovery code management.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import logging
import re
import secrets
import time
from typing import Any
from urllib.parse import quote

try:
    import pyotp
except ImportError:
    pyotp = None  # Fallback to pure RFC 6238 implementation below

try:
    import qrcode
    import qrcode.image.svg
except ImportError:
    qrcode = None

logger = logging.getLogger("app.services.totp")


def generate_totp_secret() -> str:
    """Generate a cryptographically secure 160-bit (32-character Base32) TOTP secret."""
    if pyotp is not None:
        return pyotp.random_base32()
    raw = secrets.token_bytes(20)
    return base64.b32encode(raw).decode("utf-8").replace("=", "")


def get_provisioning_uri(secret: str, email: str, issuer: str = "Local RAG") -> str:
    """Generate a standard otpauth:// URI for authenticator app enrollment."""
    clean_email = email.strip()
    clean_issuer = issuer.strip()
    if pyotp is not None:
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=clean_email, issuer_name=clean_issuer)
    
    encoded_label = f"{quote(clean_issuer)}:{quote(clean_email)}"
    return f"otpauth://totp/{encoded_label}?secret={secret}&issuer={quote(clean_issuer)}&algorithm=SHA1&digits=6&period=30"


def generate_qr_data_url(uri: str) -> str:
    """Generate a PNG or SVG data URL from the provisioning URI for frontend display."""
    if qrcode is not None:
        try:
            # Generate PNG data URI
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_M,
                box_size=10,
                border=4,
            )
            qr.add_data(uri)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            b64_png = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64_png}"
        except Exception as exc:
            logger.warning("[TOTP QR] Failed to render PNG QR code: %s. Falling back to SVG.", exc)

    # Pure SVG fallback if PIL/PNG is not available
    try:
        if qrcode is not None:
            factory = qrcode.image.svg.SvgPathImage
            svg_img = qrcode.make(uri, image_factory=factory)
            svg_bytes = io.BytesIO()
            svg_img.save(svg_bytes)
            svg_str = svg_bytes.getvalue().decode("utf-8")
            b64_svg = base64.b64encode(svg_str.encode("utf-8")).decode("utf-8")
            return f"data:image/svg+xml;base64,{b64_svg}"
    except Exception as exc:
        logger.warning("[TOTP QR] SVG factory failed: %s", exc)

    # Minimal inline SVG fallback
    svg_fallback = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'
        f'<rect width="200" height="200" fill="#f8fafc"/>'
        f'<text x="100" y="100" text-anchor="middle" fill="#475569" font-family="sans-serif" font-size="12">'
        f'Scan via manual key</text></svg>'
    )
    b64_svg = base64.b64encode(svg_fallback.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64_svg}"


def verify_totp_code(secret: str, code: str, valid_window: int = 1) -> bool:
    """Verify a 6-digit TOTP code against the secret within a clock skew window (±1 step / 30s)."""
    clean_code = str(code).strip().replace(" ", "").replace("-", "")
    if not clean_code.isdigit() or len(clean_code) != 6:
        return False

    if pyotp is not None:
        try:
            totp = pyotp.TOTP(secret)
            return bool(totp.verify(clean_code, valid_window=valid_window))
        except Exception as exc:
            logger.warning("[TOTP VERIFY] pyotp verification failed: %s", exc)

    # Pure Python RFC 6238 fallback
    import struct
    try:
        now = int(time.time())
        interval = 30
        padded_secret = secret.upper()
        # Add padding if missing
        missing_padding = len(padded_secret) % 8
        if missing_padding != 0:
            padded_secret += "=" * (8 - missing_padding)
        key = base64.b32decode(padded_secret, casefold=True)

        for offset in range(-valid_window, valid_window + 1):
            counter = (now + (offset * interval)) // interval
            msg = struct.pack(">Q", counter)
            digest = hmac.new(key, msg, hashlib.sha1).digest()
            o = digest[19] & 0x0F
            otp_num = (struct.unpack(">I", digest[o : o + 4])[0] & 0x7FFFFFFF) % 1_000_000
            expected = f"{otp_num:06d}"
            if hmac.compare_digest(clean_code, expected):
                return True
    except Exception as exc:
        logger.warning("[TOTP VERIFY] RFC 6238 fallback failed: %s", exc)

    return False


def generate_recovery_codes(count: int = 10) -> list[str]:
    """Generate a list of formatted cryptographically random recovery codes (e.g. 'A1B2-C3D4')."""
    codes: list[str] = []
    for _ in range(count):
        part1 = secrets.token_hex(2).upper()
        part2 = secrets.token_hex(2).upper()
        codes.append(f"{part1}-{part2}")
    return codes


generate_backup_codes = generate_recovery_codes



def _hash_single_backup_code(code: str, salt: bytes) -> str:
    clean = code.strip().upper().replace("-", "").replace(" ", "")
    h = hashlib.pbkdf2_hmac("sha256", clean.encode("utf-8"), salt, 50_000)
    salt_hex = salt.hex()
    hash_hex = h.hex()
    return f"{salt_hex}${hash_hex}"


def hash_recovery_codes(plain_codes: list[str]) -> str:
    """Hash all plain recovery codes into a JSON serialized string for database storage."""
    records = []
    for code in plain_codes:
        salt = secrets.token_bytes(16)
        code_hash = _hash_single_backup_code(code, salt)
        records.append({
            "hash": code_hash,
            "used": False,
            "used_at": None,
        })
    return json.dumps(records)


def verify_and_consume_recovery_code(
    plain_code: str,
    stored_json: str | None,
) -> tuple[bool, str | None, int]:
    """Verify a recovery code against stored hashes and mark it as consumed.
    
    Returns:
        tuple[is_valid: bool, updated_json: str | None, remaining_count: int]
    """
    if not plain_code or not stored_json:
        return False, None, 0

    clean = plain_code.strip().upper().replace("-", "").replace(" ", "")
    if len(clean) != 8:
        return False, stored_json, 0

    try:
        records: list[dict[str, Any]] = json.loads(stored_json)
    except Exception:
        return False, None, 0

    matched_index: int | None = None

    for idx, rec in enumerate(records):
        if rec.get("used", False):
            continue
        code_hash_str = rec.get("hash", "")
        parts = code_hash_str.split("$")
        if len(parts) != 2:
            continue
        try:
            salt = bytes.fromhex(parts[0])
            expected_hash = bytes.fromhex(parts[1])
            computed_hash = hashlib.pbkdf2_hmac("sha256", clean.encode("utf-8"), salt, 50_000)
            if hmac.compare_digest(computed_hash, expected_hash):
                matched_index = idx
                break
        except Exception:
            continue

    if matched_index is None:
        remaining = sum(1 for r in records if not r.get("used", False))
        return False, stored_json, remaining

    # Mark code as consumed
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    records[matched_index]["used"] = True
    records[matched_index]["used_at"] = now_iso

    remaining = sum(1 for r in records if not r.get("used", False))
    updated_json = json.dumps(records)
    logger.info("[RECOVERY CODE CONSUMED] Remaining unused backup codes: %d", remaining)
    return True, updated_json, remaining
