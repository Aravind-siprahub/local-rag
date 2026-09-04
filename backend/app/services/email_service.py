"""Email delivery and verification OTP hashing services."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import secrets
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger("app.services.email")


def generate_verification_otp() -> str:
    """Generate a cryptographically secure 6-digit numeric OTP."""
    return f"{secrets.randbelow(900000) + 100000:06d}"


def hash_verification_otp(otp: str, salt: str | None = None) -> str:
    """Hash the OTP using PBKDF2-HMAC-SHA256 with a unique salt.
    
    Format: pbkdf2_sha256$<salt_hex>$<hash_hex>
    Plaintext OTPs are never persisted to the database.
    """
    clean_otp = otp.strip()
    if salt is None:
        salt = secrets.token_hex(16)
    
    key = hashlib.pbkdf2_hmac(
        "sha256",
        clean_otp.encode("utf-8"),
        salt.encode("utf-8"),
        iterations=100_000,
    )
    return f"pbkdf2_sha256${salt}${key.hex()}"


def verify_otp(otp: str, stored_hash: str | None) -> bool:
    """Verify candidate OTP against stored PBKDF2 hash using constant-time comparison."""
    if not stored_hash or not otp:
        return False
    
    try:
        parts = stored_hash.split("$")
        if len(parts) != 3 or parts[0] != "pbkdf2_sha256":
            return False
        
        salt = parts[1]
        expected_hex = parts[2]
        
        key = hashlib.pbkdf2_hmac(
            "sha256",
            otp.strip().encode("utf-8"),
            salt.encode("utf-8"),
            iterations=100_000,
        )
        return hmac.compare_digest(key.hex(), expected_hex)
    except Exception as exc:
        logger.error("[EMAIL OTP VERIFY ERROR] Failed to verify OTP hash: %s", exc)
        return False


def _build_email_message(recipient_email: str, otp: str, expire_minutes: int, from_email: str) -> MIMEMultipart:
    """Construct HTML & plain-text MIME email for verification code."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Your Verification Code: {otp} — Talk to My Data"
    msg["From"] = from_email
    msg["To"] = recipient_email

    plain_text = (
        f"Welcome to Talk to My Data!\n\n"
        f"Your 6-digit verification code is: {otp}\n\n"
        f"This code will expire in {expire_minutes} minutes.\n"
        f"If you did not request this code, please disregard this email.\n"
    )

    html_text = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0b0f19; color: #e2e8f0; margin: 0; padding: 24px; }}
        .card {{ max-width: 480px; margin: 0 auto; background: #131b2e; border: 1px solid #1e293b; border-radius: 16px; padding: 32px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
        .header {{ text-align: center; margin-bottom: 24px; }}
        .logo {{ font-size: 20px; font-weight: 700; color: #38bdf8; letter-spacing: -0.5px; }}
        .title {{ font-size: 18px; font-weight: 600; color: #f8fafc; margin-top: 12px; }}
        .otp-box {{ background: #1e293b; border: 1px dashed #38bdf8; border-radius: 12px; padding: 18px; text-align: center; margin: 24px 0; }}
        .otp-code {{ font-family: 'Courier New', Courier, monospace; font-size: 32px; font-weight: 800; letter-spacing: 8px; color: #38bdf8; }}
        .note {{ font-size: 13px; color: #94a3b8; line-height: 1.5; text-align: center; }}
        .footer {{ font-size: 11px; color: #64748b; margin-top: 24px; text-align: center; border-top: 1px solid #1e293b; padding-top: 16px; }}
      </style>
    </head>
    <body>
      <div class="card">
        <div class="header">
          <div class="logo">⚡ Talk to My Data</div>
          <div class="title">Verify Your Email Address</div>
        </div>
        <p style="font-size: 14px; color: #cbd5e1; text-align: center;">
          Enter the following 6-digit code to complete your registration:
        </p>
        <div class="otp-box">
          <div class="otp-code">{otp}</div>
        </div>
        <p class="note">
          This verification code expires in <strong>{expire_minutes} minutes</strong>.<br>
          For your security, never share this code with anyone.
        </p>
        <div class="footer">
          If you did not request this verification code, you can safely ignore this email.
        </div>
      </div>
    </body>
    </html>
    """

    msg.attach(MIMEText(plain_text, "plain"))
    msg.attach(MIMEText(html_text, "html"))
    return msg


async def send_verification_email(recipient_email: str, otp: str) -> bool:
    """Send verification email via SMTP if configured, or log to server console in dev."""
    settings = get_settings()
    expire_minutes = getattr(settings, "EMAIL_VERIFICATION_EXPIRE_MINUTES", 10)
    from_email = getattr(settings, "SMTP_FROM_EMAIL", "noreply@talktomydata.local")
    smtp_host = getattr(settings, "SMTP_HOST", None)

    # Always log the dispatch attempt securely
    logger.info(
        "[EMAIL VERIFICATION] Dispatching OTP for recipient=%s (expires in %d min)",
        recipient_email,
        expire_minutes,
    )

    if not smtp_host:
        # In local/development mode without SMTP credentials, log the OTP clearly for developer/testing
        logger.info(
            "[DEV EMAIL MOCK] >>> Verification OTP for %s: [%s] (valid for %d min) <<<",
            recipient_email,
            otp,
            expire_minutes,
        )
        return True

    # When SMTP is configured, send email in worker thread
    def _send_sync() -> bool:
        try:
            msg = _build_email_message(recipient_email, otp, expire_minutes, from_email)
            port = getattr(settings, "SMTP_PORT", 587)
            user = getattr(settings, "SMTP_USER", None)
            password = getattr(settings, "SMTP_PASSWORD", None)
            use_tls = getattr(settings, "SMTP_TLS", True)

            server = smtplib.SMTP(smtp_host, port, timeout=10)
            if use_tls:
                server.starttls()
            if user and password:
                server.login(user, password)
            server.send_message(msg)
            server.quit()
            logger.info("[EMAIL SENT] Verification email successfully sent to %s", recipient_email)
            return True
        except Exception as exc:
            logger.error("[EMAIL SEND FAILED] Failed to send email via SMTP to %s: %s", recipient_email, exc)
            return False

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _send_sync)


def _build_password_reset_message(recipient_email: str, otp: str, expire_minutes: int, from_email: str) -> MIMEMultipart:
    """Construct HTML & plain-text MIME email for password reset."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Password Reset Code: {otp} — Talk to My Data"
    msg["From"] = from_email
    msg["To"] = recipient_email

    plain_text = (
        f"Password Reset Request — Talk to My Data\n\n"
        f"We received a request to reset your password. Use the following 6-digit code:\n\n"
        f"Your reset code is: {otp}\n\n"
        f"This code will expire in {expire_minutes} minutes.\n"
        f"If you did not request this password reset, you can safely ignore this email.\n"
    )

    html_text = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0b0f19; color: #e2e8f0; margin: 0; padding: 24px; }}
        .card {{ max-width: 480px; margin: 0 auto; background: #131b2e; border: 1px solid #1e293b; border-radius: 16px; padding: 32px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
        .header {{ text-align: center; margin-bottom: 24px; }}
        .logo {{ font-size: 20px; font-weight: 700; color: #38bdf8; letter-spacing: -0.5px; }}
        .title {{ font-size: 18px; font-weight: 600; color: #f8fafc; margin-top: 12px; }}
        .otp-box {{ background: #1e293b; border: 1px dashed #f59e0b; border-radius: 12px; padding: 18px; text-align: center; margin: 24px 0; }}
        .otp-code {{ font-family: 'Courier New', Courier, monospace; font-size: 32px; font-weight: 800; letter-spacing: 8px; color: #f59e0b; }}
        .note {{ font-size: 13px; color: #94a3b8; line-height: 1.5; text-align: center; }}
        .footer {{ font-size: 11px; color: #64748b; margin-top: 24px; text-align: center; border-top: 1px solid #1e293b; padding-top: 16px; }}
      </style>
    </head>
    <body>
      <div class="card">
        <div class="header">
          <div class="logo">⚡ Talk to My Data</div>
          <div class="title">Reset Your Password</div>
        </div>
        <p style="font-size: 14px; color: #cbd5e1; text-align: center;">
          We received a request to reset your password. Use the following 6-digit code to continue:
        </p>
        <div class="otp-box">
          <div class="otp-code">{otp}</div>
        </div>
        <p class="note">
          This code expires in <strong>{expire_minutes} minutes</strong>.<br>
          If you did not request this password reset, you can safely ignore this email.
        </p>
        <div class="footer">
          Talk to My Data &bull; Automated Security Notification
        </div>
      </div>
    </body>
    </html>
    """

    msg.attach(MIMEText(plain_text, "plain"))
    msg.attach(MIMEText(html_text, "html"))
    return msg


async def send_password_reset_email(recipient_email: str, otp: str) -> bool:
    """Send password reset email via SMTP if configured, or log to server console in dev."""
    settings = get_settings()
    expire_minutes = getattr(settings, "EMAIL_VERIFICATION_EXPIRE_MINUTES", 10)
    from_email = getattr(settings, "SMTP_FROM_EMAIL", "noreply@talktomydata.local")
    smtp_host = getattr(settings, "SMTP_HOST", None)

    logger.info(
        "[PASSWORD RESET EMAIL] Dispatching reset OTP for recipient=%s (expires in %d min)",
        recipient_email,
        expire_minutes,
    )

    if not smtp_host:
        logger.info(
            "[DEV EMAIL MOCK] >>> Password Reset OTP for %s: [%s] (valid for %d min) <<<",
            recipient_email,
            otp,
            expire_minutes,
        )
        return True

    def _send_sync() -> bool:
        try:
            msg = _build_password_reset_message(recipient_email, otp, expire_minutes, from_email)
            port = getattr(settings, "SMTP_PORT", 587)
            user = getattr(settings, "SMTP_USER", None)
            password = getattr(settings, "SMTP_PASSWORD", None)
            use_tls = getattr(settings, "SMTP_TLS", True)

            server = smtplib.SMTP(smtp_host, port, timeout=10)
            if use_tls:
                server.starttls()
            if user and password:
                server.login(user, password)
            server.send_message(msg)
            server.quit()
            logger.info("[PASSWORD RESET EMAIL SENT] Reset email successfully sent to %s", recipient_email)
            return True
        except Exception as exc:
            logger.error("[PASSWORD RESET EMAIL FAILED] Failed to send email via SMTP to %s: %s", recipient_email, exc)
            return False

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _send_sync)
