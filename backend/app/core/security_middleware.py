"""Security utilities: Prompt Injection protection and File Validation."""
from __future__ import annotations

import re
from fastapi import HTTPException, status

_PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above)\s+instructions",
    r"system\s+prompt\s+override",
    r"<\|im_start\|>",
    r"\[INST\]",
]

def sanitize_prompt(prompt_text: str) -> str:
    """Sanitize prompt text to mitigate prompt injection markers."""
    if not prompt_text:
        return prompt_text

    sanitized = prompt_text
    for pattern in _PROMPT_INJECTION_PATTERNS:
        sanitized = re.sub(pattern, "[REDACTED_PROMPT_OVERRIDE]", sanitized, flags=re.IGNORECASE)
    return sanitized

def validate_uploaded_file(filename: str, content: bytes, max_size_mb: int = 25) -> None:
    """Validate file size and magic bytes for PDF, DOCX, TXT, and Markdown files."""
    max_bytes = max_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {max_size_mb} MB.",
        )

    ext = filename.split(".")[-1].lower() if "." in filename else ""
    allowed_extensions = {"pdf", "docx", "txt", "md"}
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '.{ext}'. Supported: {', '.join(allowed_extensions)}",
        )

    # Magic byte verification
    if ext == "pdf" and not content.startswith(b"%PDF"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content does not match a valid PDF format.",
        )
    elif ext == "docx" and not content.startswith(b"PK"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content does not match a valid DOCX zip archive format.",
        )
