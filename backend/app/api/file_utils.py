"""HTTP-layer file-reading helpers.

Deliberately lives here, not in `app.services`: `UploadFile` is a
Starlette/FastAPI type, and services must stay framework-independent (see
`app/services/document_upload_service.py`'s module docstring). This module
is the boundary that converts an `UploadFile` into plain `bytes` before
anything crosses into the service layer.
"""
from fastapi import UploadFile

from app.services.exceptions import ValidationError

_CHUNK_SIZE = 1024 * 1024  # 1 MiB


async def read_upload_within_limit(file: UploadFile, max_size_bytes: int) -> bytes:
    """Reads `file` in chunks, aborting as soon as `max_size_bytes` is
    exceeded — rather than trusting the client-supplied `Content-Length`
    header (absent or spoofable) or fully buffering an arbitrarily large
    upload before checking its size.
    """
    chunks: list[bytes] = []
    total = 0

    while True:
        chunk = await file.read(_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_size_bytes:
            raise ValidationError(
                f"File exceeds the {max_size_bytes}-byte upload limit."
            )
        chunks.append(chunk)

    return b"".join(chunks)
