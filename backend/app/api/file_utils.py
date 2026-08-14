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


def validate_image_bytes(content: bytes) -> str:
    """Validate image format by inspecting magic bytes.
    Returns the MIME type (e.g. 'image/png') if valid, else raises ValidationError.
    """
    if len(content) < 12:
        raise ValidationError("File is too small to be a valid image.")
    
    # Check PNG
    if content.startswith(b'\x89PNG\r\n\x1a\n'):
        return "image/png"
    
    # Check JPEG
    if content.startswith(b'\xff\xd8\xff'):
        return "image/jpeg"
    
    # Check WEBP
    if content.startswith(b'RIFF') and content[8:12] == b'WEBP':
        return "image/webp"
        
    raise ValidationError("Unsupported or malformed image format. Only PNG, JPEG, and WEBP are supported.")


def resize_image(image_bytes: bytes, max_size: int = 1024) -> bytes:
    """Resize image to fit within max_size x max_size while maintaining aspect ratio."""
    try:
        from PIL import Image
        import io
        
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            # Use Resampling.LANCZOS for quality
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            # Save using original format or PNG/JPEG
            img.save(output, format=img.format or "PNG")
            return output.getvalue()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Failed to resize image: %s", e)
    return image_bytes
