"""Local filesystem implementation of `FileStorage`."""
import asyncio
import hashlib
import uuid
from pathlib import Path

from app.core.config import get_settings
from app.storage.base import SavedFile


class LocalFileStorage:
    """Saves files under a configurable local directory (`UPLOAD_DIR`).

    `storage_key` is a random filename (not the client's original filename)
    to avoid path-traversal and collision issues; the original filename and
    content type are recorded separately on `DocumentVersion`.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        settings = get_settings()
        self.base_dir = Path(base_dir if base_dir is not None else settings.UPLOAD_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, *, content: bytes, original_filename: str) -> SavedFile:
        # File I/O is blocking; hand it to a worker thread so it doesn't
        # block the event loop the rest of the async app runs on.
        return await asyncio.to_thread(self._save_sync, content, original_filename)

    def _save_sync(self, content: bytes, original_filename: str) -> SavedFile:
        suffix = Path(original_filename).suffix.lower()
        storage_key = f"{uuid.uuid4()}{suffix}"
        destination = self.base_dir / storage_key
        destination.write_bytes(content)

        return SavedFile(
            storage_key=storage_key,
            size_bytes=len(content),
            checksum_sha256=hashlib.sha256(content).hexdigest(),
        )
