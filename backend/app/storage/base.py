"""Storage abstraction — the service layer depends on this, not on any
concrete backend, so a future S3/GCS implementation is a drop-in swap that
never touches `app.services`.
"""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SavedFile:
    """What a storage backend hands back after persisting a file."""

    storage_key: str
    size_bytes: int
    checksum_sha256: str


class FileStorage(Protocol):
    """Minimal contract every storage backend must satisfy."""

    async def save(self, *, content: bytes, original_filename: str) -> SavedFile: ...
