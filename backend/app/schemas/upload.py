"""Response schema for `POST /documents/upload`."""
import uuid

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    id: uuid.UUID
    filename: str
    bucket: str = "documents"
    storagePath: str
    status: str = "Pending"

    # Backward-compatible fields
    document_id: uuid.UUID | None = None
    version_id: uuid.UUID | None = None
    processing_job_id: uuid.UUID | None = None
    original_filename: str | None = None
    mime_type: str | None = None
    file_size_bytes: int | None = None
    checksum_sha256: str | None = None
    storage_key: str | None = None
