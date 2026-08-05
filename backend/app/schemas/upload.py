"""Response schema for `POST /documents/upload`."""
import uuid

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    document_id: uuid.UUID
    version_id: uuid.UUID
    processing_job_id: uuid.UUID
    original_filename: str
    mime_type: str
    file_size_bytes: int
    checksum_sha256: str
    storage_key: str
