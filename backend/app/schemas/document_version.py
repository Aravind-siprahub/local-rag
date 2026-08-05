"""Schemas for `app.models.document_version.DocumentVersion`."""
import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field

from app.models.enums import DocumentVersionStatus
from app.schemas.common import ORMModel, PaginatedResponse, TimestampSchema

_SHA256_HEX_PATTERN = r"^[0-9a-f]{64}$"


class DocumentVersionBase(BaseModel):
    version_number: Annotated[int, Field(gt=0)]
    storage_key: Annotated[str, Field(min_length=1)]
    original_filename: Annotated[str, Field(min_length=1, max_length=1000)]
    mime_type: Annotated[str, Field(min_length=1, max_length=255)]
    file_size_bytes: Annotated[int, Field(gt=0)]
    checksum_sha256: Annotated[str, Field(pattern=_SHA256_HEX_PATTERN)]
    page_count: Annotated[int, Field(ge=0)] | None = None


class DocumentVersionCreate(DocumentVersionBase):
    document_id: uuid.UUID
    uploaded_by: uuid.UUID


class DocumentVersionUpdate(BaseModel):
    """Covers the pipeline-progress fields a worker updates as a version
    moves through parsing/chunking/embedding/indexing.
    """

    status: DocumentVersionStatus | None = None
    error_message: str | None = None
    page_count: Annotated[int, Field(ge=0)] | None = None
    parsed_at: datetime | None = None
    chunked_at: datetime | None = None
    embedded_at: datetime | None = None
    completed_at: datetime | None = None


class DocumentVersionResponse(DocumentVersionBase, TimestampSchema, ORMModel):
    id: uuid.UUID
    document_id: uuid.UUID
    uploaded_by: uuid.UUID
    status: DocumentVersionStatus
    error_message: str | None = None
    parsed_at: datetime | None = None
    chunked_at: datetime | None = None
    embedded_at: datetime | None = None
    completed_at: datetime | None = None


DocumentVersionListResponse = PaginatedResponse[DocumentVersionResponse]
