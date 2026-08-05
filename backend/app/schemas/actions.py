"""Request bodies for non-CRUD action endpoints.

These don't belong in the domain schema files alongside `*Create`/`*Update`:
they exist purely because an endpoint's request shape differs from the
domain's own Create/Update schema (e.g. `document_version_number` is
server-computed, so the upload endpoint's body must not require it).
"""
import uuid
from typing import Annotated

from pydantic import BaseModel, Field

from app.schemas.system_setting import SystemSettingBase


class SetCurrentVersionRequest(BaseModel):
    version_id: uuid.UUID


class DocumentVersionUploadRequest(BaseModel):
    """Body for `POST /document-versions`. Same fields as
    `DocumentVersionCreate` minus `version_number` — the service computes
    that automatically (see `DocumentVersionService.create_next_version`).
    """

    document_id: uuid.UUID
    uploaded_by: uuid.UUID
    storage_key: Annotated[str, Field(min_length=1)]
    original_filename: Annotated[str, Field(min_length=1, max_length=1000)]
    mime_type: Annotated[str, Field(min_length=1, max_length=255)]
    file_size_bytes: Annotated[int, Field(gt=0)]
    checksum_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    page_count: Annotated[int, Field(ge=0)] | None = None


class FailJobRequest(BaseModel):
    error_message: Annotated[str, Field(min_length=1)]


class SystemSettingUpsertRequest(SystemSettingBase):
    """Body for `PUT /system-settings/{key}` — create-or-replace."""

    updated_by: uuid.UUID | None = None
