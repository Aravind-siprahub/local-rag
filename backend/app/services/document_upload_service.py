"""Orchestrates the document upload workflow.

Deliberately separate from `DocumentService`/`DocumentVersionService`: this
is a *composition* service that coordinates existing services plus file
storage to implement one user-facing workflow (upload), while
`DocumentService` etc. remain focused on their own single entity. It stays
independent of FastAPI — it takes plain `bytes`, not Starlette's
`UploadFile` — so it can be unit tested with fakes and no HTTP layer at all
(see `tests/test_document_upload_service.py`).

Transaction note: `DocumentService.create_document`,
`DocumentVersionService.create_next_version`, and
`ProcessingJobService.create_job` each commit their own unit of work —
that's how they were built, and this step must not modify them — so this
workflow is *not* one atomic database transaction across all three writes.
If a later step fails, `_compensate_*` best-effort undoes what the earlier
steps already committed (a Saga-style compensation), rather than silently
leaving an inconsistent partial upload.
"""
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.models.enums import ProcessingJobType
from app.models.processing_job import ProcessingJob
from app.repositories.user_repository import UserRepository
from app.services.document_service import DocumentService
from app.services.document_version_service import DocumentVersionService
from app.services.exceptions import ValidationError
from app.services.owner_resolution import resolve_owner_user_id
from app.services.processing_job_service import ProcessingJobService
from app.storage.base import FileStorage
from app.storage.local_file_storage import LocalFileStorage

logger = logging.getLogger(__name__)

# Extension -> content types accepted for it. Client-supplied Content-Type
# is unreliable in practice (many browsers/OSes send "application/
# octet-stream" for .md, some send nothing) — extension is the primary
# signal, Content-Type is a secondary sanity check with a generic fallback.
ALLOWED_EXTENSIONS: dict[str, tuple[str, ...]] = {
    ".pdf": ("application/pdf",),
    ".docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",),
    ".doc": ("application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ".txt": ("text/plain",),
    ".md": ("text/markdown", "text/x-markdown", "text/plain"),
    ".markdown": ("text/markdown", "text/x-markdown", "text/plain"),
    ".csv": ("text/csv", "application/csv", "text/plain"),
    ".xlsx": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"),
    ".pptx": ("application/vnd.openxmlformats-officedocument.presentationml.presentation",),
    ".json": ("application/json", "text/plain"),
}
_GENERIC_CONTENT_TYPES = frozenset({"application/octet-stream", "", None})


@dataclass(frozen=True)
class UploadResult:
    """Everything `POST /documents/upload` needs to build its response."""

    document_id: uuid.UUID
    version_id: uuid.UUID
    processing_job_id: uuid.UUID
    original_filename: str
    mime_type: str
    file_size_bytes: int
    checksum_sha256: str
    storage_key: str
    bucket_name: str
    storage_path: str
    storage_provider: str


class DocumentUploadService:
    def __init__(
        self,
        session: AsyncSession,
        storage: FileStorage,
        *,
        document_service: DocumentService | None = None,
        version_service: DocumentVersionService | None = None,
        job_service: ProcessingJobService | None = None,
    ) -> None:
        self.session = session
        self.storage = storage
        # Constructor-injected with sensible defaults: production code gets
        # real services bound to the same session; unit tests pass fakes
        # (see tests/test_document_upload_service.py) without touching a
        # database or the filesystem.
        self.documents = document_service or DocumentService(session)
        self.versions = version_service or DocumentVersionService(session)
        self.jobs = job_service or ProcessingJobService(session)

    def validate_file(
        self, *, filename: str, content_type: str | None, size_bytes: int, max_size_bytes: int
    ) -> None:
        """Pure, I/O-free validation — trivial to unit test on its own.

        Raises `ValidationError` (mapped to HTTP 422 by
        `app/core/exceptions.py`) on any failure.
        """
        if size_bytes <= 0:
            raise ValidationError("Uploaded file is empty.")
        if size_bytes > max_size_bytes:
            raise ValidationError(
                f"File is {size_bytes} bytes, exceeding the {max_size_bytes}-byte upload limit."
            )

        extension = Path(filename).suffix.lower()
        allowed_content_types = ALLOWED_EXTENSIONS.get(extension)
        if allowed_content_types is None:
            raise ValidationError(
                f"Unsupported file extension {extension!r}. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            )
        if content_type not in allowed_content_types and content_type not in _GENERIC_CONTENT_TYPES:
            raise ValidationError(
                f"Content-Type {content_type!r} does not match a {extension!r} file."
            )

    async def upload(
        self,
        *,
        user_id: uuid.UUID,
        filename: str,
        content_type: str | None,
        content: bytes,
        max_size_bytes: int,
        title: str | None = None,
    ) -> UploadResult:
        """Validate -> store on disk -> create Document -> create
        DocumentVersion -> create a pending ProcessingJob.

        Deliberately stops there: parsing/chunking/embedding the file is a
        separate concern (a future worker picks up the pending job) — this
        method's job is upload bookkeeping only.
        """
        self.validate_file(
            filename=filename, content_type=content_type, size_bytes=len(content), max_size_bytes=max_size_bytes
        )

        extension = Path(filename).suffix.lower()
        allowed_types = ALLOWED_EXTENSIONS.get(extension, ["application/octet-stream"])
        resolved_content_type = content_type if (content_type and content_type not in _GENERIC_CONTENT_TYPES) else allowed_types[0]

        # Resolve once up front so Document.user_id and DocumentVersion.uploaded_by
        # always share the same real user (Swagger placeholder UUIDs included).
        # Skipped when session is absent (unit tests inject fakes only).
        if self.session is not None:
            user_id = await resolve_owner_user_id(user_id, UserRepository(self.session))

        document: Document = await self.documents.create_document(user_id=user_id, title=title or filename)
        document_id = document.id

        # Construct storage path: user_id/document_id/original_filename
        storage_path = f"{user_id}/{document_id}/{filename}".replace("//", "/").lstrip("/")
        bucket_name = getattr(self.storage, "bucket_name", "documents")
        storage_provider = getattr(self.storage, "storage_provider", "supabase")

        logger.info(
            "UPLOAD: bucket=%s path=%s filename=%s document_id=%s",
            bucket_name,
            storage_path,
            filename,
            document_id,
        )

        saved = None
        try:
            if hasattr(self.storage, "upload_file"):
                saved = await self.storage.upload_file(
                    content=content,
                    storage_path=storage_path,
                    mime_type=resolved_content_type,
                )
                try:
                    local_destination = Path(get_settings().UPLOAD_DIR) / storage_path.lstrip("/")
                    local_destination.parent.mkdir(parents=True, exist_ok=True)
                    import asyncio
                    await asyncio.to_thread(local_destination.write_bytes, content)
                except Exception as exc:
                    logger.debug("Failed to write local backup copy: %s", exc)
            else:
                saved = await self.storage.save(content=content, original_filename=filename)
        except Exception as exc:
            logger.error("Supabase Storage upload failed for file %s: %s", filename, exc)
            await self._compensate_document(document_id)
            raise

        version: DocumentVersion
        try:
            version = await self.versions.create_next_version(
                document_id=document_id,
                uploaded_by=user_id,
                storage_key=saved.storage_key,
                original_filename=filename,
                mime_type=resolved_content_type,
                file_size_bytes=saved.size_bytes,
                checksum_sha256=saved.checksum_sha256,
            )
            await self.documents.update(
                document_id,
                current_version_id=version.id,
                storage_provider=storage_provider,
                bucket_name=bucket_name,
                storage_path=storage_path,
                original_filename=filename,
                filename=filename,
                mime_type=resolved_content_type,
                size=saved.size_bytes,
                file_size_bytes=saved.size_bytes,
            )
            await self.versions.update(
                version.id,
                storage_provider=storage_provider,
                bucket_name=bucket_name,
                storage_path=storage_path,
            )
        except Exception as exc:
            logger.error("Version creation failed after upload of %s; deleting remote file.", storage_path)
            if hasattr(self.storage, "delete_file"):
                await self.storage.delete_file(storage_path=storage_path)
            await self._compensate_document(document_id)
            raise

        version_id = version.id

        job: ProcessingJob
        try:
            job = await self.jobs.create_job(
                document_version_id=version_id, job_type=ProcessingJobType.PARSE
            )
        except Exception:
            logger.error("Job creation failed for version %s; compensating.", version_id)
            if hasattr(self.storage, "delete_file"):
                await self.storage.delete_file(storage_path=storage_path)
            await self._compensate_version(version_id)
            await self._compensate_document(document_id)
            raise

        return UploadResult(
            document_id=document_id,
            version_id=version_id,
            processing_job_id=job.id,
            original_filename=filename,
            mime_type=resolved_content_type,
            file_size_bytes=saved.size_bytes,
            checksum_sha256=saved.checksum_sha256,
            storage_key=saved.storage_key,
            bucket_name=bucket_name,
            storage_path=storage_path,
            storage_provider=storage_provider,
        )

    async def _compensate_document(self, document_id: uuid.UUID) -> None:
        """Best-effort cleanup. `DocumentService.delete` soft-deletes (sets
        `deleted_at`) rather than removing the row — that's the only delete
        behavior it exposes, and this step must not change that — so an
        orphaned document from a failed upload is marked deleted rather
        than truly erased.
        """
        try:
            await self.documents.delete(document_id)
        except Exception:
            logger.exception("Compensation failed while soft-deleting document %s", document_id)

    async def _compensate_version(self, version_id: uuid.UUID) -> None:
        try:
            await self.versions.delete(version_id)
        except Exception:
            logger.exception("Compensation failed while deleting document_version %s", version_id)
