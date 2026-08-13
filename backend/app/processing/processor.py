"""Orchestrate parse -> clean -> chunk and persist results via existing services."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.enums import DocumentVersionStatus, ProcessingJobStatus, ProcessingJobType
from app.models.processing_job import ProcessingJob
from app.services.chunker import chunk_document
from app.services.embedding import normalize_text_for_embedding
from app.services.parser import DocumentParser, ParsingError
from app.services.document_chunk_service import ChunkInput, DocumentChunkService
from app.services.document_version_service import DocumentVersionService
from app.services.processing_job_service import ProcessingJobService
from app.storage.s3_storage_service import S3StorageService
from app.storage.supabase_storage_service import SupabaseStorageService

if TYPE_CHECKING:
    from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessingResult:
    """Outcome of a successful document processing run."""

    job_id: uuid.UUID
    document_version_id: uuid.UUID
    chunk_count: int
    duration_ms: int


class DocumentProcessor:
    """Runs the text extraction pipeline for a pending processing job.

    Uses existing services for persistence and job state transitions. Does not
    touch embeddings, Ollama, or vector search.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        upload_dir: str | Path | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        job_service: ProcessingJobService | None = None,
        chunk_service: DocumentChunkService | None = None,
        version_service: DocumentVersionService | None = None,
        document_service: DocumentService | None = None,
    ) -> None:
        settings = get_settings()
        self.session = session
        self.upload_dir = Path(upload_dir if upload_dir is not None else settings.UPLOAD_DIR)
        self.chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
        self.document_parser = DocumentParser()
        self.jobs = job_service or ProcessingJobService(session)
        self.chunks = chunk_service or DocumentChunkService(session)
        self.versions = version_service or DocumentVersionService(session)
        if document_service is not None:
            self.documents = document_service
        elif session is not None:
            from app.services.document_service import DocumentService
            self.documents = DocumentService(session)
        else:
            self.documents = None

    async def process_job(self, job_id: uuid.UUID) -> ProcessingResult:
        """Execute the full pipeline for one processing job.

        Transitions the job pending -> running -> completed (or failed on error).
        Updates the parent document version status and timestamps on success.
        """
        job = await self.jobs.get(job_id)

        if job.job_type != ProcessingJobType.PARSE:
            await self._fail_job(job, f"Unsupported job type {job.job_type.value!r}; only 'parse' is handled.")
            raise ValueError(f"Job {job_id} has unsupported type {job.job_type.value!r}.")

        if job.status not in (ProcessingJobStatus.PENDING, ProcessingJobStatus.RUNNING):
            raise ValueError(f"Job {job_id} is in status {job.status.value!r} and cannot be processed.")

        start_mono = time.monotonic()
        started_at: datetime | None = None

        try:
            if job.status == ProcessingJobStatus.PENDING:
                job = await self.jobs.start(job_id)
            started_at = job.started_at

            version = await self.versions.get(job.document_version_id)
            doc_id = getattr(version, "document_id", getattr(version, "id", None))
            document = await self.documents.get(doc_id) if self.documents and doc_id else None

            await self.versions.update(
                version.id,
                status=DocumentVersionStatus.PARSING,
            )

            raw_bytes = await self._load_file(version, getattr(document, "storage_path", None))
            parsed_doc = await self.document_parser.parse(
                raw_bytes,
                version.original_filename,
                getattr(version, "document_id", version.id),
                version.mime_type,
            )
            
            now_parsed = datetime.now(timezone.utc)
            await self.versions.update(
                version.id,
                status=DocumentVersionStatus.PARSED,
                parsed_at=now_parsed,
            )

            semantic_chunks = chunk_document(parsed_doc)

            chunk_inputs: list[ChunkInput] = [
                {
                    "chunk_index": sc.chunk_index,
                    "content": normalize_text_for_embedding(sc.text),
                    "content_tokens": sc.token_count,
                    "page_number": sc.page_number or None,
                    "section_title": sc.breadcrumb or sc.section or None,
                    "char_start": sc.char_start,
                    "char_end": sc.char_end,
                    "metadata_": sc.to_metadata_dict(),
                }
                for sc in semantic_chunks
            ]

            await self.chunks.create_chunks_for_version(version.id, chunk_inputs)

            now = datetime.now(timezone.utc)
            await self.versions.update(
                version.id,
                status=DocumentVersionStatus.CHUNKED,
                chunked_at=now,
            )

            completed_job = await self.jobs.complete(job_id)
            duration_ms = _compute_duration_ms(start_mono, started_at, completed_job.completed_at)

            logger.info(
                "Processing job %s completed in %d ms (%d chunks for version %s)",
                job_id,
                duration_ms,
                len(semantic_chunks),
                version.id,
            )

            return ProcessingResult(
                job_id=job_id,
                document_version_id=version.id,
                chunk_count=len(semantic_chunks),
                duration_ms=duration_ms,
            )

        except ParsingError as exc:
            await self._fail_job(job, str(exc))
            logger.warning("Processing job %s failed during parsing: %s", job_id, exc)
            raise

        except Exception as exc:
            await self._fail_job(job, str(exc))
            logger.exception("Processing job %s failed unexpectedly", job_id)
            raise

    async def _load_file(self, version: Any, doc_storage_path: str | None = None) -> bytes:
        version_storage_path = (getattr(version, "storage_path", None) or "").strip().lstrip("/")
        version_storage_key = (getattr(version, "storage_key", None) or "").strip().lstrip("/")
        doc_storage_path = (doc_storage_path or "").strip().lstrip("/")

        settings = get_settings()
        if settings.s3_is_configured:
            storage_service = S3StorageService()
        else:
            storage_service = SupabaseStorageService()

        raw_bytes: bytes | None = None

        if storage_service.is_configured:
            remote_candidates: list[str] = []
            for candidate in (version_storage_path, version_storage_key, doc_storage_path):
                if candidate and candidate not in remote_candidates:
                    remote_candidates.append(candidate)

            for candidate_path in remote_candidates:
                try:
                    exists = await storage_service.exists_file(storage_path=candidate_path)
                except Exception:
                    exists = False

                if exists:
                    try:
                        raw_bytes = await storage_service.download_file(storage_path=candidate_path)
                        # Persist working path
                        if candidate_path != version_storage_path:
                            try:
                                await self.versions.update(version.id, storage_path=candidate_path)
                            except Exception:
                                pass
                        break
                    except Exception:
                        pass

        # Local fallback
        if raw_bytes is None:
            local_candidates = []
            for p in (version_storage_path, version_storage_key, doc_storage_path):
                if p:
                    local_candidates.append(self.upload_dir / p)
            if version_storage_key:
                local_candidates.append(self.upload_dir / version_storage_key)

            for local_path in local_candidates:
                if local_path.is_file():
                    raw_bytes = await asyncio.to_thread(local_path.read_bytes)
                    break

        if raw_bytes is None:
            raise FileNotFoundError(f"File not found remotely or locally for version {version.id!r}.")

        return raw_bytes

    async def _fail_job(self, job: ProcessingJob, error_message: str) -> None:
        try:
            if job.status == ProcessingJobStatus.PENDING:
                await self.jobs.start(job.id)
            await self.jobs.fail(job.id, error_message)
            await self.versions.update(
                job.document_version_id,
                status=DocumentVersionStatus.FAILED,
                error_message=error_message,
            )
        except Exception:
            logger.exception("Failed to record failure for processing job %s", job.id)


def _compute_duration_ms(
    start_mono: float,
    started_at: datetime | None,
    completed_at: datetime | None,
) -> int:
    """Prefer wall-clock job timestamps when available; fall back to monotonic."""
    if started_at is not None and completed_at is not None:
        delta = completed_at - started_at
        return max(0, int(delta.total_seconds() * 1000))
    return max(0, int((time.monotonic() - start_mono) * 1000))
