"""Orchestrate parse → clean → chunk and persist results via existing services."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.enums import DocumentVersionStatus, ProcessingJobStatus, ProcessingJobType
from app.models.processing_job import ProcessingJob
from app.processing.cleaner import clean_text
from app.processing.chunker import chunk_text
from app.processing.parser import ParsingError, parse_file
from app.services.document_chunk_service import ChunkInput, DocumentChunkService
from app.services.document_version_service import DocumentVersionService
from app.services.processing_job_service import ProcessingJobService

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
    ) -> None:
        settings = get_settings()
        self.session = session
        self.upload_dir = Path(upload_dir if upload_dir is not None else settings.UPLOAD_DIR)
        self.chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
        self.jobs = job_service or ProcessingJobService(session)
        self.chunks = chunk_service or DocumentChunkService(session)
        self.versions = version_service or DocumentVersionService(session)

    async def process_job(self, job_id: uuid.UUID) -> ProcessingResult:
        """Execute the full pipeline for one processing job.

        Transitions the job pending → running → completed (or failed on error).
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
            await self.versions.update(
                version.id,
                status=DocumentVersionStatus.PARSING,
            )

            raw_bytes = await self._load_file(version.storage_key)
            raw_text = parse_file(raw_bytes, version.original_filename, version.mime_type)
            cleaned = clean_text(raw_text)
            text_chunks = chunk_text(cleaned, self.chunk_size, self.chunk_overlap)

            chunk_inputs: list[ChunkInput] = [
                {
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                }
                for chunk in text_chunks
            ]

            await self.chunks.create_chunks_for_version(version.id, chunk_inputs)

            now = datetime.now(timezone.utc)
            await self.versions.update(
                version.id,
                status=DocumentVersionStatus.CHUNKED,
                parsed_at=now,
                chunked_at=now,
            )

            completed_job = await self.jobs.complete(job_id)
            duration_ms = _compute_duration_ms(start_mono, started_at, completed_job.completed_at)

            logger.info(
                "Processing job %s completed in %d ms (%d chunks for version %s)",
                job_id,
                duration_ms,
                len(text_chunks),
                version.id,
            )

            return ProcessingResult(
                job_id=job_id,
                document_version_id=version.id,
                chunk_count=len(text_chunks),
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

    async def _load_file(self, storage_key: str) -> bytes:
        path = self.upload_dir / storage_key
        if not path.is_file():
            raise FileNotFoundError(f"Uploaded file not found at {path!r}.")

        return await asyncio.to_thread(path.read_bytes)

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
