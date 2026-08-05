"""Orchestrate embedding jobs for chunked document versions."""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings.client import EmbeddingClient, EmbeddingClientError, OllamaEmbeddingClient
from app.embeddings.generator import EmbeddingGenerator
from app.models.enums import DocumentVersionStatus, ProcessingJobStatus, ProcessingJobType
from app.models.processing_job import ProcessingJob
from app.services.document_chunk_service import DocumentChunkService
from app.services.document_version_service import DocumentVersionService
from app.services.processing_job_service import ProcessingJobService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingWorkerResult:
    """Outcome of a successful embedding job."""

    job_id: uuid.UUID
    document_version_id: uuid.UUID
    embedded_count: int
    skipped_count: int
    duration_ms: int


class EmbeddingWorker:
    """Runs the embedding pipeline for a pending `embed` processing job.

    Transitions document version: chunked → embedding → embedded.
    Uses existing services for persistence and job state management.
    Independent of chat and vector search retrieval.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        client: EmbeddingClient | None = None,
        job_service: ProcessingJobService | None = None,
        chunk_service: DocumentChunkService | None = None,
        version_service: DocumentVersionService | None = None,
        generator: EmbeddingGenerator | None = None,
    ) -> None:
        self.session = session
        self.client = client or OllamaEmbeddingClient()
        self.jobs = job_service or ProcessingJobService(session)
        self.chunks = chunk_service or DocumentChunkService(session)
        self.versions = version_service or DocumentVersionService(session)
        self._generator = generator

    async def process_job(self, job_id: uuid.UUID) -> EmbeddingWorkerResult:
        """Execute embedding for one `embed` processing job."""
        job = await self.jobs.get(job_id)

        if job.job_type != ProcessingJobType.EMBED:
            await self._fail_job(job, f"Unsupported job type {job.job_type.value!r}; only 'embed' is handled.")
            raise ValueError(f"Job {job_id} has unsupported type {job.job_type.value!r}.")

        if job.status not in (ProcessingJobStatus.PENDING, ProcessingJobStatus.RUNNING):
            raise ValueError(f"Job {job_id} is in status {job.status.value!r} and cannot be processed.")

        start_mono = time.monotonic()
        started_at: datetime | None = None
        generator = self._generator or EmbeddingGenerator(self.session, self.client)

        try:
            if job.status == ProcessingJobStatus.PENDING:
                job = await self.jobs.start(job_id)
            started_at = job.started_at

            version = await self.versions.get(job.document_version_id)
            if version.status not in (
                DocumentVersionStatus.CHUNKED,
                DocumentVersionStatus.EMBEDDING,
            ):
                raise ValueError(
                    f"Document version {version.id} is {version.status.value!r}; "
                    "expected 'chunked' or 'embedding' before embedding."
                )

            await self.versions.update(version.id, status=DocumentVersionStatus.EMBEDDING)

            document_chunks = await self.chunks.list_by_document_version(version.id)
            if not document_chunks:
                raise ValueError(f"Document version {version.id} has no chunks to embed.")

            generation = await generator.embed_chunks(document_chunks)

            now = datetime.now(timezone.utc)
            await self.versions.update(
                version.id,
                status=DocumentVersionStatus.EMBEDDED,
                embedded_at=now,
            )

            completed_job = await self.jobs.complete(job_id)
            duration_ms = _compute_duration_ms(start_mono, started_at, completed_job.completed_at)

            logger.info(
                "Embedding job %s completed in %d ms "
                "(embedded=%d, skipped=%d, version=%s)",
                job_id,
                duration_ms,
                generation.embedded_count,
                generation.skipped_count,
                version.id,
            )

            return EmbeddingWorkerResult(
                job_id=job_id,
                document_version_id=version.id,
                embedded_count=generation.embedded_count,
                skipped_count=generation.skipped_count,
                duration_ms=duration_ms,
            )

        except (EmbeddingClientError, ValueError) as exc:
            await self._fail_job(job, str(exc))
            logger.warning("Embedding job %s failed: %s", job_id, exc)
            raise

        except Exception as exc:
            await self._fail_job(job, str(exc))
            logger.exception("Embedding job %s failed unexpectedly", job_id)
            raise

    async def close(self) -> None:
        await self.client.close()

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
            logger.exception("Failed to record failure for embedding job %s", job.id)


def _compute_duration_ms(
    start_mono: float,
    started_at: datetime | None,
    completed_at: datetime | None,
) -> int:
    if started_at is not None and completed_at is not None:
        delta = completed_at - started_at
        return max(0, int(delta.total_seconds() * 1000))
    return max(0, int((time.monotonic() - start_mono) * 1000))
