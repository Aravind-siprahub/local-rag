"""Unit tests for `app.embeddings.worker`."""
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from app.embeddings.generator import EmbeddingGenerationResult
from app.models.enums import DocumentVersionStatus, ProcessingJobStatus, ProcessingJobType
from app.embeddings.worker import EmbeddingWorker


@dataclass
class _FakeChunk:
    id: uuid.UUID
    chunk_index: int
    content: str


@dataclass
class _FakeVersion:
    id: uuid.UUID
    status: DocumentVersionStatus
    error_message: str | None = None


@dataclass
class _FakeJob:
    id: uuid.UUID
    document_version_id: uuid.UUID
    job_type: ProcessingJobType
    status: ProcessingJobStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None


class FakeEmbeddingClient:
    async def embed(self, text: str) -> list[float]:
        return [0.1] * 768

    async def close(self) -> None:
        return None


class FakeGenerator:
    def __init__(self, *, embedded: int = 2, skipped: int = 0) -> None:
        self.embedded = embedded
        self.skipped = skipped
        self.calls: list[list] = []

    async def embed_chunks(self, chunks: list) -> EmbeddingGenerationResult:
        self.calls.append(chunks)
        return EmbeddingGenerationResult(
            embedded_count=self.embedded,
            skipped_count=self.skipped,
            total_chunks=len(chunks),
        )


class FakeProcessingJobService:
    def __init__(self, job: _FakeJob) -> None:
        self.job = job
        self.started: list[uuid.UUID] = []
        self.completed: list[uuid.UUID] = []
        self.failed: list[tuple[uuid.UUID, str]] = []

    async def get(self, job_id: uuid.UUID) -> _FakeJob:
        if job_id != self.job.id:
            raise KeyError(job_id)
        return self.job

    async def start(self, job_id: uuid.UUID) -> _FakeJob:
        self.started.append(job_id)
        self.job.status = ProcessingJobStatus.RUNNING
        self.job.started_at = datetime.now(timezone.utc)
        return self.job

    async def complete(self, job_id: uuid.UUID) -> _FakeJob:
        self.completed.append(job_id)
        self.job.status = ProcessingJobStatus.COMPLETED
        self.job.completed_at = datetime.now(timezone.utc)
        return self.job

    async def fail(self, job_id: uuid.UUID, error_message: str) -> _FakeJob:
        self.failed.append((job_id, error_message))
        self.job.status = ProcessingJobStatus.FAILED
        self.job.error_message = error_message
        self.job.completed_at = datetime.now(timezone.utc)
        return self.job


class FakeDocumentChunkService:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self.chunks = chunks

    async def list_by_document_version(self, document_version_id: uuid.UUID, **kwargs) -> list[_FakeChunk]:
        return self.chunks


class FakeDocumentVersionService:
    def __init__(self, version: _FakeVersion) -> None:
        self.version = version
        self.updates: list[dict] = []

    async def get(self, version_id: uuid.UUID) -> _FakeVersion:
        if version_id != self.version.id:
            raise KeyError(version_id)
        return self.version

    async def update(self, version_id: uuid.UUID, **values) -> _FakeVersion:
        self.updates.append(values)
        for key, value in values.items():
            setattr(self.version, key, value)
        return self.version


def _make_worker(
    tmp_path=None,
    *,
    chunks: list[_FakeChunk] | None = None,
    version_status: DocumentVersionStatus = DocumentVersionStatus.CHUNKED,
    generator: FakeGenerator | None = None,
) -> tuple[EmbeddingWorker, _FakeJob, FakeProcessingJobService, FakeDocumentVersionService, FakeGenerator]:
    version_id = uuid.uuid4()
    version = _FakeVersion(id=version_id, status=version_status)
    if chunks is None:
        chunks = [
            _FakeChunk(id=uuid.uuid4(), chunk_index=0, content="one"),
            _FakeChunk(id=uuid.uuid4(), chunk_index=1, content="two"),
        ]
    job = _FakeJob(
        id=uuid.uuid4(),
        document_version_id=version_id,
        job_type=ProcessingJobType.EMBED,
        status=ProcessingJobStatus.PENDING,
    )
    jobs = FakeProcessingJobService(job)
    versions = FakeDocumentVersionService(version)
    gen = generator or FakeGenerator(embedded=len(chunks))
    worker = EmbeddingWorker(
        session=None,
        client=FakeEmbeddingClient(),
        job_service=jobs,
        chunk_service=FakeDocumentChunkService(chunks),
        version_service=versions,
        generator=gen,
    )
    return worker, job, jobs, versions, gen


class TestEmbeddingWorker:
    @pytest.mark.asyncio
    async def test_successful_job_embeds_and_completes(self) -> None:
        worker, job, jobs, versions, gen = _make_worker()

        result = await worker.process_job(job.id)

        assert result.embedded_count == 2
        assert result.skipped_count == 0
        assert result.duration_ms >= 0
        assert jobs.started == [job.id]
        assert jobs.completed == [job.id]
        assert job.status == ProcessingJobStatus.COMPLETED
        assert versions.version.status == DocumentVersionStatus.EMBEDDED
        assert len(gen.calls) == 1
        status_path = [update.get("status") for update in versions.updates]
        assert DocumentVersionStatus.EMBEDDING in status_path
        assert DocumentVersionStatus.EMBEDDED in status_path

    @pytest.mark.asyncio
    async def test_skipped_duplicates_reported(self) -> None:
        gen = FakeGenerator(embedded=1, skipped=1)
        worker, job, jobs, versions, _ = _make_worker(generator=gen)

        result = await worker.process_job(job.id)

        assert result.embedded_count == 1
        assert result.skipped_count == 1
        assert job.status == ProcessingJobStatus.COMPLETED
        assert versions.version.status == DocumentVersionStatus.EMBEDDED

    @pytest.mark.asyncio
    async def test_no_chunks_fails_job(self) -> None:
        worker, job, jobs, versions, _ = _make_worker(chunks=[])

        with pytest.raises(ValueError, match="no chunks"):
            await worker.process_job(job.id)

        assert job.status == ProcessingJobStatus.FAILED
        assert jobs.failed
        assert versions.version.status == DocumentVersionStatus.FAILED

    @pytest.mark.asyncio
    async def test_wrong_version_status_fails(self) -> None:
        worker, job, jobs, _, _ = _make_worker(version_status=DocumentVersionStatus.UPLOADED)

        with pytest.raises(ValueError, match="expected 'chunked'"):
            await worker.process_job(job.id)

        assert job.status == ProcessingJobStatus.FAILED

    @pytest.mark.asyncio
    async def test_unsupported_job_type_fails(self) -> None:
        worker, job, jobs, _, _ = _make_worker()
        job.job_type = ProcessingJobType.PARSE

        with pytest.raises(ValueError, match="unsupported type"):
            await worker.process_job(job.id)

        assert job.status == ProcessingJobStatus.FAILED
