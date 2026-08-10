"""Unit tests for `app.processing.processor`.

Uses fakes for services and a temp upload directory — no database or HTTP.
"""
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from app.models.enums import DocumentVersionStatus, ProcessingJobStatus, ProcessingJobType
from app.processing.parser import CorruptedFileError
from app.processing.processor import DocumentProcessor


@dataclass
class _FakeVersion:
  id: uuid.UUID
  storage_key: str
  original_filename: str
  mime_type: str


@dataclass
class _FakeJob:
  id: uuid.UUID
  document_version_id: uuid.UUID
  job_type: ProcessingJobType
  status: ProcessingJobStatus
  started_at: datetime | None = None
  completed_at: datetime | None = None
  error_message: str | None = None


@dataclass
class _FakeChunk:
  chunk_index: int
  content: str
  char_start: int | None
  char_end: int | None


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
  def __init__(self) -> None:
    self.created: list[tuple[uuid.UUID, list]] = []

  async def create_chunks_for_version(self, document_version_id: uuid.UUID, chunks: list) -> list[_FakeChunk]:
    self.created.append((document_version_id, chunks))
    return [
      _FakeChunk(
        chunk_index=chunk["chunk_index"],
        content=chunk["content"],
        char_start=chunk.get("char_start"),
        char_end=chunk.get("char_end"),
      )
      for chunk in chunks
    ]


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


def _make_processor(
  tmp_path,
  *,
  filename: str = "notes.txt",
  content: bytes = (
    b"Hello world from the semantic chunking pipeline test fixture.\n\n"
    b"Second paragraph with enough meaningful content to pass validation rules."
  ),
  job_status: ProcessingJobStatus = ProcessingJobStatus.PENDING,
) -> tuple[DocumentProcessor, _FakeJob, FakeProcessingJobService, FakeDocumentChunkService, FakeDocumentVersionService]:
  version_id = uuid.uuid4()
  storage_key = f"test-{uuid.uuid4()}.txt"
  (tmp_path / storage_key).write_bytes(content)

  version = _FakeVersion(
    id=version_id,
    storage_key=storage_key,
    original_filename=filename,
    mime_type="text/plain",
  )
  job = _FakeJob(
    id=uuid.uuid4(),
    document_version_id=version_id,
    job_type=ProcessingJobType.PARSE,
    status=job_status,
  )

  jobs = FakeProcessingJobService(job)
  chunks = FakeDocumentChunkService()
  versions = FakeDocumentVersionService(version)

  processor = DocumentProcessor(
    session=None,
    upload_dir=tmp_path,
    chunk_size=50,
    chunk_overlap=10,
    job_service=jobs,
    chunk_service=chunks,
    version_service=versions,
  )
  return processor, job, jobs, chunks, versions


class TestDocumentProcessor:
  @pytest.mark.asyncio
  async def test_successful_processing_stores_chunks_and_completes_job(self, tmp_path) -> None:
    processor, job, jobs, chunks, versions = _make_processor(tmp_path)

    result = await processor.process_job(job.id)

    assert result.chunk_count >= 1
    assert result.document_version_id == job.document_version_id
    assert result.duration_ms >= 0
    assert jobs.started == [job.id]
    assert jobs.completed == [job.id]
    assert len(chunks.created) == 1
    stored_chunks = chunks.created[0][1]
    assert len(stored_chunks) == result.chunk_count
    assert all("chunk_index" in chunk for chunk in stored_chunks)
    assert all("char_start" in chunk for chunk in stored_chunks)
    assert all("char_end" in chunk for chunk in stored_chunks)
    assert job.status == ProcessingJobStatus.COMPLETED
    assert versions.version.status == DocumentVersionStatus.CHUNKED

  @pytest.mark.asyncio
  async def test_missing_file_fails_job(self, tmp_path) -> None:
    version_id = uuid.uuid4()
    version = _FakeVersion(
      id=version_id,
      storage_key="missing.txt",
      original_filename="missing.txt",
      mime_type="text/plain",
    )
    job = _FakeJob(
      id=uuid.uuid4(),
      document_version_id=version_id,
      job_type=ProcessingJobType.PARSE,
      status=ProcessingJobStatus.PENDING,
    )
    jobs = FakeProcessingJobService(job)
    versions = FakeDocumentVersionService(version)
    processor = DocumentProcessor(
      session=None,
      upload_dir=tmp_path,
      job_service=jobs,
      chunk_service=FakeDocumentChunkService(),
      version_service=versions,
    )

    with pytest.raises(FileNotFoundError):
      await processor.process_job(job.id)

    assert job.status == ProcessingJobStatus.FAILED
    assert jobs.failed
    assert versions.version.status == DocumentVersionStatus.FAILED

  @pytest.mark.asyncio
  async def test_corrupted_file_fails_job(self, tmp_path) -> None:
    processor, job, jobs, _, versions = _make_processor(
      tmp_path,
      filename="broken.pdf",
      content=b"not-a-pdf",
    )

    with pytest.raises(CorruptedFileError):
      await processor.process_job(job.id)

    assert job.status == ProcessingJobStatus.FAILED
    assert jobs.failed
    assert versions.version.status == DocumentVersionStatus.FAILED

  @pytest.mark.asyncio
  async def test_processes_docx_file(self, tmp_path) -> None:
    from tests.test_processing_parser import _make_docx

    docx_bytes = _make_docx([
      "Processor DOCX line one with sufficient length for semantic chunk validation.",
      "Line two also contains enough meaningful characters to produce a valid chunk.",
    ])
    processor, job, jobs, chunks, _ = _make_processor(
      tmp_path,
      filename="report.docx",
      content=docx_bytes,
    )
  # Fix storage key extension for docx
    version_service = processor.versions
    storage_key = f"test-{uuid.uuid4()}.docx"
    (tmp_path / storage_key).write_bytes(docx_bytes)
    version_service.version.storage_key = storage_key
    version_service.version.original_filename = "report.docx"
    version_service.version.mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    result = await processor.process_job(job.id)

    assert result.chunk_count >= 1
    assert job.status == ProcessingJobStatus.COMPLETED
    assert "Processor DOCX" in chunks.created[0][1][0]["content"]

  @pytest.mark.asyncio
  async def test_unsupported_job_type_fails(self, tmp_path) -> None:
    processor, job, jobs, _, _ = _make_processor(tmp_path)
    job.job_type = ProcessingJobType.EMBED

    with pytest.raises(ValueError, match="unsupported type"):
      await processor.process_job(job.id)

    assert job.status == ProcessingJobStatus.FAILED
