"""Unit tests for `DocumentUploadService`.

None of these touch a real database or the real filesystem: `FileStorage`
and the three composed services (`DocumentService`, `DocumentVersionService`,
`ProcessingJobService`) are faked. This is the payoff of
`DocumentUploadService` depending on abstractions (constructor injection)
rather than constructing its own dependencies — exactly what "unit-testable
services" means in practice.

Run with:
    pytest tests/test_document_upload_service.py -v
"""
import uuid
from dataclasses import dataclass

import pytest

from app.models.enums import ProcessingJobType
from app.services.document_upload_service import ALLOWED_EXTENSIONS, DocumentUploadService
from app.services.exceptions import ValidationError
from app.storage.base import SavedFile


@dataclass
class _FakeDocument:
    id: uuid.UUID


@dataclass
class _FakeVersion:
    id: uuid.UUID


@dataclass
class _FakeJob:
    id: uuid.UUID


class FakeFileStorage:
    """Records what it was asked to save, without touching disk."""

    def __init__(self) -> None:
        self.saved_calls: list[dict] = []

    async def save(self, *, content: bytes, original_filename: str) -> SavedFile:
        self.saved_calls.append({"content": content, "original_filename": original_filename})
        return SavedFile(storage_key=f"fake-{original_filename}", size_bytes=len(content), checksum_sha256="deadbeef")


class FakeDocumentService:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.created_ids: list[uuid.UUID] = []
        self.deleted: list[uuid.UUID] = []
        self.updated: list[dict] = []
        self.fail_create = False

    async def create_document(self, **kwargs):
        if self.fail_create:
            raise RuntimeError("simulated document creation failure")
        self.created.append(kwargs)
        doc = _FakeDocument(id=uuid.uuid4())
        self.created_ids.append(doc.id)
        return doc

    async def update(self, id_: uuid.UUID, **kwargs) -> None:
        self.updated.append({"id": id_, **kwargs})

    async def delete(self, id_: uuid.UUID) -> None:
        self.deleted.append(id_)


class FakeDocumentVersionService:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.created_ids: list[uuid.UUID] = []
        self.deleted: list[uuid.UUID] = []
        self.updated: list[dict] = []
        self.fail_create = False

    async def create_next_version(self, **kwargs):
        if self.fail_create:
            raise RuntimeError("simulated version creation failure")
        self.created.append(kwargs)
        version = _FakeVersion(id=uuid.uuid4())
        self.created_ids.append(version.id)
        return version

    async def update(self, id_: uuid.UUID, **kwargs) -> None:
        self.updated.append({"id": id_, **kwargs})

    async def delete(self, id_: uuid.UUID) -> None:
        self.deleted.append(id_)


class FakeProcessingJobService:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.fail_create = False

    async def create_job(self, **kwargs):
        if self.fail_create:
            raise RuntimeError("simulated job creation failure")
        self.created.append(kwargs)
        return _FakeJob(id=uuid.uuid4())


def make_service(
    *, documents=None, versions=None, jobs=None, storage=None
) -> tuple[DocumentUploadService, FakeDocumentService, FakeDocumentVersionService, FakeProcessingJobService, FakeFileStorage]:
    documents = documents or FakeDocumentService()
    versions = versions or FakeDocumentVersionService()
    jobs = jobs or FakeProcessingJobService()
    storage = storage or FakeFileStorage()
    service = DocumentUploadService(
        session=None,  # never touched: every composed dependency is faked
        storage=storage,
        document_service=documents,
        version_service=versions,
        job_service=jobs,
    )
    return service, documents, versions, jobs, storage


class TestValidateFile:
    """Pure validation logic — no async, no I/O."""

    def test_rejects_empty_file(self) -> None:
        service, *_ = make_service()
        with pytest.raises(ValidationError, match="empty"):
            service.validate_file(filename="a.txt", content_type="text/plain", size_bytes=0, max_size_bytes=1000)

    def test_rejects_oversized_file(self) -> None:
        service, *_ = make_service()
        with pytest.raises(ValidationError, match="exceeding"):
            service.validate_file(filename="a.txt", content_type="text/plain", size_bytes=2000, max_size_bytes=1000)

    def test_rejects_unsupported_extension(self) -> None:
        service, *_ = make_service()
        with pytest.raises(ValidationError, match="Unsupported file extension"):
            service.validate_file(
                filename="virus.exe", content_type="application/octet-stream", size_bytes=10, max_size_bytes=1000
            )

    def test_rejects_mismatched_content_type(self) -> None:
        service, *_ = make_service()
        with pytest.raises(ValidationError, match="does not match"):
            service.validate_file(
                filename="report.pdf", content_type="image/png", size_bytes=10, max_size_bytes=1000
            )

    @pytest.mark.parametrize("extension", sorted(ALLOWED_EXTENSIONS))
    def test_accepts_every_supported_extension_with_matching_type(self, extension: str) -> None:
        service, *_ = make_service()
        content_type = next(iter(ALLOWED_EXTENSIONS[extension]))
        service.validate_file(
            filename=f"file{extension}", content_type=content_type, size_bytes=10, max_size_bytes=1000
        )  # does not raise

    def test_accepts_generic_octet_stream_content_type(self) -> None:
        """A .md upload with a generic Content-Type (common in practice —
        many clients don't know 'text/markdown')."""
        service, *_ = make_service()
        service.validate_file(
            filename="notes.md", content_type="application/octet-stream", size_bytes=10, max_size_bytes=1000
        )  # does not raise


class TestUploadWorkflow:
    @pytest.mark.asyncio
    async def test_successful_upload_creates_document_version_and_job(self) -> None:
        service, documents, versions, jobs, storage = make_service()
        user_id = uuid.uuid4()

        result = await service.upload(
            user_id=user_id,
            filename="notes.txt",
            content_type="text/plain",
            content=b"hello world",
            max_size_bytes=1000,
            title="My Notes",
        )

        assert result.document_id is not None
        assert result.version_id is not None
        assert result.processing_job_id is not None
        assert result.original_filename == "notes.txt"
        assert result.file_size_bytes == len(b"hello world")
        assert result.checksum_sha256 == "deadbeef"

        assert len(storage.saved_calls) == 1
        assert documents.created[0]["user_id"] == user_id
        assert documents.created[0]["title"] == "My Notes"
        assert versions.created[0]["document_id"] == result.document_id
        assert versions.created[0]["uploaded_by"] == user_id
        assert jobs.created[0]["document_version_id"] == result.version_id
        assert jobs.created[0]["job_type"] == ProcessingJobType.PARSE

        # No compensation should have run on the happy path.
        assert documents.deleted == []
        assert versions.deleted == []

    @pytest.mark.asyncio
    async def test_title_defaults_to_filename_when_omitted(self) -> None:
        service, documents, *_ = make_service()
        await service.upload(
            user_id=uuid.uuid4(), filename="report.pdf", content_type="application/pdf",
            content=b"%PDF-1.4", max_size_bytes=1000, title=None,
        )
        assert documents.created[0]["title"] == "report.pdf"

    @pytest.mark.asyncio
    async def test_invalid_file_is_rejected_before_any_write(self) -> None:
        service, documents, versions, jobs, storage = make_service()

        with pytest.raises(ValidationError):
            await service.upload(
                user_id=uuid.uuid4(), filename="virus.exe", content_type="application/octet-stream",
                content=b"MZ", max_size_bytes=1000,
            )

        assert storage.saved_calls == []
        assert documents.created == []
        assert versions.created == []
        assert jobs.created == []

    @pytest.mark.asyncio
    async def test_version_failure_compensates_by_deleting_document(self) -> None:
        versions = FakeDocumentVersionService()
        versions.fail_create = True
        service, documents, versions, jobs, storage = make_service(versions=versions)

        with pytest.raises(RuntimeError, match="simulated version creation failure"):
            await service.upload(
                user_id=uuid.uuid4(), filename="notes.txt", content_type="text/plain",
                content=b"hello", max_size_bytes=1000,
            )

        assert len(documents.created) == 1  # the document really was created...
        assert len(documents.deleted) == 1  # ...then compensated away
        assert documents.deleted[0] == documents.created_ids[0]  # the *same* document was compensated
        assert jobs.created == []

    @pytest.mark.asyncio
    async def test_job_failure_compensates_by_deleting_version_and_document(self) -> None:
        jobs = FakeProcessingJobService()
        jobs.fail_create = True
        service, documents, versions, jobs, storage = make_service(jobs=jobs)

        with pytest.raises(RuntimeError, match="simulated job creation failure"):
            await service.upload(
                user_id=uuid.uuid4(), filename="notes.txt", content_type="text/plain",
                content=b"hello", max_size_bytes=1000,
            )

        assert len(documents.created) == 1
        assert len(versions.created) == 1
        assert len(versions.deleted) == 1  # version compensated
        assert versions.deleted[0] == versions.created_ids[0]
        assert len(documents.deleted) == 1  # document compensated
        assert documents.deleted[0] == documents.created_ids[0]
