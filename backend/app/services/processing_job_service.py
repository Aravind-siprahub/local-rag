"""Business logic for `app.models.processing_job.ProcessingJob`.

The bulk of this service is a state machine (pending -> running ->
completed/failed) — validating *transitions*, not column values, is exactly
the kind of business rule a service layer (rather than a DB CHECK
constraint or a Pydantic validator) is responsible for.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import ProcessingJobStatus, ProcessingJobType
from app.models.processing_job import ProcessingJob
from app.repositories.document_version_repository import DocumentVersionRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.services.base_service import BaseService
from app.services.exceptions import NotFoundError, ValidationError


class ProcessingJobService(BaseService[ProcessingJob, uuid.UUID, ProcessingJobRepository]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, ProcessingJobRepository(session))
        self._versions = DocumentVersionRepository(session)

    async def create_job(
        self, *, document_version_id: uuid.UUID, job_type: ProcessingJobType
    ) -> ProcessingJob:
        version = await self._versions.get(document_version_id)
        if version is None:
            raise NotFoundError(f"DocumentVersion with id={document_version_id!r} was not found.")

        return await self.create(document_version_id=document_version_id, job_type=job_type)

    async def start(self, job_id: uuid.UUID) -> ProcessingJob:
        """pending -> running."""
        job = await self.get(job_id)
        if job.status != ProcessingJobStatus.PENDING:
            raise ValidationError(f"Cannot start a job in status {job.status.value!r}; must be 'pending'.")

        return await self.update(job_id, status=ProcessingJobStatus.RUNNING, started_at=datetime.now(timezone.utc))

    async def complete(self, job_id: uuid.UUID) -> ProcessingJob:
        """running -> completed."""
        job = await self.get(job_id)
        if job.status != ProcessingJobStatus.RUNNING:
            raise ValidationError(f"Cannot complete a job in status {job.status.value!r}; must be 'running'.")

        return await self.update(
            job_id, status=ProcessingJobStatus.COMPLETED, completed_at=datetime.now(timezone.utc)
        )

    async def fail(self, job_id: uuid.UUID, error_message: str) -> ProcessingJob:
        """pending or running -> failed, incrementing the retry counter."""
        job = await self.get(job_id)
        if job.status not in (ProcessingJobStatus.PENDING, ProcessingJobStatus.RUNNING):
            raise ValidationError(
                f"Cannot fail a job in status {job.status.value!r}; must be 'pending' or 'running'."
            )

        return await self.update(
            job_id,
            status=ProcessingJobStatus.FAILED,
            error_message=error_message,
            retry_count=job.retry_count + 1,
            completed_at=datetime.now(timezone.utc),
        )

    async def list_by_document_version(self, document_version_id: uuid.UUID) -> list[ProcessingJob]:
        return await self.repository.list_by_document_version(document_version_id)

    async def list_active(self, *, limit: int = 100, offset: int = 0) -> list[ProcessingJob]:
        return await self.repository.list_active(limit=limit, offset=offset)
