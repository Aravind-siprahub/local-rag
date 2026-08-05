"""Data access for `app.models.processing_job.ProcessingJob`."""
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import ProcessingJobStatus
from app.models.processing_job import ProcessingJob
from app.repositories.base_repository import BaseRepository


class ProcessingJobRepository(BaseRepository[ProcessingJob, uuid.UUID]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, ProcessingJob)

    async def list_by_document_version(self, document_version_id: uuid.UUID) -> list[ProcessingJob]:
        stmt = (
            select(ProcessingJob)
            .where(ProcessingJob.document_version_id == document_version_id)
            .order_by(ProcessingJob.created_at)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_active(self, *, limit: int = 100, offset: int = 0) -> list[ProcessingJob]:
        """Jobs still pending or running — matches the partial index
        `processing_jobs_active_status_idx`, the worker's polling query.
        """
        stmt = (
            select(ProcessingJob)
            .where(ProcessingJob.status.in_([ProcessingJobStatus.PENDING, ProcessingJobStatus.RUNNING]))
            .order_by(ProcessingJob.created_at)
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
