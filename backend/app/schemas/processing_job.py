"""Schemas for `app.models.processing_job.ProcessingJob`."""
import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field

from app.models.enums import ProcessingJobStatus, ProcessingJobType
from app.schemas.common import ORMModel, PaginatedResponse, TimestampSchema


class ProcessingJobBase(BaseModel):
    job_type: ProcessingJobType


class ProcessingJobCreate(ProcessingJobBase):
    document_version_id: uuid.UUID


class ProcessingJobUpdate(BaseModel):
    """Covers the fields a worker sets as a job progresses:
    pending -> running -> completed/failed.
    """

    status: ProcessingJobStatus | None = None
    error_message: str | None = None
    retry_count: Annotated[int, Field(ge=0)] | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ProcessingJobResponse(ProcessingJobBase, TimestampSchema, ORMModel):
    id: uuid.UUID
    document_version_id: uuid.UUID
    status: ProcessingJobStatus
    error_message: str | None = None
    retry_count: Annotated[int, Field(ge=0)]
    started_at: datetime | None = None
    completed_at: datetime | None = None


ProcessingJobListResponse = PaginatedResponse[ProcessingJobResponse]
