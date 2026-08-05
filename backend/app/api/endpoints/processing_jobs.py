"""Processing job endpoints."""
import uuid

from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import PaginationParams, get_processing_job_service
from app.schemas.actions import FailJobRequest
from app.schemas.processing_job import ProcessingJobCreate, ProcessingJobListResponse, ProcessingJobResponse
from app.services.processing_job_service import ProcessingJobService

router = APIRouter(prefix="/processing-jobs", tags=["Processing Jobs"])


@router.post(
    "", response_model=ProcessingJobResponse, status_code=status.HTTP_201_CREATED, summary="Create a processing job"
)
async def create_processing_job(
    payload: ProcessingJobCreate, service: ProcessingJobService = Depends(get_processing_job_service)
) -> ProcessingJobResponse:
    job = await service.create_job(document_version_id=payload.document_version_id, job_type=payload.job_type)
    return ProcessingJobResponse.model_validate(job)


@router.get(
    "",
    response_model=ProcessingJobListResponse,
    summary="List processing jobs, optionally by document version or active-only",
)
async def list_processing_jobs(
    document_version_id: uuid.UUID | None = Query(default=None),
    active_only: bool = Query(default=False, description="Only jobs still pending or running."),
    pagination: PaginationParams = Depends(),
    service: ProcessingJobService = Depends(get_processing_job_service),
) -> ProcessingJobListResponse:
    if document_version_id is not None:
        jobs = await service.list_by_document_version(document_version_id)
        total = len(jobs)
    elif active_only:
        jobs = await service.list_active(limit=pagination.limit, offset=pagination.offset)
        total = len(jobs)
    else:
        jobs = await service.list(limit=pagination.limit, offset=pagination.offset)
        total = await service.count()

    return ProcessingJobListResponse(
        items=[ProcessingJobResponse.model_validate(j) for j in jobs],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/{job_id}", response_model=ProcessingJobResponse, summary="Get a processing job by id")
async def get_processing_job(
    job_id: uuid.UUID, service: ProcessingJobService = Depends(get_processing_job_service)
) -> ProcessingJobResponse:
    job = await service.get(job_id)
    return ProcessingJobResponse.model_validate(job)


@router.post(
    "/{job_id}/start", response_model=ProcessingJobResponse, summary="Transition a job from pending to running"
)
async def start_processing_job(
    job_id: uuid.UUID, service: ProcessingJobService = Depends(get_processing_job_service)
) -> ProcessingJobResponse:
    job = await service.start(job_id)
    return ProcessingJobResponse.model_validate(job)


@router.post(
    "/{job_id}/complete",
    response_model=ProcessingJobResponse,
    summary="Transition a job from running to completed",
)
async def complete_processing_job(
    job_id: uuid.UUID, service: ProcessingJobService = Depends(get_processing_job_service)
) -> ProcessingJobResponse:
    job = await service.complete(job_id)
    return ProcessingJobResponse.model_validate(job)


@router.post(
    "/{job_id}/fail",
    response_model=ProcessingJobResponse,
    summary="Transition a job to failed, incrementing its retry counter",
)
async def fail_processing_job(
    job_id: uuid.UUID,
    payload: FailJobRequest,
    service: ProcessingJobService = Depends(get_processing_job_service),
) -> ProcessingJobResponse:
    job = await service.fail(job_id, payload.error_message)
    return ProcessingJobResponse.model_validate(job)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a processing job")
async def delete_processing_job(
    job_id: uuid.UUID, service: ProcessingJobService = Depends(get_processing_job_service)
) -> None:
    await service.delete(job_id)
