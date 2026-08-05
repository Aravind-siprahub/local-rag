"""Document version endpoints."""
import uuid

from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import PaginationParams, get_document_version_service
from app.schemas.actions import DocumentVersionUploadRequest
from app.schemas.document_version import (
    DocumentVersionListResponse,
    DocumentVersionResponse,
    DocumentVersionUpdate,
)
from app.services.document_version_service import DocumentVersionService

router = APIRouter(prefix="/document-versions", tags=["Document Versions"])


@router.post(
    "",
    response_model=DocumentVersionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a new version of a document",
    description="version_number is computed automatically (existing max + 1); it is not part of the request body.",
)
async def create_document_version(
    payload: DocumentVersionUploadRequest,
    service: DocumentVersionService = Depends(get_document_version_service),
) -> DocumentVersionResponse:
    version = await service.create_next_version(**payload.model_dump())
    return DocumentVersionResponse.model_validate(version)


@router.get("", response_model=DocumentVersionListResponse, summary="List versions of a document")
async def list_document_versions(
    document_id: uuid.UUID = Query(..., description="Parent document id."),
    pagination: PaginationParams = Depends(),
    service: DocumentVersionService = Depends(get_document_version_service),
) -> DocumentVersionListResponse:
    versions = await service.list_by_document(document_id, limit=pagination.limit, offset=pagination.offset)
    return DocumentVersionListResponse(
        items=[DocumentVersionResponse.model_validate(v) for v in versions],
        total=len(versions),  # filtered count unavailable without modifying the repository layer
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/{version_id}", response_model=DocumentVersionResponse, summary="Get a document version by id")
async def get_document_version(
    version_id: uuid.UUID, service: DocumentVersionService = Depends(get_document_version_service)
) -> DocumentVersionResponse:
    version = await service.get(version_id)
    return DocumentVersionResponse.model_validate(version)


@router.patch(
    "/{version_id}",
    response_model=DocumentVersionResponse,
    summary="Update a document version's pipeline status",
)
async def update_document_version(
    version_id: uuid.UUID,
    payload: DocumentVersionUpdate,
    service: DocumentVersionService = Depends(get_document_version_service),
) -> DocumentVersionResponse:
    updates = payload.model_dump(exclude_unset=True)
    version = await service.update(version_id, **updates)
    return DocumentVersionResponse.model_validate(version)


@router.delete("/{version_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a document version")
async def delete_document_version(
    version_id: uuid.UUID, service: DocumentVersionService = Depends(get_document_version_service)
) -> None:
    await service.delete(version_id)
