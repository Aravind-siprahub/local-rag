"""Document version endpoints."""
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import (
    PaginationParams,
    get_current_user,
    get_document_service,
    get_document_version_service,
    require_document_upload_permission,
)
from app.api.security import verify_ownership
from app.models.user import User
from app.schemas.actions import DocumentVersionUploadRequest
from app.schemas.document_version import (
    DocumentVersionListResponse,
    DocumentVersionResponse,
    DocumentVersionUpdate,
)
from app.services.document_service import DocumentService
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
    current_user: User = Depends(require_document_upload_permission),
    doc_service: DocumentService = Depends(get_document_service),
    service: DocumentVersionService = Depends(get_document_version_service),
) -> DocumentVersionResponse:
    doc = await doc_service.get(payload.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {payload.document_id} not found.")
    verify_ownership(doc.user_id, current_user, "document")

    version = await service.create_next_version(**payload.model_dump())
    return DocumentVersionResponse.model_validate(version)


@router.get("", response_model=DocumentVersionListResponse, summary="List versions of a document")
async def list_document_versions(
    document_id: uuid.UUID = Query(..., description="Parent document id."),
    pagination: PaginationParams = Depends(),
    current_user: User = Depends(get_current_user),
    doc_service: DocumentService = Depends(get_document_service),
    service: DocumentVersionService = Depends(get_document_version_service),
) -> DocumentVersionListResponse:
    doc = await doc_service.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")
    verify_ownership(doc.user_id, current_user, "document")

    versions = await service.list_by_document(document_id, limit=pagination.limit, offset=pagination.offset)
    return DocumentVersionListResponse(
        items=[DocumentVersionResponse.model_validate(v) for v in versions],
        total=len(versions),
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/{version_id}", response_model=DocumentVersionResponse, summary="Get a document version by id")
async def get_document_version(
    version_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    doc_service: DocumentService = Depends(get_document_service),
    service: DocumentVersionService = Depends(get_document_version_service),
) -> DocumentVersionResponse:
    version = await service.get(version_id)
    if not version:
        raise HTTPException(status_code=404, detail=f"Document version {version_id} not found.")

    doc = await doc_service.get(version.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Parent document not found.")
    verify_ownership(doc.user_id, current_user, "document version")

    return DocumentVersionResponse.model_validate(version)


@router.patch(
    "/{version_id}",
    response_model=DocumentVersionResponse,
    summary="Update a document version's pipeline status",
)
async def update_document_version(
    version_id: uuid.UUID,
    payload: DocumentVersionUpdate,
    current_user: User = Depends(get_current_user),
    doc_service: DocumentService = Depends(get_document_service),
    service: DocumentVersionService = Depends(get_document_version_service),
) -> DocumentVersionResponse:
    version = await service.get(version_id)
    if not version:
        raise HTTPException(status_code=404, detail=f"Document version {version_id} not found.")

    doc = await doc_service.get(version.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Parent document not found.")
    verify_ownership(doc.user_id, current_user, "document version")

    updates = payload.model_dump(exclude_unset=True)
    updated_version = await service.update(version_id, **updates)
    return DocumentVersionResponse.model_validate(updated_version)


@router.delete("/{version_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a document version")
async def delete_document_version(
    version_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    doc_service: DocumentService = Depends(get_document_service),
    service: DocumentVersionService = Depends(get_document_version_service),
) -> None:
    version = await service.get(version_id)
    if not version:
        return

    doc = await doc_service.get(version.document_id)
    if doc:
        verify_ownership(doc.user_id, current_user, "document version")

    await service.delete(version_id)
