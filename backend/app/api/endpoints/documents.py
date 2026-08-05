"""Document endpoints."""
import uuid

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.api.dependencies import PaginationParams, get_document_service, get_document_upload_service
from app.api.file_utils import read_upload_within_limit
from app.core.config import get_settings
from app.schemas.actions import SetCurrentVersionRequest
from app.schemas.document import DocumentCreate, DocumentListResponse, DocumentResponse, DocumentUpdate
from app.schemas.upload import DocumentUploadResponse
from app.services.document_service import DocumentService
from app.services.document_upload_service import DocumentUploadService

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED, summary="Create a document")
async def create_document(
    payload: DocumentCreate, service: DocumentService = Depends(get_document_service)
) -> DocumentResponse:
    document = await service.create_document(
        user_id=payload.user_id, title=payload.title, description=payload.description, tags=payload.tags
    )
    return DocumentResponse.model_validate(document)


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document file (PDF, DOCX, TXT, or Markdown)",
    description=(
        "Saves the file to local storage, creates a Document and its first "
        "DocumentVersion, and queues a pending ProcessingJob (job_type='parse') "
        "for a future worker to pick up. Does NOT parse, chunk, or embed the "
        "file — upload is a separate concern from processing."
    ),
)
async def upload_document(
    user_id: uuid.UUID = Form(
        ...,
        description="Owner user id. Call GET /users to list existing users, "
        "or POST /users to create one first.",
        examples=["a2a5b0fa-85e9-4e4f-8ad1-6c87e57edebc"],
    ),
    title: str | None = Form(default=None, description="Defaults to the uploaded filename if omitted."),
    file: UploadFile = File(..., description="PDF, DOCX, TXT, or Markdown (.md/.markdown) file."),
    service: DocumentUploadService = Depends(get_document_upload_service),
) -> DocumentUploadResponse:
    settings = get_settings()
    content = await read_upload_within_limit(file, settings.max_upload_size_bytes)

    result = await service.upload(
        user_id=user_id,
        filename=file.filename or "upload",
        content_type=file.content_type,
        content=content,
        max_size_bytes=settings.max_upload_size_bytes,
        title=title,
    )
    return DocumentUploadResponse(
        document_id=result.document_id,
        version_id=result.version_id,
        processing_job_id=result.processing_job_id,
        original_filename=result.original_filename,
        mime_type=result.mime_type,
        file_size_bytes=result.file_size_bytes,
        checksum_sha256=result.checksum_sha256,
        storage_key=result.storage_key,
    )


@router.get("", response_model=DocumentListResponse, summary="List documents, optionally filtered by owner")
async def list_documents(
    user_id: uuid.UUID | None = None,
    pagination: PaginationParams = Depends(),
    service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    if user_id is not None:
        documents = await service.list_by_user(user_id, limit=pagination.limit, offset=pagination.offset)
        # No filtered-count method exists on the (frozen) repository layer,
        # so `total` here reflects this page only, not the true filtered
        # total — see this router's module docs / the project's summary.
        total = len(documents)
    else:
        documents = await service.list(limit=pagination.limit, offset=pagination.offset)
        total = await service.count()

    return DocumentListResponse(
        items=[DocumentResponse.model_validate(d) for d in documents],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/{document_id}", response_model=DocumentResponse, summary="Get a document by id")
async def get_document(
    document_id: uuid.UUID, service: DocumentService = Depends(get_document_service)
) -> DocumentResponse:
    document = await service.get(document_id)
    return DocumentResponse.model_validate(document)


@router.patch("/{document_id}", response_model=DocumentResponse, summary="Update a document")
async def update_document(
    document_id: uuid.UUID, payload: DocumentUpdate, service: DocumentService = Depends(get_document_service)
) -> DocumentResponse:
    updates = payload.model_dump(exclude_unset=True)
    document = await service.update(document_id, **updates)
    return DocumentResponse.model_validate(document)


@router.post(
    "/{document_id}/current-version",
    response_model=DocumentResponse,
    summary="Promote a version to be this document's current version",
)
async def set_current_version(
    document_id: uuid.UUID,
    payload: SetCurrentVersionRequest,
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    document = await service.set_current_version(document_id, payload.version_id)
    return DocumentResponse.model_validate(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Soft-delete a document",
    description="Sets deleted_at; existing versions/chunks/embeddings are left intact.",
)
async def delete_document(
    document_id: uuid.UUID, service: DocumentService = Depends(get_document_service)
) -> None:
    await service.delete(document_id)
