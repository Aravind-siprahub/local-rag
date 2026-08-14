"""Document endpoints."""
import logging
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import PaginationParams, get_current_user, get_db, get_document_service, get_document_upload_service
from app.api.file_utils import read_upload_within_limit
from app.api.security import verify_ownership
from app.core.config import get_settings
from app.db.session import AsyncSessionLocal
from app.models.document_chunk import DocumentChunk
from app.models.embedding import Embedding
from app.models.enums import DocumentStatus, ProcessingJobType
from app.models.user import User
from app.schemas.actions import SetCurrentVersionRequest
from app.schemas.document import DocumentCreate, DocumentListResponse, DocumentResponse, DocumentUpdate
from app.schemas.upload import DocumentUploadResponse
from app.services.document_service import DocumentService
from app.services.document_upload_service import DocumentUploadService
from app.services.document_version_service import DocumentVersionService
from app.services.ingestion_service import IngestionService
from app.services.processing_job_service import ProcessingJobService
from app.processing.processor import DocumentProcessor
from app.embeddings.worker import EmbeddingWorker
from app.storage.s3_storage_service import S3StorageService
from app.storage.supabase_storage_service import SupabaseStorageService
from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Document management API endpoints
router = APIRouter(prefix="/documents", tags=["Documents"])


async def _run_ingestion_background(document_id: uuid.UUID, parse_job_id: uuid.UUID | None = None) -> None:
    """Async wrapper for background execution of document ingestion."""
    from app.processing.background_runner import BackgroundJobRunner
    BackgroundJobRunner.enqueue_document(document_id)



@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED, summary="Create a document")
async def create_document(
    payload: DocumentCreate,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    document = await service.create_document(
        user_id=current_user.id, title=payload.title, description=payload.description, tags=payload.tags
    )
    return DocumentResponse.model_validate(document)


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document file (PDF, DOCX, TXT, or Markdown)",
    description=(
        "Saves the file to local storage, creates a Document and DocumentVersion, "
        "and automatically runs the full 6-stage ingestion pipeline."
    ),
)
async def upload_document(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    title: str | None = Form(default=None, description="Defaults to the uploaded filename if omitted."),
    file: UploadFile = File(..., description="PDF, DOCX, TXT, or Markdown (.md/.markdown) file."),
    service: DocumentUploadService = Depends(get_document_upload_service),
) -> DocumentUploadResponse:
    settings = get_settings()
    content = await read_upload_within_limit(file, settings.max_upload_size_bytes)

    result = await service.upload(
        user_id=current_user.id,
        filename=file.filename or "upload",
        content_type=file.content_type,
        content=content,
        max_size_bytes=settings.max_upload_size_bytes,
        title=title,
    )

    # Queue immediate background ingestion pass: parse -> chunk -> embed -> index
    background_tasks.add_task(_run_ingestion_background, result.document_id, result.processing_job_id)

    return DocumentUploadResponse(
        id=result.document_id,
        filename=result.original_filename,
        bucket=result.bucket_name,
        storagePath=result.storage_path,
        status="Pending",
        document_id=result.document_id,
        version_id=result.version_id,
        processing_job_id=result.processing_job_id,
        original_filename=result.original_filename,
        mime_type=result.mime_type,
        file_size_bytes=result.file_size_bytes,
        checksum_sha256=result.checksum_sha256,
        storage_key=result.storage_key,
    )


@router.post(
    "/{document_id}/process",
    summary="Trigger/retry full ingestion pipeline for a document",
)
async def process_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Manually trigger or retry full ingestion pipeline for a document."""
    service = DocumentService(session)
    doc = await service.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")
    verify_ownership(doc.user_id, current_user, "document")

    try:
        ingestion = IngestionService(session)
        result = await ingestion.run_pipeline(document_id)
        await session.commit()
        return {
            "message": f"Ingestion completed for document {document_id}.",
            "document_id": str(document_id),
            "status": "ready",
            "character_count": result.character_count,
            "chunk_count": result.chunk_count,
            "embedding_count": result.embedding_count,
            "vector_count": result.vector_count,
        }
    except Exception as exc:
        logger.exception("Manual ingestion process failed for document %s", document_id)
        try:
            await session.commit()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Ingestion failed for {document_id}: {exc}")


@router.get(
    "/{document_id}/debug",
    summary="Debug processing pipeline for a document",
)
async def debug_document_processing(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Diagnostic endpoint to inspect text extraction, chunking, and embedding counts."""
    try:
        service = DocumentService(session)
        doc = await service.get(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")
        verify_ownership(doc.user_id, current_user, "document")

        version_service = DocumentVersionService(session)
        version = None
        if hasattr(version_service, "get_current_version"):
            version = await version_service.get_current_version(document_id)
        if not version:
            versions = await version_service.list_by_document(document_id)
            version = versions[-1] if versions else None

        if not version:
            doc_status_str = getattr(doc.status, "value", str(doc.status))
            return {
                "status": doc_status_str,
                "textExtracted": False,
                "characters": 0,
                "chunks": 0,
                "embeddings": 0,
                "vectorsIndexed": 0,
                "lastError": "No DocumentVersion associated with this document.",
            }

        # Count chunks
        stmt_chunks = select(DocumentChunk).where(DocumentChunk.document_version_id == version.id)
        res_chunks = list((await session.execute(stmt_chunks)).scalars().all())
        chunk_count = len(res_chunks)

        # Calculate total characters extracted
        total_chars = sum(len(c.content) for c in res_chunks)

        # Count embeddings & indexed vectors
        chunk_ids = [c.id for c in res_chunks]
        vector_count = 0
        if chunk_ids:
            stmt_vectors = select(func.count(Embedding.id)).where(Embedding.chunk_id.in_(chunk_ids))
            vector_count = (await session.execute(stmt_vectors)).scalar_one()

        current_stage = "Completed" if doc.status == DocumentStatus.READY else ("Failed" if doc.status == DocumentStatus.FAILED else "Processing")
        return {
            "id": str(document_id),
            "status": getattr(doc.status, "value", str(doc.status)),
            "textExtracted": total_chars > 0,
            "characters": total_chars,
            "chunks": chunk_count,
            "embeddings": vector_count,
            "vectors": vector_count,
            "lastError": getattr(version, "error_message", None),
            "currentStage": current_stage,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in debug_document_processing for document %s", document_id)
        return {
            "id": str(document_id),
            "status": "error",
            "error": str(exc),
            "textExtracted": False,
            "characters": 0,
            "chunks": 0,
            "embeddings": 0,
            "vectors": 0,
            "lastError": str(exc),
            "currentStage": "Failed",
        }


@router.get("", response_model=DocumentListResponse, summary="List documents, optionally filtered by owner")
async def list_documents(
    background_tasks: BackgroundTasks,
    user_id: uuid.UUID | None = None,
    pagination: PaginationParams = Depends(),
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    if user_id is not None and str(user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Cannot access documents belonging to another user.",
        )

    target_user_id = current_user.id
    documents = await service.list_by_user(target_user_id, limit=pagination.limit, offset=pagination.offset)
    total = len(documents)

    # Auto-heal: trigger ingestion for any documents stuck in UPLOADED or PROCESSING state
    for doc in documents:
        status_val = getattr(doc.status, "value", str(doc.status))
        if status_val in ("uploaded", "processing"):
            background_tasks.add_task(_run_ingestion_background, doc.id)

    return DocumentListResponse(
        items=[DocumentResponse.model_validate(d) for d in documents],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/{document_id}", response_model=DocumentResponse, summary="Get a document by id")
async def get_document(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    document = await service.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")
    verify_ownership(document.user_id, current_user, "document")

    # Auto-heal: If document is still in UPLOADED status, trigger background ingestion
    if document.status == DocumentStatus.UPLOADED:
        background_tasks.add_task(_run_ingestion_background, document_id)

    return DocumentResponse.model_validate(document)


@router.patch("/{document_id}", response_model=DocumentResponse, summary="Update a document")
async def update_document(
    document_id: uuid.UUID,
    payload: DocumentUpdate,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    document = await service.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")
    verify_ownership(document.user_id, current_user, "document")

    updates = payload.model_dump(exclude_unset=True)
    updated_document = await service.update(document_id, **updates)
    return DocumentResponse.model_validate(updated_document)


@router.post(
    "/{document_id}/current-version",
    response_model=DocumentResponse,
    summary="Promote a version to be this document's current version",
)
async def set_current_version(
    document_id: uuid.UUID,
    payload: SetCurrentVersionRequest,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    document = await service.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")
    verify_ownership(document.user_id, current_user, "document")

    updated_document = await service.set_current_version(document_id, payload.version_id)
    return DocumentResponse.model_validate(updated_document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and clean up Supabase Storage & vectors",
    description="Deletes vector embeddings, chunks, metadata, and remote Supabase Storage files.",
)
async def delete_document(
    document_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
) -> None:
    service = DocumentService(session)
    document = await service.get(document_id)
    if not document:
        return
    verify_ownership(document.user_id, current_user, "document")

    # Delete remote objects from storage (S3 > REST > skip)
    version_service = DocumentVersionService(session)
    versions = await version_service.list_by_document(document_id, limit=100)
    _settings = get_settings()
    storage_service = S3StorageService() if _settings.s3_is_configured else SupabaseStorageService()

    for version in versions:
        storage_path = getattr(version, "storage_path", None) or f"{document.user_id}/{document_id}/{version.original_filename}"
        if storage_service.is_configured:
            try:
                await storage_service.delete_file(storage_path=storage_path)
            except Exception as exc:
                logger.warning("Failed to delete remote file %s from storage: %s", storage_path, exc)

    await service.delete(document_id)
