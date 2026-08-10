"""FastAPI dependency providers: the DB session and one factory per service.

Endpoints depend on `get_*_service`, never construct a service directly —
this is what makes `app.dependency_overrides[get_user_service] = ...` work
for testing without a real database, and keeps the DI wiring in one place.
"""
from fastapi import Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.services.chat_message_service import ChatMessageService
from app.services.chat_session_service import ChatSessionService
from app.services.document_service import DocumentService
from app.services.document_upload_service import DocumentUploadService
from app.services.document_version_service import DocumentVersionService
from app.services.processing_job_service import ProcessingJobService
from app.services.system_setting_service import SystemSettingService
from app.services.user_service import UserService
from app.rag.service import RAGService
from app.storage.local_file_storage import LocalFileStorage


def get_user_service(session: AsyncSession = Depends(get_db)) -> UserService:
    return UserService(session)


def get_document_service(session: AsyncSession = Depends(get_db)) -> DocumentService:
    return DocumentService(session)


from app.storage.s3_storage_service import S3StorageService
from app.storage.supabase_storage_service import SupabaseStorageService

def _get_storage_backend():
    """Return the best available storage backend.

    Priority: S3 (boto3) > Supabase REST > local disk
    S3 is preferred because it is more reliable than the custom REST client.
    """
    from app.core.config import get_settings
    settings = get_settings()
    if settings.s3_is_configured:
        return S3StorageService()
    supabase = SupabaseStorageService()
    if supabase.is_configured:
        return supabase
    return LocalFileStorage()

def get_document_upload_service(session: AsyncSession = Depends(get_db)) -> DocumentUploadService:
    return DocumentUploadService(session, _get_storage_backend())


def get_document_version_service(session: AsyncSession = Depends(get_db)) -> DocumentVersionService:
    return DocumentVersionService(session)


def get_chat_session_service(session: AsyncSession = Depends(get_db)) -> ChatSessionService:
    return ChatSessionService(session)


def get_chat_message_service(session: AsyncSession = Depends(get_db)) -> ChatMessageService:
    return ChatMessageService(session)


def get_processing_job_service(session: AsyncSession = Depends(get_db)) -> ProcessingJobService:
    return ProcessingJobService(session)


def get_system_setting_service(session: AsyncSession = Depends(get_db)) -> SystemSettingService:
    return SystemSettingService(session)


def get_rag_service(session: AsyncSession = Depends(get_db)) -> RAGService:
    return RAGService(session)


class PaginationParams:
    """Reusable `limit`/`offset` query params — `Depends(PaginationParams)`
    (or bare `Depends()` on a `PaginationParams`-typed parameter) instead of
    redeclaring the same two `Query(...)` args in every list endpoint.
    """

    def __init__(
        self,
        limit: int = Query(default=100, ge=1, le=500, description="Max rows to return."),
        offset: int = Query(default=0, ge=0, description="Rows to skip."),
    ) -> None:
        self.limit = limit
        self.offset = offset
