"""FastAPI dependency providers: the DB session and one factory per service.

Endpoints depend on `get_*_service`, never construct a service directly —
this is what makes `app.dependency_overrides[get_user_service] = ...` work
for testing without a real database, and keeps the DI wiring in one place.
"""
from fastapi import Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.user import User
from app.rag.service import RAGService
from app.repositories.user_repository import UserRepository
from app.services.chat_message_service import ChatMessageService
from app.services.chat_session_service import ChatSessionService
from app.services.document_service import DocumentService
from app.services.document_upload_service import DocumentUploadService
from app.services.document_version_service import DocumentVersionService
from app.services.owner_resolution import OPENAPI_PLACEHOLDER_UUID, resolve_owner_user_id
from app.services.processing_job_service import ProcessingJobService
from app.services.system_setting_service import SystemSettingService
from app.services.user_service import UserService
from app.storage.local_file_storage import LocalFileStorage


async def get_current_user(
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> User:
    """Extract and cryptographically verify authenticated user from Authorization: Bearer <jwt_token> header.
    
    Unverified client-controlled identity indicators (such as X-User-Id header or user_id query parameters)
    are strictly untrusted and ignored for authentication purposes.
    """
    import uuid
    from app.api.security import InvalidTokenError, TokenExpiredError, decode_access_token

    auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: Missing or invalid Authorization Bearer header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header[7:].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: Empty Bearer token supplied.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_access_token(token)
        user_id_str = payload.get("sub")
        if not user_id_str:
            raise InvalidTokenError("JWT token missing 'sub' claim.")
        user_uuid = uuid.UUID(user_id_str)
    except TokenExpiredError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: Token has expired. {exc}",
            headers={"WWW-Authenticate": "Bearer error=\"invalid_token\", error_description=\"token_expired\""},
        ) from exc
    except (InvalidTokenError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: Invalid or tampered token. {exc}",
            headers={"WWW-Authenticate": "Bearer error=\"invalid_token\""},
        ) from exc

    user_repo = UserRepository(session)
    user = await user_repo.get(user_uuid)
    if not user or not user.is_active or user.deleted_at is not None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: User {user_uuid} is inactive, deleted, or does not exist.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


UPLOAD_ALLOWED_ROLES: set[str] = {"admin"}


def can_upload_documents(user: User | None) -> bool:
    """Return True if the user has permission to upload documents."""
    if not user or not user.role:
        return False
    role_val = getattr(user.role, "value", str(user.role)).lower()
    return role_val in UPLOAD_ALLOWED_ROLES


async def require_document_upload_permission(
    current_user: User = Depends(get_current_user),
) -> User:
    """Authorize document upload actions.

    Currently restricted to Admin users. Raises HTTP 403 Forbidden if unauthorized.
    """
    if not can_upload_documents(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to upload documents.",
        )
    return current_user


def get_user_service(session: AsyncSession = Depends(get_db)) -> UserService:
    return UserService(session)


def get_document_service(session: AsyncSession = Depends(get_db)) -> DocumentService:
    return DocumentService(session)


from app.storage.base import FileStorage
from app.storage.s3_storage_service import S3StorageService
from app.storage.supabase_storage_service import SupabaseStorageService

def _get_storage_backend() -> FileStorage:
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
    from app.llm.ollama_client import get_global_ollama_client
    return RAGService(session, llm_client=get_global_ollama_client())


def get_memory_manager(session: AsyncSession = Depends(get_db)):
    from app.memory.manager import MemoryManager
    return MemoryManager(session)


def get_long_term_store(session: AsyncSession = Depends(get_db)):
    from app.memory.long_term_store import LongTermMemoryStore
    return LongTermMemoryStore(session)



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
