"""Business/application layer — one service per domain, all built on
`BaseService`. Coordinates one or more repositories and owns the
transaction boundary (commit/rollback/refresh); repositories only `flush()`.

Services depend only on `app.models`, `app.repositories`, and each other's
plain-Python exceptions in `exceptions.py` — never on FastAPI, so they stay
importable and unit-testable without a running app. Cross-domain business
rules (e.g. "a document must belong to an existing user") are implemented by
a service instantiating another domain's *repository* directly, not another
*service* — keeping exactly one transaction owner per call.
"""
from app.services.base_service import BaseService
from app.services.chat_message_service import ChatMessageService
from app.services.chat_session_service import ChatSessionService
from app.services.citation_service import CitationService
from app.services.document_chunk_service import DocumentChunkService
from app.services.document_service import DocumentService
from app.services.document_upload_service import DocumentUploadService, UploadResult
from app.services.document_version_service import DocumentVersionService
from app.services.embedding_service import EmbeddingService
from app.services.exceptions import ConflictError, NotFoundError, ServiceError, ValidationError
from app.services.processing_job_service import ProcessingJobService
from app.services.system_setting_service import SystemSettingService
from app.services.user_service import UserService

__all__ = [
    "BaseService",
    "ServiceError",
    "NotFoundError",
    "ConflictError",
    "ValidationError",
    "UserService",
    "DocumentService",
    "DocumentUploadService",
    "UploadResult",
    "DocumentVersionService",
    "DocumentChunkService",
    "EmbeddingService",
    "ChatSessionService",
    "ChatMessageService",
    "CitationService",
    "ProcessingJobService",
    "SystemSettingService",
]
