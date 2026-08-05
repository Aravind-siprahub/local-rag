"""Pydantic v2 schemas — one module per domain, mirroring `app/models/`.

Every domain module exports `*Base` / `*Create` / `*Response` schemas, plus
`*Update` where the entity actually has mutable fields (immutable/insert-only
tables — chunks, embeddings, chat messages, citations — deliberately have no
Update schema; see each module's docstring for why). `*ListResponse` aliases
are all `PaginatedResponse[...]` from `common.py`, not hand-duplicated shapes.
"""
from app.schemas.actions import (
    DocumentVersionUploadRequest,
    FailJobRequest,
    SetCurrentVersionRequest,
    SystemSettingUpsertRequest,
)
from app.schemas.chat import (
    ChatCitationResponse,
    ChatDocumentFilters,
    ChatRequest,
    ChatResponse,
    ChatTokenUsageResponse,
)
from app.schemas.chat_message import (
    ChatMessageBase,
    ChatMessageCreate,
    ChatMessageListResponse,
    ChatMessageResponse,
)
from app.schemas.chat_session import (
    ChatSessionBase,
    ChatSessionCreate,
    ChatSessionListResponse,
    ChatSessionResponse,
    ChatSessionUpdate,
)
from app.schemas.citation import CitationBase, CitationCreate, CitationListResponse, CitationResponse
from app.schemas.common import CreatedAtSchema, OptionalUUID, ORMModel, PaginatedResponse, TimestampSchema
from app.schemas.document import (
    DocumentBase,
    DocumentCreate,
    DocumentListResponse,
    DocumentResponse,
    DocumentUpdate,
)
from app.schemas.document_chunk import (
    DocumentChunkBase,
    DocumentChunkCreate,
    DocumentChunkListResponse,
    DocumentChunkResponse,
)
from app.schemas.document_version import (
    DocumentVersionBase,
    DocumentVersionCreate,
    DocumentVersionListResponse,
    DocumentVersionResponse,
    DocumentVersionUpdate,
)
from app.schemas.embedding import EmbeddingBase, EmbeddingCreate, EmbeddingListResponse, EmbeddingResponse
from app.schemas.health import HealthErrorResponse, HealthResponse
from app.schemas.processing_job import (
    ProcessingJobBase,
    ProcessingJobCreate,
    ProcessingJobListResponse,
    ProcessingJobResponse,
    ProcessingJobUpdate,
)
from app.schemas.system_setting import (
    SystemSettingBase,
    SystemSettingCreate,
    SystemSettingListResponse,
    SystemSettingResponse,
    SystemSettingUpdate,
)
from app.schemas.upload import DocumentUploadResponse
from app.schemas.user import UserBase, UserCreate, UserListResponse, UserResponse, UserUpdate

__all__ = [
    # actions (API-layer-only request bodies)
    "SetCurrentVersionRequest",
    "DocumentVersionUploadRequest",
    "FailJobRequest",
    "SystemSettingUpsertRequest",
    # common
    "ORMModel",
    "CreatedAtSchema",
    "TimestampSchema",
    "PaginatedResponse",
    "OptionalUUID",
    # health
    "HealthResponse",
    "HealthErrorResponse",
    # user
    "UserBase",
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "UserListResponse",
    # document
    "DocumentBase",
    "DocumentCreate",
    "DocumentUpdate",
    "DocumentResponse",
    "DocumentListResponse",
    # upload
    "DocumentUploadResponse",
    # document_version
    "DocumentVersionBase",
    "DocumentVersionCreate",
    "DocumentVersionUpdate",
    "DocumentVersionResponse",
    "DocumentVersionListResponse",
    # document_chunk
    "DocumentChunkBase",
    "DocumentChunkCreate",
    "DocumentChunkResponse",
    "DocumentChunkListResponse",
    # embedding
    "EmbeddingBase",
    "EmbeddingCreate",
    "EmbeddingResponse",
    "EmbeddingListResponse",
    # chat (RAG API)
    "ChatDocumentFilters",
    "ChatRequest",
    "ChatResponse",
    "ChatCitationResponse",
    "ChatTokenUsageResponse",
    # chat_session
    "ChatSessionBase",
    "ChatSessionCreate",
    "ChatSessionUpdate",
    "ChatSessionResponse",
    "ChatSessionListResponse",
    # chat_message
    "ChatMessageBase",
    "ChatMessageCreate",
    "ChatMessageResponse",
    "ChatMessageListResponse",
    # citation
    "CitationBase",
    "CitationCreate",
    "CitationResponse",
    "CitationListResponse",
    # processing_job
    "ProcessingJobBase",
    "ProcessingJobCreate",
    "ProcessingJobUpdate",
    "ProcessingJobResponse",
    "ProcessingJobListResponse",
    # system_setting
    "SystemSettingBase",
    "SystemSettingCreate",
    "SystemSettingUpdate",
    "SystemSettingResponse",
    "SystemSettingListResponse",
]
