"""Data-access layer — one repository per domain, all built on
`BaseRepository`. Encapsulates queries against `app.models`; never imports
`app.schemas`, per the Clean Architecture dependency rule (see
`base_repository.py`'s module docstring).

Reserved for the next step: services orchestrate these repositories and are
the first layer allowed to `commit()` a transaction.
"""
from app.repositories.base_repository import BaseRepository
from app.repositories.chat_message_repository import ChatMessageRepository
from app.repositories.chat_session_repository import ChatSessionRepository
from app.repositories.citation_repository import CitationRepository
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.document_version_repository import DocumentVersionRepository
from app.repositories.embedding_repository import EmbeddingRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.repositories.system_setting_repository import SystemSettingRepository
from app.repositories.user_repository import UserRepository

__all__ = [
    "BaseRepository",
    "UserRepository",
    "DocumentRepository",
    "DocumentVersionRepository",
    "DocumentChunkRepository",
    "EmbeddingRepository",
    "ChatSessionRepository",
    "ChatMessageRepository",
    "CitationRepository",
    "ProcessingJobRepository",
    "SystemSettingRepository",
]
