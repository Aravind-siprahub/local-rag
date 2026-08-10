"""Business logic for `app.models.document_chunk.DocumentChunk`.

No update workflow is exposed: chunks are immutable once produced (the ORM
model has no `updated_at`), matching the design already established in
`app/models/document_chunk.py` and `app/schemas/document_chunk.py`.
Re-chunking a document version means creating new chunk rows, not editing
existing ones.
"""
import uuid
from typing import Any, TypedDict

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_chunk import DocumentChunk
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.document_version_repository import DocumentVersionRepository
from app.services.base_service import BaseService
from app.services.exceptions import NotFoundError


class ChunkInput(TypedDict, total=False):
    chunk_index: int
    content: str
    content_tokens: int | None
    page_number: int | None
    section_title: str | None
    char_start: int | None
    char_end: int | None
    metadata_: dict[str, Any]


class DocumentChunkService(BaseService[DocumentChunk, uuid.UUID, DocumentChunkRepository]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, DocumentChunkRepository(session))
        self._versions = DocumentVersionRepository(session)

    async def create_chunks_for_version(
        self, document_version_id: uuid.UUID, chunks: list[ChunkInput]
    ) -> list[DocumentChunk]:
        """Business rule: chunks must belong to an existing document
        version, and the whole batch is written as one unit of work — if
        any chunk fails to write, none of them should be committed, since a
        partially-chunked document is worse than a document not yet chunked
        at all.
        """
        version = await self._versions.get(document_version_id)
        if version is None:
            raise NotFoundError(f"DocumentVersion with id={document_version_id!r} was not found.")

        try:
            created = [
                await self.repository.create(document_version_id=document_version_id, **chunk)
                for chunk in chunks
            ]
        except Exception:
            await self.session.rollback()
            raise

        await self.session.commit()
        return created

    async def list_by_document_version(
        self, document_version_id: uuid.UUID, *, limit: int = 1000, offset: int = 0
    ) -> list[DocumentChunk]:
        return await self.repository.list_by_document_version(document_version_id, limit=limit, offset=offset)
