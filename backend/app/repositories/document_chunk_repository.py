"""Data access for `app.models.document_chunk.DocumentChunk`."""
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_chunk import DocumentChunk
from app.repositories.base_repository import BaseRepository


class DocumentChunkRepository(BaseRepository[DocumentChunk, uuid.UUID]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, DocumentChunk)

    async def list_by_document_version(
        self, document_version_id: uuid.UUID, *, limit: int = 1000, offset: int = 0
    ) -> list[DocumentChunk]:
        """Ordered by `chunk_index` to preserve original document order."""
        stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.document_version_id == document_version_id)
            .order_by(DocumentChunk.chunk_index)
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_version_and_index(
        self, document_version_id: uuid.UUID, chunk_index: int
    ) -> DocumentChunk | None:
        """Backs the `document_chunks_index_unique` constraint."""
        stmt = select(DocumentChunk).where(
            DocumentChunk.document_version_id == document_version_id,
            DocumentChunk.chunk_index == chunk_index,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
