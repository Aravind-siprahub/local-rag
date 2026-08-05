"""Data access for `app.models.document.Document`."""
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.enums import DocumentStatus
from app.repositories.base_repository import BaseRepository


class DocumentRepository(BaseRepository[Document, uuid.UUID]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Document)

    async def list_by_user(
        self, user_id: uuid.UUID, *, include_deleted: bool = False, limit: int = 100, offset: int = 0
    ) -> list[Document]:
        stmt = select(Document).where(Document.user_id == user_id)
        if not include_deleted:
            stmt = stmt.where(Document.deleted_at.is_(None))
        stmt = stmt.limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_by_status(
        self, status: DocumentStatus, *, limit: int = 100, offset: int = 0
    ) -> list[Document]:
        stmt = (
            select(Document)
            .where(Document.status == status, Document.deleted_at.is_(None))
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
