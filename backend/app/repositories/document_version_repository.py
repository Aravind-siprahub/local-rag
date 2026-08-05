"""Data access for `app.models.document_version.DocumentVersion`."""
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_version import DocumentVersion
from app.repositories.base_repository import BaseRepository


class DocumentVersionRepository(BaseRepository[DocumentVersion, uuid.UUID]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, DocumentVersion)

    async def list_by_document(
        self, document_id: uuid.UUID, *, limit: int = 100, offset: int = 0
    ) -> list[DocumentVersion]:
        stmt = (
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.version_number)
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_document_and_version(
        self, document_id: uuid.UUID, version_number: int
    ) -> DocumentVersion | None:
        """Backs the `document_versions_doc_version_unique` constraint —
        looks up one specific version of a document.
        """
        stmt = select(DocumentVersion).where(
            DocumentVersion.document_id == document_id,
            DocumentVersion.version_number == version_number,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
