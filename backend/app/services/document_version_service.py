"""Business logic for `app.models.document_version.DocumentVersion`."""
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_version import DocumentVersion
from app.repositories.document_repository import DocumentRepository
from app.repositories.document_version_repository import DocumentVersionRepository
from app.services.base_service import BaseService
from app.services.exceptions import NotFoundError


class DocumentVersionService(BaseService[DocumentVersion, uuid.UUID, DocumentVersionRepository]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, DocumentVersionRepository(session))
        self._documents = DocumentRepository(session)

    async def create_next_version(
        self,
        *,
        document_id: uuid.UUID,
        uploaded_by: uuid.UUID,
        storage_key: str,
        original_filename: str,
        mime_type: str,
        file_size_bytes: int,
        checksum_sha256: str,
        page_count: int | None = None,
    ) -> DocumentVersion:
        """Business rule: callers upload a *file* for a document, not a
        specific version number — this computes the next `version_number`
        automatically (existing versions' max + 1, or 1 for the first)
        rather than requiring the caller to know it and racing the
        `document_versions_doc_version_unique` constraint by guessing.

        Note: under concurrent uploads for the same document, two calls
        could compute the same "next" number; the database's unique
        constraint is still the final authority and will reject the loser
        with an `IntegrityError` — this method optimizes the common case,
        it does not replace that guarantee.
        """
        document = await self._documents.get(document_id)
        if document is None:
            raise NotFoundError(f"Document with id={document_id!r} was not found.")

        existing_versions = await self._versions_for(document_id)
        next_version_number = max((v.version_number for v in existing_versions), default=0) + 1

        return await self.create(
            document_id=document_id,
            version_number=next_version_number,
            storage_key=storage_key,
            original_filename=original_filename,
            mime_type=mime_type,
            file_size_bytes=file_size_bytes,
            checksum_sha256=checksum_sha256,
            page_count=page_count,
            uploaded_by=uploaded_by,
        )

    async def _versions_for(self, document_id: uuid.UUID) -> list[DocumentVersion]:
        return await self.repository.list_by_document(document_id, limit=10_000)

    async def list_by_document(
        self, document_id: uuid.UUID, *, limit: int = 100, offset: int = 0
    ) -> list[DocumentVersion]:
        return await self.repository.list_by_document(document_id, limit=limit, offset=offset)

    async def get_current_version(self, document_id: uuid.UUID) -> DocumentVersion | None:
        """Get the current version for a document (or the latest version if current is unset)."""
        document = await self._documents.get(document_id)
        if document and document.current_version_id:
            curr = await self.get(document.current_version_id)
            if curr:
                return curr
        versions = await self.list_by_document(document_id, limit=100)
        return versions[-1] if versions else None

