"""Business logic for `app.models.document.Document`."""
import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.repositories.document_repository import DocumentRepository
from app.repositories.document_version_repository import DocumentVersionRepository
from app.repositories.user_repository import UserRepository
from app.services.base_service import BaseService
from app.services.exceptions import NotFoundError, ValidationError
from app.services.owner_resolution import resolve_owner_user_id


class DocumentService(BaseService[Document, uuid.UUID, DocumentRepository]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, DocumentRepository(session))
        # Cross-entity business rules below need these; instantiated on the
        # same session so they participate in the same transaction/unit of
        # work as this service's own repository.
        self._users = UserRepository(session)
        self._versions = DocumentVersionRepository(session)

    async def create_document(
        self, *, user_id: uuid.UUID, title: str, description: str | None = None, tags: list[str] | None = None
    ) -> Document:
        """Business rule: a document must belong to an existing user."""
        user_id = await resolve_owner_user_id(user_id, self._users)

        owner = await self._users.get(user_id)
        if owner is None:
            raise NotFoundError(
                f"User with id={user_id!r} was not found. "
                "Create a user via POST /users and use the returned id."
            )

        return await self.create(user_id=user_id, title=title, description=description, tags=tags or [])

    async def set_current_version(self, document_id: uuid.UUID, version_id: uuid.UUID) -> Document:
        """Business rule: the version being promoted to "current" must
        actually belong to this document — otherwise a caller could point a
        document at an unrelated document's version.
        """
        document = await self.get(document_id)
        version = await self._versions.get(version_id)
        if version is None or version.document_id != document_id:
            raise ValidationError(
                f"Version {version_id!r} does not belong to document {document_id!r}."
            )

        return await self.update(document_id, current_version_id=version_id)

    async def list_by_user(self, user_id: uuid.UUID, *, limit: int = 100, offset: int = 0) -> list[Document]:
        return await self.repository.list_by_user(user_id, limit=limit, offset=offset)

    async def delete(self, id_: uuid.UUID) -> None:
        """Soft delete: sets `deleted_at`. Existing `document_versions` and
        their chunks/embeddings are left intact — retrieval/service code
        that lists documents is expected to filter out soft-deleted ones
        (see `DocumentRepository.list_by_user`'s `include_deleted` flag).
        """
        await self.update(id_, deleted_at=datetime.now(timezone.utc))
