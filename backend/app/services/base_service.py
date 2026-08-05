"""Generic service, subclassed by every domain service.

Layering: services own the transaction boundary. Repositories only
`flush()` (see `app/repositories/base_repository.py`'s docstring); services
are the first layer allowed to `commit()` or `rollback()`. This is what
"services coordinate repositories and own transaction management" means in
practice — a service method is a unit of work: one or more repository calls,
followed by exactly one commit (on success) or rollback (on failure).

`session.refresh(obj)` after commit matters even though `AsyncSessionLocal`
is configured with `expire_on_commit=False` (so attributes aren't wiped and
re-fetched automatically): several tables have a database trigger
(`set_updated_at()`) that sets `updated_at` server-side on `UPDATE`, and
`document_versions`/`document_chunks` etc. have server-computed defaults.
Without an explicit `refresh()`, the Python object would keep whatever
stale value it had before the write, silently disagreeing with what's
actually in the database.
"""
from typing import Generic, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.base_repository import BaseRepository
from app.services.exceptions import NotFoundError

ModelType = TypeVar("ModelType")
IDType = TypeVar("IDType")
RepositoryType = TypeVar("RepositoryType", bound=BaseRepository)


class BaseService(Generic[ModelType, IDType, RepositoryType]):
    def __init__(self, session: AsyncSession, repository: RepositoryType) -> None:
        self.session = session
        self.repository = repository

    async def get(self, id_: IDType) -> ModelType:
        """Fetch by primary key, raising `NotFoundError` if it doesn't exist.

        Use `get_optional` instead when "doesn't exist" is a valid,
        non-exceptional outcome for the caller.
        """
        obj = await self.repository.get(id_)
        if obj is None:
            raise NotFoundError(f"{self.repository.model.__name__} with id={id_!r} was not found.")
        return obj

    async def get_optional(self, id_: IDType) -> ModelType | None:
        return await self.repository.get(id_)

    async def list(self, *, limit: int = 100, offset: int = 0) -> list[ModelType]:
        return await self.repository.list(limit=limit, offset=offset)

    async def count(self) -> int:
        return await self.repository.count()

    async def create(self, **values: object) -> ModelType:
        """Generic create. Domain services override this when creation
        requires business validation (existence checks on related entities,
        uniqueness beyond what the DB enforces, etc.) — call
        `super().create(...)` at the end once validation passes rather than
        duplicating the commit/rollback/refresh dance.
        """
        try:
            obj = await self.repository.create(**values)
        except Exception:
            await self.session.rollback()
            raise
        await self.session.commit()
        await self.session.refresh(obj)
        return obj

    async def update(self, id_: IDType, **values: object) -> ModelType:
        obj = await self.get(id_)
        try:
            updated = await self.repository.update(obj, **values)
        except Exception:
            await self.session.rollback()
            raise
        await self.session.commit()
        await self.session.refresh(updated)
        return updated

    async def delete(self, id_: IDType) -> None:
        """Hard delete. Domain services whose table has a `deleted_at`
        column (`users`, `documents`, `chat_sessions`) override this to
        soft-delete instead — see each service's `delete()`.
        """
        await self.get(id_)  # raises NotFoundError before opening a transaction for nothing
        try:
            await self.repository.delete(id_)
        except Exception:
            await self.session.rollback()
            raise
        await self.session.commit()
