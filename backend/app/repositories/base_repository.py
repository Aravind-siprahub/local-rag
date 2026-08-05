"""Generic async CRUD repository, subclassed by every domain repository.

Scope discipline (Clean Architecture's dependency rule): this module — and
every repository built on it — imports only `app.models` and SQLAlchemy.
It never imports `app.schemas`. Pydantic schemas belong to the interface
layer; if repositories accepted/returned them, the data-access layer would
depend outward on the presentation layer instead of the reverse. Callers
pass plain keyword arguments (or a `dict`) for writes and receive ORM
instances back — converting those to/from Pydantic schemas is the future
service/API layer's job, not this one's.

Transaction ownership: repositories `flush()` but never `commit()` or
`rollback()`. This mirrors `get_db()` in `app/db/session.py`, where the
caller (eventually a service function) owns the transaction boundary and
decides when a unit of work is done. `flush()` is enough to send pending
SQL to the database within the current transaction — needed so
server-generated values (`id`, `created_at`, computed columns) are
populated on the returned object — without prematurely ending a
transaction a calling service might still be composing.
"""
from typing import Any, Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import Base

ModelType = TypeVar("ModelType", bound=Base)
IDType = TypeVar("IDType")


class BaseRepository(Generic[ModelType, IDType]):
    """Common async CRUD operations for a single ORM model.

    `IDType` is generic because primary keys aren't uniformly `UUID` in
    this schema — `SystemSetting`'s primary key is a `str` key.
    """

    def __init__(self, session: AsyncSession, model: type[ModelType]) -> None:
        self.session = session
        self.model = model

    async def get(self, id_: IDType) -> ModelType | None:
        """Fetch by primary key, or `None` if it doesn't exist."""
        return await self.session.get(self.model, id_)

    async def list(self, *, limit: int = 100, offset: int = 0) -> list[ModelType]:
        """Fetch a page of rows in default (primary key) order."""
        stmt = select(self.model).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count(self) -> int:
        """Total row count, ignoring `limit`/`offset` — for pagination totals."""
        stmt = select(func.count()).select_from(self.model)
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def create(self, **values: Any) -> ModelType:
        """Instantiate and stage a new row. Flushes so DB-generated defaults
        (id, created_at, ...) are populated on the returned instance.
        """
        obj = self.model(**values)
        self.session.add(obj)
        await self.session.flush()
        return obj

    async def update(self, db_obj: ModelType, **values: Any) -> ModelType:
        """Apply attribute changes to an already-loaded instance and flush.

        Takes the loaded object rather than an id so callers that already
        fetched it (e.g. to check it exists, or as part of a larger unit of
        work) don't pay for a second `SELECT`.
        """
        for key, value in values.items():
            setattr(db_obj, key, value)
        await self.session.flush()
        return db_obj

    async def delete(self, id_: IDType) -> bool:
        """Delete by primary key. Returns whether a row was actually deleted."""
        obj = await self.get(id_)
        if obj is None:
            return False
        await self.session.delete(obj)
        await self.session.flush()
        return True
