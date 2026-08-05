"""Data access for `app.models.user.User`."""
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.repositories.base_repository import BaseRepository


class UserRepository(BaseRepository[User, uuid.UUID]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, User)

    async def get_by_email(self, email: str) -> User | None:
        """`email` is `CITEXT`, so this comparison is case-insensitive at
        the database level — no `.lower()`/`.ilike()` needed here.
        """
        stmt = select(User).where(User.email == email)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_active(self, *, limit: int = 100, offset: int = 0) -> list[User]:
        """Excludes soft-deleted users — mirrors the `deleted_at IS NULL`
        scoping used throughout the partial indexes on this table.
        """
        stmt = select(User).where(User.deleted_at.is_(None)).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
