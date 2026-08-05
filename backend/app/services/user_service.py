"""Business logic for `app.models.user.User`."""
import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import UserRole
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.services.base_service import BaseService
from app.services.exceptions import ConflictError


class UserService(BaseService[User, uuid.UUID, UserRepository]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, UserRepository(session))

    async def create_user(
        self,
        *,
        email: str,
        hashed_password: str,
        full_name: str | None = None,
        role: UserRole = UserRole.MEMBER,
    ) -> User:
        """Business rule: an email must belong to at most one *active*
        account. The database already enforces this (the partial unique
        index `users_email_active_uidx`), but checking here first turns a
        raw `IntegrityError` into a clear `ConflictError` the caller can
        act on before ever opening a write.
        """
        existing = await self.repository.get_by_email(email)
        if existing is not None and existing.deleted_at is None:
            raise ConflictError(f"A user with email {email!r} already exists.")

        return await self.create(
            email=email, hashed_password=hashed_password, full_name=full_name, role=role
        )

    async def delete(self, id_: uuid.UUID) -> None:
        """Soft delete: sets `deleted_at` rather than removing the row, so
        the user's `documents`/`chat_sessions` history stays intact. This
        also frees their email for reuse (the unique index is scoped to
        `deleted_at IS NULL`).
        """
        await self.update(id_, deleted_at=datetime.now(timezone.utc), is_active=False)
