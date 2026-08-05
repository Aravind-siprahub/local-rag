"""Business logic for `app.models.chat_session.ChatSession`."""
import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_session import ChatSession
from app.repositories.chat_session_repository import ChatSessionRepository
from app.repositories.user_repository import UserRepository
from app.services.base_service import BaseService
from app.services.exceptions import NotFoundError


class ChatSessionService(BaseService[ChatSession, uuid.UUID, ChatSessionRepository]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, ChatSessionRepository(session))
        self._users = UserRepository(session)

    async def create_session(self, *, user_id: uuid.UUID, title: str = "New chat") -> ChatSession:
        """Business rule: a chat session must belong to an existing user."""
        from app.services.owner_resolution import resolve_owner_user_id

        user_id = await resolve_owner_user_id(user_id, self._users)
        owner = await self._users.get(user_id)
        if owner is None:
            raise NotFoundError(
                f"User with id={user_id!r} was not found. "
                "Create a user via POST /users and use the returned id."
            )

        return await self.create(user_id=user_id, title=title)

    async def archive(self, session_id: uuid.UUID) -> ChatSession:
        return await self.update(session_id, is_archived=True)

    async def unarchive(self, session_id: uuid.UUID) -> ChatSession:
        return await self.update(session_id, is_archived=False)

    async def list_by_user(
        self, user_id: uuid.UUID, *, include_archived: bool = False, limit: int = 100, offset: int = 0
    ) -> list[ChatSession]:
        return await self.repository.list_by_user(
            user_id, include_archived=include_archived, limit=limit, offset=offset
        )

    async def delete(self, id_: uuid.UUID) -> None:
        """Soft delete: sets `deleted_at`, preserving the transcript
        (`chat_messages`/`citations`) for audit purposes rather than
        cascading a hard delete through it.
        """
        await self.update(id_, deleted_at=datetime.now(timezone.utc))
