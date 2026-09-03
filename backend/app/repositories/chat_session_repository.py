"""Data access for `app.models.chat_session.ChatSession`."""
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_session import ChatSession
from app.repositories.base_repository import BaseRepository


class ChatSessionRepository(BaseRepository[ChatSession, uuid.UUID]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, ChatSession)

    async def list_by_user(
        self,
        user_id: uuid.UUID,
        *,
        include_archived: bool = False,
        include_deleted: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ChatSession]:
        """Most-recently-active first, matching `chat_sessions_user_recent_idx`."""
        stmt = select(ChatSession).where(ChatSession.user_id == user_id)
        if not include_deleted:
            stmt = stmt.where(ChatSession.deleted_at.is_(None))
        if not include_archived:
            stmt = stmt.where(ChatSession.is_archived.is_(False))
        stmt = stmt.order_by(
            ChatSession.last_message_at.desc().nullslast(),
            ChatSession.created_at.desc(),
        ).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
