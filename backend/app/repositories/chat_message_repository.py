"""Data access for `app.models.chat_message.ChatMessage`."""
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_message import ChatMessage
from app.repositories.base_repository import BaseRepository


class ChatMessageRepository(BaseRepository[ChatMessage, uuid.UUID]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, ChatMessage)

    async def list_by_session(
        self, session_id: uuid.UUID, *, limit: int = 200, offset: int = 0
    ) -> list[ChatMessage]:
        """Oldest first — the natural order for rendering a transcript."""
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
