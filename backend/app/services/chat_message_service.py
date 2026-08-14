"""Business logic for `app.models.chat_message.ChatMessage`.

No update workflow: messages are append-only (no `updated_at` column) —
correcting a message means posting a new one, not editing history.
"""
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_message import ChatMessage
from app.models.enums import MessageRole
from app.repositories.chat_message_repository import ChatMessageRepository
from app.repositories.chat_session_repository import ChatSessionRepository
from app.services.base_service import BaseService
from app.services.exceptions import NotFoundError


class ChatMessageService(BaseService[ChatMessage, uuid.UUID, ChatMessageRepository]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, ChatMessageRepository(session))
        self._sessions = ChatSessionRepository(session)

    async def create_message(
        self,
        *,
        session_id: uuid.UUID,
        role: MessageRole,
        content: str,
        model_used: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        latency_ms: int | None = None,
        generation_time_ms: int | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> ChatMessage:
        """Business rules:
        - the parent chat session must exist and not be soft-deleted;
        - posting a message bumps that session's `last_message_at` — both
          writes happen in the same unit of work (one commit), so a message
          is never persisted with a stale/missing session timestamp.
        """
        chat_session = await self._sessions.get(session_id)
        if chat_session is None or chat_session.deleted_at is not None:
            raise NotFoundError(f"ChatSession with id={session_id!r} was not found.")

        try:
            message = await self.repository.create(
                session_id=session_id,
                role=role,
                content=content,
                model_used=model_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                generation_time_ms=generation_time_ms,
                attachments=attachments,
            )
            await self._sessions.update(chat_session, last_message_at=datetime.now(timezone.utc))
        except Exception:
            await self.session.rollback()
            raise

        await self.session.commit()
        await self.session.refresh(message)
        return message

    async def list_by_session(
        self, session_id: uuid.UUID, *, limit: int = 200, offset: int = 0
    ) -> list[ChatMessage]:
        return await self.repository.list_by_session(session_id, limit=limit, offset=offset)
