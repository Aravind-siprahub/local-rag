"""Data access for `app.models.chat_message.ChatMessage`."""
from __future__ import annotations

import inspect
import uuid
from typing import Any, List

from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_message import ChatMessage
from app.models.citation import Citation
from app.models.document_chunk import DocumentChunk
from app.models.document_version import DocumentVersion
from app.repositories.base_repository import BaseRepository


def _safe_scalars_list(result: Any) -> List[ChatMessage]:
    if inspect.isawaitable(result):
        return []
    if hasattr(result, "scalars"):
        try:
            sc = result.scalars()
            if inspect.isawaitable(sc):
                return []
            if hasattr(sc, "all"):
                res = sc.all()
                if inspect.isawaitable(res):
                    return []
                return list(res)
        except Exception:
            return []
    return []


class ChatMessageRepository(BaseRepository[ChatMessage, uuid.UUID]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, ChatMessage)

    async def list_by_session(
        self, session_id: uuid.UUID, *, limit: int = 200, offset: int = 0
    ) -> List[ChatMessage]:
        """Oldest first — the natural order for rendering a transcript."""
        stmt = (
            select(ChatMessage)
            .options(
                selectinload(ChatMessage.citations)
                .selectinload(Citation.chunk)
                .selectinload(DocumentChunk.document_version)
                .selectinload(DocumentVersion.document)
            )
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return _safe_scalars_list(result)
