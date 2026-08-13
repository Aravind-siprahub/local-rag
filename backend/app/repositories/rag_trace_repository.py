"""Data access for `app.models.rag_trace.RAGTrace`."""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.rag_trace import RAGTrace
from app.repositories.base_repository import BaseRepository


class RAGTraceRepository(BaseRepository[RAGTrace, uuid.UUID]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, RAGTrace)

    async def get_by_request_id(self, request_id: str) -> RAGTrace | None:
        stmt = select(RAGTrace).where(RAGTrace.request_id == request_id).order_by(RAGTrace.created_at.desc())
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def list_by_session(self, session_id: uuid.UUID, *, limit: int = 50) -> list[RAGTrace]:
        stmt = (
            select(RAGTrace)
            .where(RAGTrace.session_id == session_id)
            .order_by(RAGTrace.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_by_user(self, user_id: uuid.UUID, *, limit: int = 50) -> list[RAGTrace]:
        stmt = (
            select(RAGTrace)
            .where(RAGTrace.user_id == user_id)
            .order_by(RAGTrace.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
