"""Data access for `app.models.citation.Citation`."""
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.citation import Citation
from app.repositories.base_repository import BaseRepository


class CitationRepository(BaseRepository[Citation, uuid.UUID]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Citation)

    async def list_by_message(self, message_id: uuid.UUID) -> list[Citation]:
        """Ordered by `rank` — the order chunks were actually presented to the model."""
        stmt = select(Citation).where(Citation.message_id == message_id).order_by(Citation.rank)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_by_chunk(self, chunk_id: uuid.UUID) -> list[Citation]:
        """Reverse lookup: every message that cited a given chunk."""
        stmt = select(Citation).where(Citation.chunk_id == chunk_id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
