"""Business logic for `app.models.citation.Citation`.

No update workflow: a citation is an immutable record of what was retrieved
for a message (no `updated_at` column) — if retrieval changes, regenerate
the message's citations rather than editing one in place.
"""
import uuid
from typing import TypedDict

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.citation import Citation
from app.repositories.chat_message_repository import ChatMessageRepository
from app.repositories.citation_repository import CitationRepository
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.services.base_service import BaseService
from app.services.exceptions import NotFoundError


class CitationInput(TypedDict, total=False):
    chunk_id: uuid.UUID
    rank: int
    similarity_score: float | None


class CitationService(BaseService[Citation, uuid.UUID, CitationRepository]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, CitationRepository(session))
        self._messages = ChatMessageRepository(session)
        self._chunks = DocumentChunkRepository(session)

    async def create_citations_for_message(
        self, message_id: uuid.UUID, citations: list[CitationInput]
    ) -> list[Citation]:
        """Business rules: the message must exist, every cited chunk must
        exist, and the whole batch (typically "the top-k retrieved chunks
        for this answer") is written as one unit of work.
        """
        message = await self._messages.get(message_id)
        if message is None:
            raise NotFoundError(f"ChatMessage with id={message_id!r} was not found.")

        for citation in citations:
            chunk = await self._chunks.get(citation["chunk_id"])
            if chunk is None:
                raise NotFoundError(f"DocumentChunk with id={citation['chunk_id']!r} was not found.")

        try:
            created = [
                await self.repository.create(message_id=message_id, **citation) for citation in citations
            ]
        except Exception:
            await self.session.rollback()
            raise

        await self.session.commit()
        for citation_obj in created:
            await self.session.refresh(citation_obj)
        return created

    async def list_by_message(self, message_id: uuid.UUID) -> list[Citation]:
        return await self.repository.list_by_message(message_id)
