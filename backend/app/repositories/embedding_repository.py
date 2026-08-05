"""Data access for `app.models.embedding.Embedding`."""
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.embedding import Embedding
from app.repositories.base_repository import BaseRepository


class EmbeddingRepository(BaseRepository[Embedding, uuid.UUID]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Embedding)

    async def list_by_chunk(self, chunk_id: uuid.UUID) -> list[Embedding]:
        stmt = select(Embedding).where(Embedding.chunk_id == chunk_id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_chunk_and_model(self, chunk_id: uuid.UUID, model_name: str) -> Embedding | None:
        """Backs the `embeddings_chunk_model_unique` constraint."""
        stmt = select(Embedding).where(Embedding.chunk_id == chunk_id, Embedding.model_name == model_name)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def find_similar(
        self, query_embedding: list[float], *, model_name: str, limit: int = 10
    ) -> list[Embedding]:
        """Nearest-neighbor search by cosine distance, scoped to one
        embedding model (matching vectors from different models are not
        comparable). Relies on the `embeddings_embedding_hnsw_cosine_idx`
        HNSW index — see `app/models/embedding.py`.

        This is a plain ranked query, not retrieval "business logic": it
        does not filter by document ownership or build a RAG prompt — that
        composition belongs to a future service layer.
        """
        stmt = (
            select(Embedding)
            .where(Embedding.model_name == model_name)
            .order_by(Embedding.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
