"""Business logic for `app.models.embedding.Embedding`.

No update workflow: embeddings are insert-only (unique per
`chunk_id` + `model_name`) — re-embedding with a new model version creates a
new row rather than editing a vector in place.
"""
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.embedding import EMBEDDING_DIM, Embedding
from app.models.enums import VectorMetric
from app.repositories.document_chunk_repository import DocumentChunkRepository
from app.repositories.embedding_repository import EmbeddingRepository
from app.services.base_service import BaseService
from app.services.exceptions import ConflictError, NotFoundError, ValidationError


class EmbeddingService(BaseService[Embedding, uuid.UUID, EmbeddingRepository]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, EmbeddingRepository(session))
        self._chunks = DocumentChunkRepository(session)

    async def create_embedding(
        self,
        *,
        chunk_id: uuid.UUID,
        model_name: str,
        embedding: list[float],
        model_version: str | None = None,
        dimensions: int = EMBEDDING_DIM,
        metric: VectorMetric = VectorMetric.COSINE,
    ) -> Embedding:
        """Business rules:
        - the chunk being embedded must exist;
        - the vector's actual length must match its declared `dimensions`
          (a caller could otherwise construct an internally-inconsistent
          row bypassing the API-layer Pydantic validation entirely);
        - at most one embedding per (chunk, model) — checked here for a
          clear `ConflictError` on top of the DB's own unique constraint.
        """
        chunk = await self._chunks.get(chunk_id)
        if chunk is None:
            raise NotFoundError(f"DocumentChunk with id={chunk_id!r} was not found.")

        if len(embedding) != dimensions:
            raise ValidationError(
                f"embedding has {len(embedding)} dimensions but dimensions={dimensions} was declared."
            )

        existing = await self.repository.get_by_chunk_and_model(chunk_id, model_name)
        if existing is not None:
            raise ConflictError(
                f"Chunk {chunk_id!r} already has an embedding from model {model_name!r}."
            )

        return await self.create(
            chunk_id=chunk_id,
            model_name=model_name,
            model_version=model_version,
            dimensions=dimensions,
            metric=metric,
            embedding=embedding,
        )

    async def find_similar(
        self, query_embedding: list[float], *, model_name: str, limit: int = 10
    ) -> list[Embedding]:
        """Thin pass-through to the repository's ANN search. Deliberately
        does not filter by document ownership or assemble a RAG prompt —
        that composition is a future, higher-level concern, not embedding
        storage business logic.
        """
        if len(query_embedding) != EMBEDDING_DIM:
            raise ValidationError(f"query_embedding must have {EMBEDDING_DIM} dimensions.")
        return await self.repository.find_similar(query_embedding, model_name=model_name, limit=limit)
