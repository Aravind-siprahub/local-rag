"""Unit tests for `app.embeddings.generator`."""
import uuid
from dataclasses import dataclass

import pytest

from app.embeddings.client import EmbeddingClientError
from app.embeddings.generator import EmbeddingGenerator


@dataclass
class _FakeChunk:
    id: uuid.UUID
    chunk_index: int
    content: str


@dataclass
class _FakeEmbedding:
    id: uuid.UUID = uuid.uuid4()


class FakeEmbeddingClient:
    def __init__(self, vector: list[float] | None = None, *, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.vector = vector or [0.5] * 768
        self.fail = fail

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if self.fail:
            raise EmbeddingClientError("simulated failure")
        return self.vector

    async def close(self) -> None:
        return None


class FakeEmbeddingRepository:
    def __init__(self) -> None:
        self.existing: set[tuple[uuid.UUID, str]] = set()

    async def get_by_chunk_and_model(self, chunk_id: uuid.UUID, model_name: str) -> _FakeEmbedding | None:
        if (chunk_id, model_name) in self.existing:
            return _FakeEmbedding()
        return None


class FakeEmbeddingService:
    def __init__(self, repository: FakeEmbeddingRepository | None = None) -> None:
        self.repository = repository or FakeEmbeddingRepository()
        self.created: list[dict] = []
        self.session = None

    async def create_embedding(self, **kwargs) -> _FakeEmbedding:
        self.created.append(kwargs)
        chunk_id = kwargs["chunk_id"]
        model_name = kwargs["model_name"]
        self.repository.existing.add((chunk_id, model_name))
        return _FakeEmbedding()


def _make_chunks(count: int = 2) -> list[_FakeChunk]:
    return [
        _FakeChunk(id=uuid.uuid4(), chunk_index=i, content=f"chunk text {i}")
        for i in range(count)
    ]


class TestEmbeddingGenerator:
    @pytest.mark.asyncio
    async def test_embeds_all_chunks(self) -> None:
        client = FakeEmbeddingClient()
        service = FakeEmbeddingService()
        generator = EmbeddingGenerator(
            session=None,
            client=client,
            model_name="test-model",
            dimensions=768,
            embedding_service=service,
        )
        chunks = _make_chunks(2)

        result = await generator.embed_chunks(chunks)

        assert result.embedded_count == 2
        assert result.skipped_count == 0
        assert result.total_chunks == 2
        assert len(client.calls) == 2
        assert len(service.created) == 2
        assert all(len(row["embedding"]) == 768 for row in service.created)

    @pytest.mark.asyncio
    async def test_skips_duplicate_chunks(self) -> None:
        client = FakeEmbeddingClient()
        repository = FakeEmbeddingRepository()
        chunks = _make_chunks(2)
        repository.existing.add((chunks[0].id, "test-model"))
        service = FakeEmbeddingService(repository)
        generator = EmbeddingGenerator(
            session=None,
            client=client,
            model_name="test-model",
            dimensions=768,
            embedding_service=service,
        )

        result = await generator.embed_chunks(chunks)

        assert result.embedded_count == 1
        assert result.skipped_count == 1
        assert len(client.calls) == 1
        assert len(service.created) == 1
        assert service.created[0]["chunk_id"] == chunks[1].id

    @pytest.mark.asyncio
    async def test_raises_on_client_failure(self) -> None:
        client = FakeEmbeddingClient(fail=True)
        service = FakeEmbeddingService()
        generator = EmbeddingGenerator(
            session=None,
            client=client,
            model_name="test-model",
            dimensions=768,
            embedding_service=service,
        )

        with pytest.raises(EmbeddingClientError, match="Failed to embed"):
            await generator.embed_chunks(_make_chunks(1))

        assert service.created == []
