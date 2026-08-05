"""Unit tests for `app.retrieval.retriever`."""
import uuid
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from app.retrieval.retriever import RetrievalError, Retriever
from app.retrieval.search import SearchFilters, SearchHit


@dataclass
class FakeEmbeddingClient:
    vector: list[float] | None = None
    calls: list[str] | None = None

    def __post_init__(self) -> None:
        if self.vector is None:
            self.vector = [0.1] * 768
        if self.calls is None:
            self.calls = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return self.vector

    async def close(self) -> None:
        return None


def _make_hits(count: int = 2) -> list[SearchHit]:
    return [
        SearchHit(
            chunk_id=uuid.uuid4(),
            chunk_text=f"text {i}",
            document_id=uuid.uuid4(),
            document_version_id=uuid.uuid4(),
            distance=0.1 + i * 0.1,
        )
        for i in range(count)
    ]


class TestRetriever:
    @pytest.mark.asyncio
    async def test_retrieve_embeds_question_and_returns_ranked_results(self) -> None:
        client = FakeEmbeddingClient()
        hits = _make_hits(2)
        search_mock = AsyncMock(return_value=hits)

        with patch("app.retrieval.retriever.search_similar", search_mock):
            retriever = Retriever(session=None, client=client, top_k=5, similarity_threshold=0.0)
            results = await retriever.retrieve("What is revenue?")

        assert client.calls == ["What is revenue?"]
        search_mock.assert_awaited_once()
        assert len(results) == 2
        assert results[0].rank == 1
        assert results[0].similarity_score >= results[1].similarity_score
        assert results[0].chunk_text == "text 0"

    @pytest.mark.asyncio
    async def test_retrieve_passes_filters_to_search(self) -> None:
        client = FakeEmbeddingClient()
        search_mock = AsyncMock(return_value=_make_hits(1))
        user_id = uuid.uuid4()
        document_id = uuid.uuid4()
        version_id = uuid.uuid4()
        filters = SearchFilters(user_id=user_id, document_id=document_id, document_version_id=version_id)

        with patch("app.retrieval.retriever.search_similar", search_mock):
            retriever = Retriever(session=None, client=client)
            await retriever.retrieve("question", filters=filters, top_k=3)

        call_kwargs = search_mock.await_args.kwargs
        assert call_kwargs["top_k"] == 3
        assert call_kwargs["filters"] == filters

    @pytest.mark.asyncio
    async def test_retrieve_applies_similarity_threshold(self) -> None:
        client = FakeEmbeddingClient()
        hits = [
            SearchHit(
                chunk_id=uuid.uuid4(),
                chunk_text="good",
                document_id=uuid.uuid4(),
                document_version_id=uuid.uuid4(),
                distance=0.1,
            ),
            SearchHit(
                chunk_id=uuid.uuid4(),
                chunk_text="weak",
                document_id=uuid.uuid4(),
                document_version_id=uuid.uuid4(),
                distance=0.8,
            ),
        ]
        search_mock = AsyncMock(return_value=hits)

        with patch("app.retrieval.retriever.search_similar", search_mock):
            retriever = Retriever(session=None, client=client)
            results = await retriever.retrieve("question", similarity_threshold=0.5)

        assert len(results) == 1
        assert results[0].chunk_text == "good"

    @pytest.mark.asyncio
    async def test_retrieve_empty_search_returns_empty_list(self) -> None:
        client = FakeEmbeddingClient()
        search_mock = AsyncMock(return_value=[])

        with patch("app.retrieval.retriever.search_similar", search_mock):
            retriever = Retriever(session=None, client=client)
            results = await retriever.retrieve("no matches")

        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_rejects_empty_question(self) -> None:
        retriever = Retriever(session=None, client=FakeEmbeddingClient())
        with pytest.raises(RetrievalError, match="empty"):
            await retriever.retrieve("   ")

    @pytest.mark.asyncio
    async def test_retrieve_rejects_invalid_top_k(self) -> None:
        retriever = Retriever(session=None, client=FakeEmbeddingClient())
        with pytest.raises(RetrievalError, match="top_k"):
            await retriever.retrieve("question", top_k=0)

    @pytest.mark.asyncio
    async def test_retrieve_rejects_invalid_threshold(self) -> None:
        retriever = Retriever(session=None, client=FakeEmbeddingClient())
        with pytest.raises(RetrievalError, match="similarity_threshold"):
            await retriever.retrieve("question", similarity_threshold=1.5)
