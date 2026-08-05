"""Unit tests for `app.retrieval.ranking`."""
import uuid

from app.retrieval.ranking import cosine_distance_to_similarity, rank_results
from app.retrieval.search import SearchHit


def _hit(distance: float, suffix: str = "a") -> SearchHit:
    return SearchHit(
        chunk_id=uuid.uuid4(),
        chunk_text=f"chunk {suffix}",
        document_id=uuid.uuid4(),
        document_version_id=uuid.uuid4(),
        distance=distance,
    )


class TestCosineDistanceToSimilarity:
    def test_identical_vectors(self) -> None:
        assert cosine_distance_to_similarity(0.0) == 1.0

    def test_opposite_vectors(self) -> None:
        assert cosine_distance_to_similarity(1.0) == 0.0


class TestRankResults:
    def test_assigns_ranks_in_distance_order(self) -> None:
        hits = [_hit(0.1, "best"), _hit(0.3, "mid"), _hit(0.5, "low")]
        results = rank_results(hits, similarity_threshold=0.0)

        assert len(results) == 3
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3
        assert results[0].similarity_score > results[1].similarity_score
        assert results[0].chunk_text == "chunk best"

    def test_filters_by_similarity_threshold(self) -> None:
        hits = [_hit(0.1), _hit(0.4), _hit(0.6)]
        results = rank_results(hits, similarity_threshold=0.5)

        assert len(results) == 2
        assert all(result.similarity_score >= 0.5 for result in results)
        assert results[0].rank == 1
        assert results[1].rank == 2

    def test_empty_hits_returns_empty_list(self) -> None:
        assert rank_results([], similarity_threshold=0.0) == []

    def test_all_below_threshold_returns_empty(self) -> None:
        hits = [_hit(0.9), _hit(0.95)]
        assert rank_results(hits, similarity_threshold=0.5) == []

    def test_preserves_metadata(self) -> None:
        chunk_id = uuid.uuid4()
        document_id = uuid.uuid4()
        version_id = uuid.uuid4()
        hit = SearchHit(
            chunk_id=chunk_id,
            chunk_text="hello",
            document_id=document_id,
            document_version_id=version_id,
            distance=0.2,
        )
        result = rank_results([hit], similarity_threshold=0.0)[0]

        assert result.chunk_id == chunk_id
        assert result.chunk_text == "hello"
        assert result.document_id == document_id
        assert result.document_version_id == version_id
