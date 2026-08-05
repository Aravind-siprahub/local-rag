"""Convert cosine distances to similarity scores and assign ranks."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.retrieval.search import SearchHit


@dataclass(frozen=True)
class RankedResult:
    """One retrieval result after threshold filtering and ranking."""

    chunk_id: uuid.UUID
    chunk_text: str
    document_id: uuid.UUID
    document_version_id: uuid.UUID
    similarity_score: float
    rank: int


def cosine_distance_to_similarity(distance: float) -> float:
    """Map pgvector cosine distance to cosine similarity.

    For normalized vectors, cosine distance is in [0, 2] and similarity is
    `1 - distance`, yielding 1.0 for identical vectors.
    """
    return 1.0 - distance


def rank_results(hits: list[SearchHit], similarity_threshold: float) -> list[RankedResult]:
    """Filter by similarity threshold and assign 1-based ranks.

    Input hits must already be ordered by ascending distance (most similar
    first). Results below the threshold are dropped; ranks reflect final order.
    """
    ranked: list[RankedResult] = []
    rank = 1

    for hit in hits:
        similarity = cosine_distance_to_similarity(hit.distance)
        if similarity < similarity_threshold:
            continue
        ranked.append(
            RankedResult(
                chunk_id=hit.chunk_id,
                chunk_text=hit.chunk_text,
                document_id=hit.document_id,
                document_version_id=hit.document_version_id,
                similarity_score=similarity,
                rank=rank,
            )
        )
        rank += 1

    return ranked
