"""Convert cosine distances to similarity scores, assign ranks, and execute real neural reranking."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from app.retrieval.search import SearchHit

logger = logging.getLogger(__name__)

# Global cached neural reranker instance
_reranker_instance: Any = None
_reranker_model_name: str | None = None
_reranker_init_attempted: bool = False


@dataclass(frozen=True)
class RankedResult:
    """One retrieval result after threshold filtering and ranking."""

    chunk_id: uuid.UUID
    chunk_text: str
    document_id: uuid.UUID
    document_version_id: uuid.UUID
    document_title: str
    similarity_score: float
    rank: int
    section_title: str | None = None
    page_number: int | None = None
    metadata_: dict | None = None


def cosine_distance_to_similarity(distance: float) -> float:
    """Map pgvector cosine distance to cosine similarity."""
    return 1.0 - distance


def rank_results(hits: list[SearchHit], similarity_threshold: float) -> list[RankedResult]:
    """Filter by similarity threshold and assign 1-based ranks."""
    ranked: list[RankedResult] = []
    rank = 1

    logger.info("[RANKING EVALUATE] hits_in=%d threshold=%.4f", len(hits), similarity_threshold)
    for hit in hits:
        similarity = cosine_distance_to_similarity(hit.distance)
        if similarity < similarity_threshold:
            logger.info("  Dropped hit chunk_id=%s sim=%.4f < threshold=%.4f", hit.chunk_id, similarity, similarity_threshold)
            continue
        ranked.append(
            RankedResult(
                chunk_id=hit.chunk_id,
                chunk_text=hit.chunk_text,
                document_id=hit.document_id,
                document_version_id=hit.document_version_id,
                document_title=hit.document_title,
                similarity_score=similarity,
                rank=rank,
                section_title=hit.section_title,
                page_number=hit.page_number,
                metadata_=hit.metadata_,
            )
        )

        logger.info("  Accepted hit #%d chunk_id=%s sim=%.4f", rank, hit.chunk_id, similarity)
        rank += 1

    return ranked


def rank_hybrid_rrf(
    semantic_hits: list[SearchHit],
    fulltext_hits: list[SearchHit],
    similarity_threshold: float = 0.0,
    k: int = 60,
) -> list[RankedResult]:
    """Combine vector semantic hits and full-text keyword hits using Reciprocal Rank Fusion (RRF)."""
    scores: dict[uuid.UUID, float] = {}
    hit_map: dict[uuid.UUID, SearchHit] = {}

    for pos, hit in enumerate(semantic_hits, 1):
        hit_map[hit.chunk_id] = hit
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + (1.0 / (k + pos))

    for pos, hit in enumerate(fulltext_hits, 1):
        if hit.chunk_id not in hit_map:
            hit_map[hit.chunk_id] = hit
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + (1.0 / (k + pos))

    sorted_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    ranked: list[RankedResult] = []
    rank = 1

    for chunk_id in sorted_chunk_ids:
        hit = hit_map[chunk_id]
        sim = cosine_distance_to_similarity(hit.distance)
        if sim < similarity_threshold:
            continue
        ranked.append(
            RankedResult(
                chunk_id=hit.chunk_id,
                chunk_text=hit.chunk_text,
                document_id=hit.document_id,
                document_version_id=hit.document_version_id,
                document_title=hit.document_title,
                similarity_score=round(sim, 4),
                rank=rank,
                section_title=hit.section_title,
                page_number=hit.page_number,
                metadata_=hit.metadata_,
            )
        )
        rank += 1

    return ranked


def _get_neural_reranker() -> tuple[Any, str | None]:
    """Lazy-load and cache the neural Cross-Encoder reranker singleton."""
    global _reranker_instance, _reranker_model_name
    if _reranker_instance is not None:
        return _reranker_instance, _reranker_model_name

    import sys
    import subprocess
    import traceback

    errors = []

    # Attempt 1: Direct FlashRank import
    try:
        from flashrank import Ranker
        _reranker_instance = Ranker()
        _reranker_model_name = "FlashRank (ms-marco-TinyBERT-L-2-v2)"
        logger.info("[RERANKER INIT] Successfully loaded FlashRank neural model")
        return _reranker_instance, _reranker_model_name
    except Exception as exc:
        errors.append(f"FlashRank not available: {type(exc).__name__}: {exc}")

    # Attempt 2: Optional sentence-transformers fallback
    try:
        import importlib
        st_module = importlib.import_module("sentence_transformers")
        CrossEncoder = getattr(st_module, "CrossEncoder")
        st_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        _reranker_instance = CrossEncoder(st_model_name)
        _reranker_model_name = f"sentence-transformers CrossEncoder ({st_model_name})"
        logger.info("[RERANKER INIT] Successfully loaded sentence-transformers model %r", st_model_name)
        return _reranker_instance, _reranker_model_name
    except Exception as exc:
        errors.append(f"sentence-transformers error: {type(exc).__name__}: {exc}")

    return None, f"heuristic-fallback ({' | '.join(errors)})"


def rerank_cross_encoder(
    query: str,
    candidates: list[RankedResult],
    final_top_k: int = 5,
) -> list[RankedResult]:
    """Cross-encoder / semantic reranking of candidate chunks.

    Reranks top candidate chunks using joint query-passage cross-attention scoring.
    Preserves all chunk metadata and returns top `final_top_k` results.
    """
    if not candidates:
        return []

    reranker, model_name = _get_neural_reranker()

    logger.info(
        "[RERANKER] model=%s candidates_before=%d final_top_k=%d",
        model_name,
        len(candidates),
        final_top_k,
    )

    scored_candidates: list[tuple[float, RankedResult]] = []

    if reranker is not None:
        try:
            # Case 1: FlashRank implementation
            if "FlashRank" in (model_name or ""):
                from flashrank import RerankRequest

                passages = [
                    {
                        "id": idx,
                        "text": cand.chunk_text,
                        "meta": {"cand_idx": idx},
                    }
                    for idx, cand in enumerate(candidates)
                ]
                rerank_request = RerankRequest(query=query, passages=passages)
                results = reranker.rerank(rerank_request)

                for item in results:
                    score = float(item.get("score", 0.0))
                    meta_dict = item.get("meta", {})
                    cand_idx = meta_dict.get("cand_idx") if isinstance(meta_dict, dict) else None
                    if cand_idx is not None and 0 <= cand_idx < len(candidates):
                        cand = candidates[cand_idx]
                        scored_candidates.append((score, cand))

            # Case 2: sentence-transformers CrossEncoder implementation
            else:
                pairs = [[query, cand.chunk_text] for cand in candidates]
                scores = reranker.predict(pairs)
                for cand, score in zip(candidates, scores):
                    scored_candidates.append((float(score), cand))

            # Sort descending by neural cross-attention score
            scored_candidates.sort(key=lambda item: item[0], reverse=True)

        except Exception as exc:
            logger.warning(
                "[RERANKER FALLBACK] Neural inference error (%s). Falling back to heuristic scoring.",
                exc,
            )
            scored_candidates = _fallback_heuristic_rerank(query, candidates)
    else:
        logger.info("[RERANKER FALLBACK] Neural model unavailable. Using heuristic fallback scorer.")
        scored_candidates = _fallback_heuristic_rerank(query, candidates)

    reranked: list[RankedResult] = []
    for new_rank, (score, cand) in enumerate(scored_candidates[:final_top_k], 1):
        reranked.append(
            RankedResult(
                chunk_id=cand.chunk_id,
                chunk_text=cand.chunk_text,
                document_id=cand.document_id,
                document_version_id=cand.document_version_id,
                document_title=cand.document_title,
                similarity_score=round(score, 4),
                rank=new_rank,
                section_title=cand.section_title,
                page_number=cand.page_number,
                metadata_=cand.metadata_,
            )
        )

    logger.info(
        "[RERANKER] top_results=%s",
        [
            {
                "rank": r.rank,
                "chunk_id": str(r.chunk_id),
                "section": r.section_title,
                "score": r.similarity_score,
            }
            for r in reranked
        ],
    )

    return reranked


def _fallback_heuristic_rerank(
    query: str,
    candidates: list[RankedResult],
) -> list[tuple[float, RankedResult]]:
    """Explicit fallback scoring function when neural model is unavailable."""
    query_tokens = set(query.lower().split())
    scored_candidates: list[tuple[float, RankedResult]] = []

    for candidate in candidates:
        text = candidate.chunk_text.lower()
        title = (candidate.document_title or "").lower()
        section = (candidate.section_title or "").lower()

        matching_tokens = sum(1 for token in query_tokens if token in text or token in section or token in title)
        token_score = matching_tokens / max(len(query_tokens), 1)

        section_boost = 0.25 if any(t in section for t in query_tokens if len(t) > 3) else 0.0
        title_boost = 0.15 if any(t in title for t in query_tokens if len(t) > 3) else 0.0
        phrase_boost = 0.3 if query.lower().strip() in text else 0.0
        base_score = candidate.similarity_score

        composite_score = (base_score * 0.4) + (token_score * 0.35) + section_boost + title_boost + phrase_boost
        scored_candidates.append((composite_score, candidate))

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    return scored_candidates



