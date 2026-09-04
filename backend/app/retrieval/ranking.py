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
    similarity_score: float
    rank: int
    document_version_id: uuid.UUID | None = None
    document_title: str = ""
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
    semantic_ids: set[uuid.UUID] = {hit.chunk_id for hit in semantic_hits}

    for pos, hit in enumerate(semantic_hits, 1):
        hit_map[hit.chunk_id] = hit
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + (1.0 / (k + pos))

    for pos, hit in enumerate(fulltext_hits, 1):
        if hit.chunk_id not in hit_map:
            hit_map[hit.chunk_id] = hit
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + (1.0 / (k + pos))

    fulltext_ids: set[uuid.UUID] = {hit.chunk_id for hit in fulltext_hits}
    sorted_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    ranked: list[RankedResult] = []
    rank = 1
    for chunk_id in sorted_chunk_ids:
        hit = hit_map[chunk_id]
        sim = cosine_distance_to_similarity(hit.distance)
        if chunk_id in semantic_ids and chunk_id not in fulltext_ids and sim < similarity_threshold:
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

    # Attempt 1: sentence-transformers (Primary)
    try:
        import importlib
        st_module = importlib.import_module("sentence_transformers")
        CrossEncoder = getattr(st_module, "CrossEncoder")
        st_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        _reranker_instance = CrossEncoder(st_model_name)
        _reranker_model_name = f"sentence-transformers CrossEncoder ({st_model_name})"
        logger.info("[RERANKER INIT] Successfully loaded sentence-transformers model %r", st_model_name)
        return _reranker_instance, _reranker_model_name
    except Exception as exc:
        errors.append(f"sentence-transformers error: {type(exc).__name__}: {exc}")

    # Attempt 2: FlashRank (Fallback)
    try:
        from flashrank import Ranker
        _reranker_instance = Ranker()
        _reranker_model_name = "FlashRank (ms-marco-TinyBERT-L-2-v2)"
        logger.info("[RERANKER INIT] Successfully loaded FlashRank neural model")
        return _reranker_instance, _reranker_model_name
    except Exception as exc:
        errors.append(f"FlashRank not available: {type(exc).__name__}: {exc}")

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

    from app.rag.query_understanding import extract_query_intent, AttributeCategory
    from app.rag.query_normalizer import normalize_query
    _, norm_q, ret_q = normalize_query(query.strip())
    intent = extract_query_intent(query)
    effective_query = intent.normalized_query if intent.normalized_query and intent.category != AttributeCategory.GENERAL else (norm_q or query)

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
                rerank_request = RerankRequest(query=effective_query, passages=passages)
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
                pairs = [[effective_query, cand.chunk_text] for cand in candidates]
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

    if scored_candidates:
        # Tech stack & architecture boosting for framework/technology queries
        query_lower = f"{query} {norm_q or ''}".lower()
        is_tech_query = any(kw in query_lower for kw in ("frontend", "backend", "tech stack", "technologies", "framework", "frameworks", "components", "architecture"))
        if is_tech_query:
            tech_keywords = {"fastapi", "react", "vue", "angular", "node", "python", "ollama", "qdrant", "postgresql", "duckdb", "vanna", "minio", "langfuse", "keycloak"}
            adjusted = []
            for score, cand in scored_candidates:
                text_lower = cand.chunk_text.lower()
                matching_techs = sum(1 for kw in tech_keywords if kw in text_lower)
                if matching_techs > 0:
                    score = score * (1.0 + 0.4 * matching_techs) + 0.12 * matching_techs
                adjusted.append((score, cand))
            scored_candidates = sorted(adjusted, key=lambda item: item[0], reverse=True)

        # Source prioritization for HR Policy queries: prefer HR Framework/Policy docs over PRD schema definitions
        hr_categories = {
            AttributeCategory.POLICY_WFH,
            AttributeCategory.POLICY_LEAVE,
            AttributeCategory.POLICY_POSH,
            AttributeCategory.POLICY_GRIEVANCE,
            AttributeCategory.POLICY_PERFORMANCE,
            AttributeCategory.POLICY_EXIT,
            AttributeCategory.POLICY_IT_SECURITY,
            AttributeCategory.POLICY_GENERAL,
            AttributeCategory.CULTURE_VALUES,
        }
        if intent.category in hr_categories:
            has_hr_docs = any(
                any(kw in (cand.document_title or "").lower() for kw in ("hr", "framework", "policy", "handbook"))
                for _, cand in scored_candidates
            )
            if has_hr_docs:
                adjusted = []
                for score, cand in scored_candidates:
                    title = (cand.document_title or "").lower()
                    if any(kw in title for kw in ("prd", "schema", "architecture", "spec", "v1.1", "v2.2")):
                        adjusted.append((score * 0.25, cand))
                    else:
                        adjusted.append((score, cand))
                scored_candidates = sorted(adjusted, key=lambda item: item[0], reverse=True)

        # Culture & Core Values boosting (prevents small cross-encoders from dropping bullet lists)
        is_culture_query = intent.category == AttributeCategory.CULTURE_VALUES or any(
            kw in query_lower for kw in ("core value", "core values", "company values", "our values", "culture")
        )
        if is_culture_query:
            adjusted = []
            for score, cand in scored_candidates:
                sec_lower = (cand.section_title or "").lower()
                text_lower = cand.chunk_text.lower()
                if "core value" in sec_lower or "core values" in sec_lower or "our values" in sec_lower:
                    score += 0.85
                elif any(kw in sec_lower for kw in ("values", "conduct", "culture")):
                    score += 0.40
                if any(kw in text_lower for kw in ("integrity –", "integrity -", "accountability –", "accountability -", "collaboration –", "excellence –", "respect –", "core values")):
                    score += 0.50
                adjusted.append((score, cand))
            scored_candidates = sorted(adjusted, key=lambda item: item[0], reverse=True)

        # Leave Policy boosting
        is_leave_query = intent.category == AttributeCategory.POLICY_LEAVE or any(
            kw in query_lower for kw in ("leave policy", "leave rules", "casual leave", "leave entitlement")
        )
        if is_leave_query:
            adjusted = []
            for score, cand in scored_candidates:
                sec_lower = (cand.section_title or "").lower()
                text_lower = cand.chunk_text.lower()
                if "leave policy" in sec_lower or "casual leave" in sec_lower or "leave application" in sec_lower:
                    score += 0.60
                elif "leave" in sec_lower:
                    score += 0.35
                if any(kw in text_lower for kw in ("casual leave", "leave entitlement", "carry forward", "leave utilization")):
                    score += 0.40
                adjusted.append((score, cand))
            scored_candidates = sorted(adjusted, key=lambda item: item[0], reverse=True)

        # General section title alignment boost
        raw_tokens = [t for t in query.lower().split() if t not in _STOP_WORDS and len(t) > 2]
        if raw_tokens:
            adjusted = []
            for score, cand in scored_candidates:
                sec_lower = (cand.section_title or "").lower()
                matching_sec_tokens = sum(1 for t in raw_tokens if t in sec_lower)
                if matching_sec_tokens >= 2:
                    score += 0.35
                elif matching_sec_tokens == 1:
                    score += 0.15
                adjusted.append((score, cand))
            scored_candidates = sorted(adjusted, key=lambda item: item[0], reverse=True)

        top_score = scored_candidates[0][0]
        if top_score >= 0.25:
            # Filter out near-zero/distractor chunks when top chunk has high relevance
            relevance_floor = max(0.01 if is_tech_query else 0.04, top_score * (0.04 if is_tech_query else 0.08))

            def _is_protected(c: RankedResult) -> bool:
                sec = (c.section_title or "").lower()
                if is_culture_query and any(k in sec for k in ("value", "conduct", "culture")):
                    return True
                if is_leave_query and "leave" in sec:
                    return True
                return False

            scored_candidates = [
                item for item in scored_candidates
                if item[0] >= relevance_floor or _is_protected(item[1])
            ]

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


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "in", "is", "it", "its",
    "my", "of", "on", "or", "should", "that", "the", "this", "to", "what", "which",
    "will", "with", "would", "you", "your", "using", "use", "used",
    "tell", "about", "explain", "give", "show", "details", "information", "info",
    "please", "want", "need", "find", "know", "get", "me", "us", "how",
}


def _fallback_heuristic_rerank(
    query: str,
    candidates: list[RankedResult],
) -> list[tuple[float, RankedResult]]:
    """Explicit fallback scoring function when neural model is unavailable."""
    raw_tokens = query.lower().split()
    content_tokens = set(t for t in raw_tokens if t not in _STOP_WORDS and len(t) > 1)
    if not content_tokens:
        content_tokens = set(raw_tokens)

    scored_candidates: list[tuple[float, RankedResult]] = []

    for candidate in candidates:
        text = candidate.chunk_text.lower()
        title = (candidate.document_title or "").lower()
        section = (candidate.section_title or "").lower()

        # Count match on key content tokens (excluding stop-words and generic corpus words)
        # Supports compound word matching (e.g. "sipra" + "hub" -> "siprahub")
        combined_text = f"{text} {title} {section}"
        compact_combined = combined_text.replace(" ", "")
        matching_tokens = 0
        for token in content_tokens:
            if token in combined_text or token in compact_combined:
                matching_tokens += 1

        token_score = matching_tokens / max(len(content_tokens), 1)


        # Attribute-specific ranking scoring boost/penalty via QueryIntent
        from app.rag.query_understanding import extract_query_intent, AttributeCategory
        intent = extract_query_intent(query)

        attr_boost = 0.0
        if intent.category == AttributeCategory.TECHNOLOGY:
            has_framework = any(tech in text for tech in ("react", "fastapi", "vite", "express", "next.js", "node.js", "nodejs", "python", "postgres", "vue", "angular", "django"))
            has_pure_port = any(port_kw in text for port_kw in ("port 4173", "port 5000", "port 8000", "port 8001", "pm2", "nginx")) and not has_framework
            if has_framework:
                attr_boost = 0.50
            elif has_pure_port:
                attr_boost = -0.35
        elif intent.category == AttributeCategory.CONFIGURATION:
            query_low = query.lower()
            is_llm_config_query = any(k in query_low for k in ("provider", "model", "omniroute", "llm", "configured", "configuration", "rag pipeline", "pipeline"))
            has_llm_config_keys = any(k in text for k in ("llm_provider", "omniroute_model", "ollama_model", "openrouter_model", "nvidia_model", "openai_model", "omniroute/auto", "omniroute", "provider=", "model="))
            has_port = any(p in text for p in ("port", "4173", "5000", "8000", "8001", "80", "443", "listening"))
            if is_llm_config_query and has_llm_config_keys:
                attr_boost = 0.85
            elif has_port:
                attr_boost = 0.50
        elif intent.category == AttributeCategory.POLICY_LEAVE:
            has_leave_rules = any(l_kw in text for l_kw in ("casual leave", "leave entitlement", "carry forward", "leave utilization", "leave benefits", "working hours", "work-life balance", "probation leave", "leave policy", "leave policies"))
            has_tech_spec = any(tech in text for tech in ("jira", "postgresql", "fastapi", "react", "port 8000", "endpoint", "/api/leave-policies", "mvp modules", "user journeys")) and not has_leave_rules
            if has_leave_rules:
                attr_boost = 0.70
            elif has_tech_spec:
                attr_boost = -0.30
            else:
                attr_boost = 0.0
        elif intent.category == AttributeCategory.CULTURE_VALUES:
            has_culture = any(c_kw in text for c_kw in ("culture", "integrity", "accountability", "professionalism", "ethical", "ethics", "code of conduct", "standards of behavior", "respect", "dignity", "values"))
            has_tech_spec = any(tech in text for tech in ("jira", "postgresql", "fastapi", "react", "port 8000", "role storage", "mvp modules", "user journeys")) and not has_culture
            if has_culture:
                attr_boost = 0.65
            elif has_tech_spec:
                attr_boost = -0.30
            else:
                attr_boost = 0.0

        query_low = query.lower()
        section_boost = 0.15 if any(t in section for t in content_tokens if len(t) > 3) else 0.0
        title_boost = 0.10 if any(t in title for t in content_tokens if len(t) > 3) else 0.0
        phrase_boost = 0.25 if query_low.strip() in text else 0.0
        base_score = candidate.similarity_score

        composite_score = (base_score * 0.25) + (token_score * 0.35) + attr_boost + section_boost + title_boost + phrase_boost
        scored_candidates.append((composite_score, candidate))

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    return scored_candidates



