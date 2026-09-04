"""End-to-end retrieval: embed question -> search -> rank."""
from __future__ import annotations

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.embeddings.client import EmbeddingClient, OllamaEmbeddingClient
from app.retrieval.ranking import RankedResult, rank_results, rank_hybrid_rrf, rerank_cross_encoder
from app.retrieval.search import SearchFilters, search_similar, search_fulltext

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Raised when retrieval input or configuration is invalid."""


class Retriever:
    """Retrieve relevant document chunks for a natural-language question.

    Embeds the question via the configured embedding client, runs pgvector
    cosine search, and returns ranked results. Independent of LLM chat
    generation and prompt building.
    """

    def __init__(
        self,
        session: AsyncSession | None = None,
        *,
        client: EmbeddingClient | None = None,
        model_name: str | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self.session = session
        self.client = client or OllamaEmbeddingClient()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.top_k = top_k if top_k is not None else settings.TOP_K
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else settings.SIMILARITY_THRESHOLD
        )

    async def retrieve(
        self,
        question: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> list[RankedResult]:
        """Embed a user question, retrieve Top 20 candidates via hybrid search, and rerank to Top 5."""
        import time
        import asyncio
        # ranking imports moved to module level

        if not question or not question.strip():
            raise RetrievalError("Question must not be empty.")

        settings = get_settings()
        candidate_top_k = top_k if top_k is not None else settings.TOP_K
        effective_threshold = (
            similarity_threshold if similarity_threshold is not None else self.similarity_threshold
        )
        final_top_k = getattr(settings, "FINAL_CONTEXT", 5)

        if candidate_top_k <= 0:
            raise RetrievalError("top_k must be greater than 0.")
        if not 0.0 <= effective_threshold <= 1.0:
            raise RetrievalError("similarity_threshold must be between 0.0 and 1.0.")

        start_time = time.monotonic()

        logger.info(
            "[RETRIEVER START] question=%r candidate_top_k=%d final_top_k=%d threshold=%.4f filters=%s",
            question.strip(),
            candidate_top_k,
            final_top_k,
            effective_threshold,
            filters,
        )

        filters_obj = filters or SearchFilters()
        mode = getattr(filters_obj, "search_mode", "hybrid")

        from app.rag.query_normalizer import normalize_query
        from app.rag.query_understanding import extract_query_intent, AttributeCategory, decompose_query_topics
        _, norm_q, ret_q = normalize_query(question.strip())
        intent = extract_query_intent(question.strip())
        decomposed_topics = decompose_query_topics(question.strip())

        # Generate sub-queries for multi-aspect questions and clean retrieval query
        sub_queries = [question.strip()]
        if ret_q and ret_q.strip() and ret_q.strip().lower() != question.strip().lower():
            sub_queries.append(ret_q.strip())
        if norm_q and norm_q.strip() and norm_q.strip().lower() not in (s.lower() for s in sub_queries):
            sub_queries.append(norm_q.strip())
        if intent.normalized_query and intent.normalized_query.strip().lower() not in (s.lower() for s in sub_queries):
            sub_queries.append(intent.normalized_query.strip())

        # Ensure multi-part questions have each decomposed sub-topic searched independently
        if len(decomposed_topics) > 1:
            for topic in decomposed_topics:
                t_clean = topic.strip()
                if t_clean and t_clean.lower() not in (s.lower() for s in sub_queries):
                    sub_queries.append(t_clean)

        # Clean topical sub-query without company names or conversational verbs
        _Q_STRIP_WORDS = {
            "siprahub", "airis", "what", "are", "is", "the", "in", "of", "to", "for",
            "tell", "me", "about", "give", "details", "explain", "please", "can", "you",
            "our", "my", "how", "do", "does", "we", "a", "an",
        }
        import re as re_mod
        clean_words = [w for w in re_mod.findall(r"\b[a-zA-Z0-9]+\b", question) if w.lower() not in _Q_STRIP_WORDS]
        if clean_words:
            clean_topical_q = " ".join(clean_words)
            if clean_topical_q.lower() not in (s.lower() for s in sub_queries):
                sub_queries.append(clean_topical_q)

        search_depth = top_k if top_k is not None else max(candidate_top_k, 30)
        candidate_pool_limit = top_k if top_k is not None else max(candidate_top_k, 35)

        embed_start = time.monotonic()
        embed_tasks = [self.client.embed(sq) for sq in sub_queries]
        embeddings = await asyncio.gather(*embed_tasks)
        embed_time_ms = int((time.monotonic() - embed_start) * 1000)

        retrieval_search_start = time.monotonic()

        if mode == "fulltext":
            clean_ft_hits = []
            seen_ft = set()
            for sq in sub_queries:
                ft_list = await search_fulltext(self.session, sq, top_k=search_depth, filters=filters_obj)
                for hit in ft_list:
                    if hit.chunk_id not in seen_ft:
                        seen_ft.add(hit.chunk_id)
                        clean_ft_hits.append(hit)
                            
            candidate_results = rank_results(clean_ft_hits, 0.0)[:candidate_pool_limit]
            hits = clean_ft_hits
        elif mode == "semantic":
            clean_sem_hits = []
            seen_sem = set()
            for emb in embeddings:
                sem_list = await search_similar(
                    self.session,
                    emb,
                    model_name=self.model_name,
                    top_k=search_depth,
                    filters=filters_obj,
                )
                for hit in sem_list:
                    if hit.chunk_id not in seen_sem:
                        seen_sem.add(hit.chunk_id)
                        clean_sem_hits.append(hit)
                            
            candidate_results = rank_results(clean_sem_hits, effective_threshold)[:candidate_pool_limit]
            if not candidate_results and clean_sem_hits:
                logger.info("[RETRIEVER THRESHOLD FALLBACK] 0 candidates at threshold=%.3f. Retrying at threshold=0.10", effective_threshold)
                candidate_results = rank_results(clean_sem_hits, 0.10)[:candidate_pool_limit]
            hits = clean_sem_hits
        else:
            clean_sem_hits = []
            seen_sem = set()
            for emb in embeddings:
                sem_list = await search_similar(
                    self.session,
                    emb,
                    model_name=self.model_name,
                    top_k=search_depth,
                    filters=filters_obj,
                )
                for hit in sem_list:
                    if hit.chunk_id not in seen_sem:
                        seen_sem.add(hit.chunk_id)
                        clean_sem_hits.append(hit)

            clean_ft_hits = []
            seen_ft = set()
            for sq in sub_queries:
                ft_list = await search_fulltext(self.session, sq, top_k=search_depth, filters=filters_obj)
                for hit in ft_list:
                    if hit.chunk_id not in seen_ft:
                        seen_ft.add(hit.chunk_id)
                        clean_ft_hits.append(hit)

            hits = clean_sem_hits + clean_ft_hits
            candidate_results = rank_hybrid_rrf(clean_sem_hits, clean_ft_hits, similarity_threshold=effective_threshold)[:candidate_pool_limit]
            if not candidate_results and (clean_sem_hits or clean_ft_hits):
                logger.info("[RETRIEVER THRESHOLD FALLBACK] 0 hybrid candidates at threshold=%.3f. Retrying at threshold=0.10", effective_threshold)
                candidate_results = rank_hybrid_rrf(clean_sem_hits, clean_ft_hits, similarity_threshold=0.10)[:candidate_pool_limit]

        retrieval_time_ms = int((time.monotonic() - retrieval_search_start) * 1000)

        # Content-based deduplication: prevent identical chunks across multiple versions from crowding candidates
        deduped_candidates: list[RankedResult] = []
        seen_cand_hashes: set[str] = set()
        for cand in candidate_results:
            c_norm = " ".join(cand.chunk_text.strip()[:100].lower().split())
            if c_norm not in seen_cand_hashes:
                seen_cand_hashes.add(c_norm)
                deduped_candidates.append(cand)
            if len(deduped_candidates) >= candidate_pool_limit:
                break
        candidate_results = deduped_candidates

        rerank_start = time.monotonic()
        rerank_q = intent.normalized_query if intent.category != AttributeCategory.GENERAL and intent.normalized_query else question.strip()
        results = rerank_cross_encoder(rerank_q, candidate_results, final_top_k=final_top_k)
        rerank_time_ms = int((time.monotonic() - rerank_start) * 1000)

        # Expand adjacent sibling chunks within the same document version (preserves complete policy details)
        results = await self._expand_chunk_windows(results, forward_window=5, backward_window=1)

        total_retrieval_ms = int((time.monotonic() - start_time) * 1000)

        logger.info(
            "[RETRIEVER METRICS] total_ms=%d embed_ms=%d search_ms=%d rerank_ms=%d candidates=%d selected=%d chunk_ids=%s",
            total_retrieval_ms,
            embed_time_ms,
            retrieval_time_ms,
            rerank_time_ms,
            len(candidate_results),
            len(results),
            [str(r.chunk_id) for r in results],
        )

        if not results:
            await self._log_empty_retrieval(
                filters=filters,
                hit_count=len(hits),
                top_k=candidate_top_k,
                threshold=effective_threshold,
            )

        return results

    async def _expand_chunk_windows(
        self,
        results: list[RankedResult],
        forward_window: int = 3,
        backward_window: int = 1,
    ) -> list[RankedResult]:
        """Expand retrieved chunks with adjacent context from the same document version.
        
        Fetches forward sibling chunks (preserving policy details) and backward context
        while respecting section boundaries and avoiding cross-section topic pollution.
        """
        if not results or not self.session:
            return results

        try:
            from sqlalchemy import select
            from app.models.document_chunk import DocumentChunk

            chunk_ids = [r.chunk_id for r in results]
            stmt = (
                select(DocumentChunk.id, DocumentChunk.document_version_id, DocumentChunk.chunk_index)
                .where(DocumentChunk.id.in_(chunk_ids))
            )
            rows = (await self.session.execute(stmt)).all()
            chunk_map = {row.id: (row.document_version_id, row.chunk_index) for row in rows}

            needed_by_version: dict[uuid.UUID, set[int]] = {}
            for cid in chunk_ids:
                if cid in chunk_map:
                    v_id, idx = chunk_map[cid]
                    if v_id not in needed_by_version:
                        needed_by_version[v_id] = set()
                    for offset in range(-backward_window, forward_window + 1):
                        target_idx = idx + offset
                        if target_idx >= 0:
                            needed_by_version[v_id].add(target_idx)

            fetched_chunks: dict[tuple[uuid.UUID, int], DocumentChunk] = {}
            for v_id, indices in needed_by_version.items():
                fetch_stmt = (
                    select(DocumentChunk)
                    .where(DocumentChunk.document_version_id == v_id)
                    .where(DocumentChunk.chunk_index.in_(list(indices)))
                    .order_by(DocumentChunk.chunk_index)
                )
                for c in (await self.session.execute(fetch_stmt)).scalars().all():
                    fetched_chunks[(v_id, c.chunk_index)] = c

            expanded_results: list[RankedResult] = []
            seen_emitted_indices: set[tuple[uuid.UUID, int]] = set()

            for r in results:
                if r.chunk_id not in chunk_map:
                    expanded_results.append(r)
                    continue

                v_id, idx = chunk_map[r.chunk_id]
                if (v_id, idx) in seen_emitted_indices:
                    continue

                base_chunk = fetched_chunks.get((v_id, idx))
                base_section = getattr(base_chunk, "section_title", None) or r.section_title

                base_root = (base_section or "").split("→")[0].strip().lower()

                stitched_parts: list[str] = []
                # Backward pass
                for offset in range(-backward_window, 0):
                    target_idx = idx + offset
                    key = (v_id, target_idx)
                    if key in fetched_chunks and key not in seen_emitted_indices:
                        adj = fetched_chunks[key]
                        adj_root = (adj.section_title or "").split("→")[0].strip().lower()
                        # Do not cross major section boundary backwards
                        if base_root and adj_root and adj_root != base_root:
                            continue
                        stitched_parts.append(adj.content.strip())
                        seen_emitted_indices.add(key)

                # Center chunk
                stitched_parts.append(r.chunk_text.strip())
                seen_emitted_indices.add((v_id, idx))

                # Forward pass (subsequent rules / continuation)
                for offset in range(1, forward_window + 1):
                    target_idx = idx + offset
                    key = (v_id, target_idx)
                    if key in fetched_chunks and key not in seen_emitted_indices:
                        adj = fetched_chunks[key]
                        adj_root = (adj.section_title or "").split("→")[0].strip().lower()
                        # Do not cross major section boundary forwards
                        if base_root and adj_root and adj_root != base_root:
                            break
                        stitched_parts.append(adj.content.strip())
                        seen_emitted_indices.add(key)

                if stitched_parts:
                    expanded_text = "\n\n".join(dict.fromkeys(stitched_parts))
                    expanded_results.append(
                        RankedResult(
                            chunk_id=r.chunk_id,
                            chunk_text=expanded_text,
                            document_id=r.document_id,
                            document_version_id=r.document_version_id,
                            document_title=r.document_title,
                            similarity_score=r.similarity_score,
                            rank=r.rank,
                            section_title=base_section,
                            page_number=r.page_number,
                            metadata_=r.metadata_,
                        )
                    )
                else:
                    expanded_results.append(r)

            return expanded_results
        except Exception as exc:
            logger.warning("[RETRIEVER WINDOW EXPANSION ERROR] %s. Returning raw results.", exc)
            return results

    async def retrieve_section_aware(
        self,
        question: str,
        *,
        filters: SearchFilters | None = None,
        max_total_chunks: int = 5,
    ) -> list[RankedResult]:
        """Section-aware document retrieval for document summary and detail queries.

        Retrieves representative chunks from major sections of the target document,
        strictly capped to prevent full-document context dumping.
        """
        from app.retrieval.search import search_document_chunks_structured, SearchHit

        hits = await search_document_chunks_structured(
            self.session,
            filters=filters,
            max_chunks=200,
        )

        if not hits:
            logger.info("[SECTION AWARE RETRIEVAL] 0 hits for filters=%s, falling back to standard retrieve", filters)
            return await self.retrieve(question, filters=filters)

        # Group hits by section_title (or page_number)
        sections: dict[str, list[SearchHit]] = {}
        for hit in hits:
            sec_name = (hit.section_title or "").strip() or f"Page {hit.page_number or 1}"
            sections.setdefault(sec_name, []).append(hit)

        selected_hits: list[SearchHit] = []
        if len(sections) == 1:
            # Single section or uncategorized: sample evenly across all chunks up to max_total_chunks
            all_hits = list(sections.values())[0]
            if len(all_hits) <= max_total_chunks:
                selected_hits = list(all_hits)
            else:
                step = len(all_hits) / max_total_chunks
                selected_hits = [all_hits[int(i * step)] for i in range(max_total_chunks)]
        else:
            # Multi-section: sample proportionally across sections
            per_section = max(2, max_total_chunks // len(sections))
            for sec_name, sec_hits in sections.items():
                selected_hits.extend(sec_hits[:per_section])
            if len(selected_hits) > max_total_chunks:
                selected_hits = selected_hits[:max_total_chunks]

        # Dedup and preserve document index order (chunk sequence order)
        seen_hit_ids: set[uuid.UUID] = set()
        deduped_selected: list[SearchHit] = []
        for hit in selected_hits:
            if hit.chunk_id not in seen_hit_ids:
                seen_hit_ids.add(hit.chunk_id)
                deduped_selected.append(hit)

        # Sort by chunk_index / document order
        deduped_selected.sort(key=lambda h: getattr(h, "chunk_index", 0) if hasattr(h, "chunk_index") else 0)

        ranked: list[RankedResult] = []
        for idx, hit in enumerate(deduped_selected[:max_total_chunks], 1):
            ranked.append(
                RankedResult(
                    chunk_id=hit.chunk_id,
                    chunk_text=hit.chunk_text,
                    document_id=hit.document_id,
                    similarity_score=1.0,
                    rank=idx,
                    document_version_id=hit.document_version_id,
                    document_title=hit.document_title,
                    section_title=hit.section_title,
                    page_number=hit.page_number,
                )
            )

        logger.info(
            "[SECTION AWARE METRICS]\nQUERY: %r\nINTENT: DOCUMENT_SUMMARY/DETAIL\nDOCUMENT_ID: %s\nTOTAL_RAW_CHUNKS: %d\nSECTIONS_FOUND: %d\nFINAL_CHUNKS_SELECTED: %d\nSECTIONS: %s",
            question,
            filters.document_id if filters else "N/A",
            len(hits),
            len(sections),
            len(ranked),
            list(sections.keys()),
        )

        return ranked



    async def _log_empty_retrieval(
        self,
        *,
        filters: SearchFilters | None,
        hit_count: int,
        top_k: int,
        threshold: float,
    ) -> None:
        """Explain zero results — common cause is user_id scoping past owned embeddings."""
        from sqlalchemy import func, select

        from app.models.document import Document
        from app.models.document_chunk import DocumentChunk
        from app.models.document_version import DocumentVersion
        from app.models.embedding import Embedding
        from app.models.enums import DocumentStatus

        filters = filters or SearchFilters()
        scoped_count: int | None = None
        global_count: int | None = None

        if self.session is None:
            return

        try:
            stmt = (
                select(func.count())
                .select_from(Embedding)
                .join(DocumentChunk, DocumentChunk.id == Embedding.chunk_id)
                .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
                .join(Document, DocumentVersion.document_id == Document.id)
                .where(Document.deleted_at.is_(None))
                .where(Document.status == DocumentStatus.READY)
                .where(
                    (Embedding.model_name == self.model_name)
                    | (Embedding.model_name.ilike(f"{self.model_name.split(':')[0]}%"))
                )
            )
            if filters.user_id is not None:
                stmt = stmt.where(Document.user_id == filters.user_id)
            if filters.document_id is not None:
                stmt = stmt.where(Document.id == filters.document_id)
            if filters.document_version_id is not None:
                stmt = stmt.where(DocumentVersion.id == filters.document_version_id)
            else:
                stmt = stmt.where((Document.current_version_id.is_(None)) | (DocumentChunk.document_version_id == Document.current_version_id))
            if filters.filename is not None:
                stmt = stmt.where(Document.title.ilike(f"%{filters.filename}%"))
            if filters.date_from is not None:
                stmt = stmt.where(Document.created_at >= filters.date_from)
            if filters.date_to is not None:
                stmt = stmt.where(Document.created_at <= filters.date_to)

            scoped_count = (await self.session.execute(stmt)).scalar_one()
            global_count = (
                await self.session.execute(
                    select(func.count())
                    .select_from(Embedding)
                    .where(
                        (Embedding.model_name == self.model_name)
                        | (Embedding.model_name.ilike(f"{self.model_name.split(':')[0]}%"))
                    )
                )
            ).scalar_one()
        except Exception as exc:
            logger.debug("Could not count embeddings for empty-retrieval diagnostics: %s", exc)

        logger.warning(
            "Retrieved 0 chunks (raw_hits=%d, top_k=%d, threshold=%.3f, filters=%s). "
            "Embeddings for model %r under these filters=%s; total embeddings for model=%s. "
            "If scoped count is 0 but global count > 0, the user_id/document filter is excluding all vectors.",
            hit_count,
            top_k,
            threshold,
            filters,
            self.model_name,
            scoped_count if scoped_count is not None else "unknown",
            global_count if global_count is not None else "unknown",
        )

    async def close(self) -> None:
        await self.client.close()
