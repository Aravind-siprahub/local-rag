import sys
import os
import asyncio

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath("backend"))

from app.rag.query_normalizer import normalize_query
from app.rag.intent_router import classify, Route
from app.retrieval.search import SearchFilters, search_similar, search_fulltext
from app.retrieval.ranking import rank_results, rank_hybrid_rrf, rerank_cross_encoder
from app.embeddings.client import OllamaEmbeddingClient
from app.core.config import get_settings
from app.db.session import AsyncSessionLocal

async def trace_query(query: str):
    print("=" * 80)
    print(f"TRACING QUERY: {query!r}")
    print("=" * 80)

    # 1. Normalization
    raw, norm, ret_q = normalize_query(query)
    print(f"[1. NORMALIZATION] raw={raw!r} | norm={norm!r} | ret_q={ret_q!r}")

    # 2. Intent Routing
    route = classify(query)
    print(f"[2. INTENT ROUTING] route={route.name} ({route.value})")

    # 3. Retrieval
    async with AsyncSessionLocal() as session:
        settings = get_settings()
        filters = SearchFilters()
        client = OllamaEmbeddingClient()
        
        print(f"[SETTINGS] TOP_K={settings.TOP_K} | SIMILARITY_THRESHOLD={settings.SIMILARITY_THRESHOLD} | FINAL_CONTEXT={getattr(settings, 'FINAL_CONTEXT', 5)}")

        # Embed query
        emb = await client.embed(query)
        print(f"[EMBEDDING] Generated vector len={len(emb)}")

        # Dense Search
        dense_hits = await search_similar(session, emb, model_name=settings.EMBEDDING_MODEL, top_k=settings.TOP_K, filters=filters)
        print(f"\n[3a. DENSE SEARCH] Count: {len(dense_hits)}")
        for idx, h in enumerate(dense_hits[:10], 1):
            print(f"  {idx}. doc_id={h.document_id} title={getattr(h, 'document_title', '?')!r} score={(1.0 - h.distance):.4f} text={h.chunk_text[:80]!r}")

        # FTS Search
        fts_hits = await search_fulltext(session, query, top_k=settings.TOP_K, filters=filters)
        print(f"\n[3b. FTS SEARCH] Count: {len(fts_hits)}")
        for idx, h in enumerate(fts_hits[:10], 1):
            print(f"  {idx}. doc_id={h.document_id} title={getattr(h, 'document_title', '?')!r} score={(1.0 - h.distance):.4f} text={h.chunk_text[:80]!r}")

        # RRF Ranking
        rrf_candidates = rank_hybrid_rrf(dense_hits, fts_hits, similarity_threshold=settings.SIMILARITY_THRESHOLD)[:settings.TOP_K]
        print(f"\n[4. RRF HYBRID RANKING] Candidates Count: {len(rrf_candidates)}")
        for idx, r in enumerate(rrf_candidates[:10], 1):
            print(f"  {idx}. doc_id={r.document_id} title={getattr(r, 'document_title', '?')!r} rrf_score={r.similarity_score:.4f} text={r.chunk_text[:80]!r}")

        # Reranker
        reranked = rerank_cross_encoder(query, rrf_candidates, final_top_k=getattr(settings, "FINAL_CONTEXT", 5))
        print(f"\n[5. RERANKER RESULTS] Selected Count: {len(reranked)}")
        for idx, r in enumerate(reranked, 1):
            print(f"  {idx}. doc_id={r.document_id} title={getattr(r, 'document_title', '?')!r} score={r.similarity_score:.4f} text={r.chunk_text[:80]!r}")

if __name__ == "__main__":
    asyncio.run(trace_query("what backend and frontend use in talk to my data"))
