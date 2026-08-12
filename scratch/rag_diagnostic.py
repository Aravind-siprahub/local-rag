"""
RAG Retrieval Diagnostic Script.
Run from the backend/ directory:
    python c:\\Users\\ARAVIND\\Desktop\\local-rag\\scratch\\rag_diagnostic.py
"""
from __future__ import annotations

import asyncio
import os
import sys

# Add the backend/ directory to sys.path AND chdir into it so
# pydantic_settings can find the .env file when loading Settings.
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)

TEST_QUERIES = [
    "What is the Problem Statement in Talk to My Data?",
    "What are the two technical problems?",
    "What frontend and backend are used?",
    "What is the tech stack?",
    "Explain the system architecture.",
    "Tell me about Nginx configuration.",
    "Tell me about SSL.",
]


async def run_diagnostics():
    from app.core.config import get_settings
    from app.db.session import AsyncSessionLocal
    from app.retrieval.retriever import Retriever
    from app.retrieval.search import SearchFilters

    settings = get_settings()

    print(f"\n{'='*80}")
    print(f"RAG Diagnostic")
    print(f"  model        = {settings.EMBEDDING_MODEL}")
    print(f"  top_k        = {settings.TOP_K}")
    print(f"  threshold    = {settings.SIMILARITY_THRESHOLD}")
    print(f"  FINAL_CONTEXT= {getattr(settings, 'FINAL_CONTEXT', 3)}")
    print(f"{'='*80}\n")

    async with AsyncSessionLocal() as session:
        retriever = Retriever(session)
        filters = SearchFilters()  # no user filter — check all indexed docs

        for query in TEST_QUERIES:
            print(f"\n{'─'*70}")
            print(f"QUERY: {query!r}")
            print(f"{'─'*70}")

            try:
                results = await retriever.retrieve(query, filters=filters)
                if not results:
                    print("  ❌ ZERO chunks returned (check embeddings and threshold)")
                    continue

                print(f"  ✅ {len(results)} chunk(s) returned after reranking\n")
                for r in results:
                    print(f"  Rank #{r.rank}")
                    print(f"    chunk_id       : {r.chunk_id}")
                    print(f"    document_title : {r.document_title}")
                    print(f"    section_title  : {r.section_title}")
                    print(f"    score (reranker): {r.similarity_score:.4f}")
                    print(f"    chunk text len : {len(r.chunk_text)} chars")
                    print(f"    chunk preview  : {r.chunk_text[:200]!r}")
                    if r.similarity_score < settings.SIMILARITY_THRESHOLD:
                        print(f"    ⚠️  OLD BUG: score {r.similarity_score:.4f} < threshold {settings.SIMILARITY_THRESHOLD} "
                              f"— would have been dropped before fix!")
                    print()

            except Exception as exc:
                import traceback
                print(f"  ❌ ERROR: {exc}")
                traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_diagnostics())
