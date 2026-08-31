"""Phase 1 - 16 End-to-End RAG Investigation & Diagnostic Test Suite."""
from __future__ import annotations

import logging
import uuid
import pytest

from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.rag.query_normalizer import normalize_query
from app.retrieval.retriever import Retriever
from app.retrieval.search import search_similar, search_fulltext, SearchFilters
from app.retrieval.ranking import rank_hybrid_rrf, rerank_cross_encoder
from app.prompting.builder import PromptBuilder
from app.llm.sanitize import sanitize_response

logger = logging.getLogger(__name__)

TARGET_QUERIES = [
    ("What is the purpose of the HR & Compliance Framework?", ["policies", "clarity", "consistency", "roles"]),
    ("What are SipraHub's core values?", ["Integrity", "Accountability", "Collaboration", "Excellence", "Respect"]),
    ("What are the standard working hours at SipraHub?", ["9:30", "6:30"]),
    ("How many hours must employees complete per day?", ["9", "8", "productive"]),
    ("How much Casual Leave is provided?", ["1", "month"]),
    ("How long does an employee normally have to file a POSH complaint?", ["3 months"]),
    ("What is the notice period for resignation?", ["agreement"]),
    ("What happens to unused Casual Leave at year-end?", ["lapses"]),
]

@pytest.mark.asyncio
async def test_run_full_investigation_suite():
    """Run full diagnostic audit across Database, Retrieval, Context, Prompt, and Sanitizer."""
    print("\n" + "=" * 80)
    print("PHASE 2 & 5: DATABASE CONTENT & DOCUMENT SCOPE AUDIT")
    print("=" * 80)

    async with AsyncSessionLocal() as session:
        # Check indexed documents
        stmt = select(Document)
        docs = (await session.execute(stmt)).scalars().all()
        print(f"Total Indexed Documents: {len(docs)}")
        for d in docs:
            print(f"  [DOC] ID: {d.id} | Title: {d.title} | Status: {d.status} | UserID: {d.user_id}")

        # Check for HR / Core Values chunks in DB
        hr_chunks_stmt = select(DocumentChunk).where(
            (DocumentChunk.content.ilike("%core values%")) |
            (DocumentChunk.content.ilike("%compliance framework%")) |
            (DocumentChunk.content.ilike("%working hours%"))
        )
        hr_chunks = (await session.execute(hr_chunks_stmt)).scalars().all()
        print(f"\nHR / Core Values Database Chunks found: {len(hr_chunks)}")
        for c in hr_chunks:
            print(f"  [CHUNK] ID: {c.id} | DocVerID: {c.document_version_id} | Page: {c.page_number} | Section: {c.section_title}")
            print(f"  Snippet: {c.content[:180]}...\n")

        retriever = Retriever(session)
        prompt_builder = PromptBuilder()

        for question, expected_keywords in TARGET_QUERIES:
            print("\n" + "-" * 80)
            print(f"QUERY AUDIT: {question}")
            print("-" * 80)

            # 1. Normalization
            orig_q, norm_q, ret_q = normalize_query(question)
            print(f"1. Query Normalization:\n   Original:   {orig_q}\n   Normalized: {norm_q}\n   Retrieval:  {ret_q}")

            # 2. Vector & FTS Standalone Search
            emb = await retriever.client.embed(ret_q or question)
            vector_hits = await search_similar(session, emb, model_name=retriever.model_name, top_k=10)
            fts_hits = await search_fulltext(session, ret_q or question, top_k=10)

            print(f"\n2. Standalone Retrieval Hits:\n   Vector Hits: {len(vector_hits)} | FTS Hits: {len(fts_hits)}")
            for rank, hit in enumerate(vector_hits[:5], 1):
                sim = 1.0 - hit.distance
                print(f"   [VECTOR #{rank}] Sim: {sim:.4f} | Chunk: {hit.chunk_id} | Page: {hit.page_number} | Text: {hit.chunk_text[:120]}...")
            for rank, hit in enumerate(fts_hits[:5], 1):
                print(f"   [FTS #{rank}] Score: {hit.distance:.4f} | Chunk: {hit.chunk_id} | Page: {hit.page_number} | Text: {hit.chunk_text[:120]}...")

            # 3. RRF & Cross-Encoder Reranking
            rrf_candidates = rank_hybrid_rrf(vector_hits, fts_hits, similarity_threshold=0.0)[:10]
            final_selected = rerank_cross_encoder(question, rrf_candidates, final_top_k=5)

            print(f"\n3. RRF & Reranked Final Selection ({len(final_selected)} chunks):")
            for r in final_selected:
                print(f"   [SELECTED #{r.rank}] Score: {r.similarity_score:.4f} | Chunk: {r.chunk_id} | Doc: {r.document_title} | Page: {r.page_number} | Section: {r.section_title}")
                print(f"   Text: {r.chunk_text[:150]}...\n")

            # 4. Context Construction
            prompt = prompt_builder.build(question, final_selected)
            print(f"4. Assembled LLM Context Length: {len(prompt.user_prompt)} chars")

            # Check if expected key phrases exist in assembled prompt
            missing = [kw for kw in expected_keywords if kw.lower() not in prompt.user_prompt.lower()]
            if missing:
                print(f"   ⚠️ WARNING: Expected keywords {missing} are MISSING from assembled LLM context!")
            else:
                print("   ✅ SUCCESS: All expected keywords present in LLM context.")
