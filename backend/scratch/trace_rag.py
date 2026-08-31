import asyncio
import uuid
import sys
import os

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.embedding import Embedding
from app.models.user import User
from app.retrieval.retriever import Retriever
from app.retrieval.search import SearchFilters, search_similar, search_fulltext
from app.embeddings.client import OllamaEmbeddingClient
from sqlalchemy import select, func

async def trace():
    async with AsyncSessionLocal() as session:
        print("=== 1. CHECKING USERS AND DOCUMENTS IN DB ===")
        doc_stmt = select(Document.id, Document.title, Document.status, Document.user_id, Document.deleted_at)
        docs = (await session.execute(doc_stmt)).all()
        print(f"Total documents in DB: {len(docs)}")
        for d in docs:
            print(f"  Doc ID: {d.id} | Title: {d.title} | Status: {d.status} | User: {d.user_id} | Deleted: {d.deleted_at}")

        chunk_stmt = select(func.count(DocumentChunk.id))
        chunk_count = (await session.execute(chunk_stmt)).scalar()
        print(f"Total document chunks in DB: {chunk_count}")

        emb_stmt = select(func.count(Embedding.id))
        emb_count = (await session.execute(emb_stmt)).scalar()
        print(f"Total embeddings in DB: {emb_count}")

        if docs:
            sample_user_id = docs[0].user_id
            print(f"\nUsing sample user_id: {sample_user_id}")

            queries = [
                "Tell about working hours in Sipra Hub",
                "What is the attendance policy?",
                "Siprahub Working Hours & Attendance Policy",
                "What is the refund policy for space station tickets?"
            ]

            embedder = OllamaEmbeddingClient()

            for q in queries:
                print(f"\n==========================================")
                print(f"TRACE QUERY: {q!r}")
                print(f"==========================================")
                
                # Check vector embedding
                q_emb = await embedder.embed(q)
                print(f"Query embedding generated: len={len(q_emb)}")

                # Test 1: Unrestricted Semantic Search (no filters)
                sem_hits = await search_similar(session, q_emb, model_name=embedder.model, top_k=5, filters=SearchFilters())
                print(f"\n[Unrestricted Semantic Search] Hits count: {len(sem_hits)}")
                for idx, h in enumerate(sem_hits, 1):
                    print(f"  {idx}. [Distance: {h.distance:.4f} | Sim: {1 - h.distance:.4f}] Doc: {h.document_title} | Section: {h.section_title}\n     Text: {h.chunk_text[:120]!r}")

                # Test 2: Unrestricted Fulltext Search (no filters)
                ft_hits = await search_fulltext(session, q, top_k=5, filters=SearchFilters())
                print(f"\n[Unrestricted Fulltext Search] Hits count: {len(ft_hits)}")
                for idx, h in enumerate(ft_hits, 1):
                    print(f"  {idx}. [Rank Score: {h.distance:.4f}] Doc: {h.document_title} | Section: {h.section_title}\n     Text: {h.chunk_text[:120]!r}")

                # Test 3: Scoped Retriever with user_id
                retriever = Retriever(session)
                ret_results = await retriever.retrieve(q, filters=SearchFilters(user_id=sample_user_id), top_k=5, similarity_threshold=0.30)
                print(f"\n[Retriever (user_id={sample_user_id}, threshold=0.30)] Hits count: {len(ret_results)}")
                for idx, r in enumerate(ret_results, 1):
                    print(f"  {idx}. [Score: {r.similarity_score:.4f} | Rank: {r.rank}] Doc: {r.document_title} | Section: {r.section_title}\n     Text: {r.chunk_text[:120]!r}")

if __name__ == "__main__":
    asyncio.run(trace())
