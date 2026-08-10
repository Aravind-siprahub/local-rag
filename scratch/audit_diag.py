import asyncio
import sys
import json
from pathlib import Path
from sqlalchemy import select, func

# Ensure backend app is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.models.document_chunk import DocumentChunk
from app.models.embedding import Embedding
from app.retrieval.search import SearchFilters, search_similar, search_fulltext
from app.retrieval.ranking import rank_results, rank_hybrid_rrf, rerank_cross_encoder
from app.embeddings.client import OllamaEmbeddingClient
from app.prompting.builder import PromptBuilder
from app.llm.ollama_client import OllamaLLMClient
from app.core.config import get_settings

async def run_audit():
    async with AsyncSessionLocal() as session:
        print("==================================================")
        print("1. DATABASE CHUNKS AUDIT FOR PRD_Talk_to_My_Data.docx")
        print("==================================================")
        
        doc_stmt = select(Document).where(Document.title.ilike("%Talk_to_My_Data%"))
        docs = (await session.execute(doc_stmt)).scalars().all()
        if not docs:
            # try finding any document
            doc_stmt = select(Document)
            docs = (await session.execute(doc_stmt)).scalars().all()
        
        print(f"Found {len(docs)} documents matching query.")
        target_doc = None
        for d in docs:
            print(f"Document ID: {d.id} | Title: '{d.title}' | Status: {d.status} | User ID: {d.user_id}")
            if "PRD" in d.title or "Talk" in d.title:
                target_doc = d
        
        if not target_doc and docs:
            target_doc = docs[0]
            
        if target_doc:
            print(f"\nTarget Document selected: {target_doc.title} ({target_doc.id})")
            version_stmt = select(DocumentVersion).where(DocumentVersion.document_id == target_doc.id)
            versions = (await session.execute(version_stmt)).scalars().all()
            for v in versions:
                print(f"Version ID: {v.id} | Status: {v.status} | Filename: {v.original_filename}")
                chunk_stmt = select(DocumentChunk).where(DocumentChunk.document_version_id == v.id).order_by(DocumentChunk.chunk_index)
                chunks = (await session.execute(chunk_stmt)).scalars().all()
                print(f"Stored Chunks Count: {len(chunks)}")
                for c in chunks:
                    vec_count = (await session.execute(select(func.count(Embedding.id)).where(Embedding.chunk_id == c.id))).scalar_one()
                    print(f"\n--- CHUNK index={c.chunk_index} ---")
                    print(f"  Chunk ID: {c.id}")
                    print(f"  Page: {c.page_number}")
                    print(f"  Section: {c.section_title}")
                    print(f"  Length: {len(c.content)}")
                    print(f"  Tokens: {c.content_tokens}")
                    print(f"  Char Start: {c.char_start} | Char End: {c.char_end}")
                    print(f"  Metadata: {c.metadata_}")
                    print(f"  Vectors count: {vec_count}")
                    print(f"  First 500 chars:\n{c.content[:500]}")
                    print("  Prev/Next relationship: chunk_index", c.chunk_index - 1, "<-", c.chunk_index, "->", c.chunk_index + 1)
        
        print("\n==================================================")
        print("2. RETRIEVAL & RERANKING AUDIT FOR 'What is Talk to My Data?'")
        print("==================================================")
        
        question = "What is Talk to My Data?"
        settings = get_settings()
        
        emb_client = OllamaEmbeddingClient()
        query_embedding = await emb_client.embed(question)
        print(f"Query embedding length: {len(query_embedding)}")
        
        # Search un-scoped (filters=SearchFilters())
        filters = SearchFilters()
        
        sem_hits = await search_similar(session, query_embedding, model_name=settings.EMBEDDING_MODEL, top_k=settings.TOP_K, filters=filters)
        ft_hits = await search_fulltext(session, question, top_k=settings.TOP_K, filters=filters)
        
        print(f"\n--- VECTOR SEARCH HITS (top_k={settings.TOP_K}) ---")
        for rank, h in enumerate(sem_hits, 1):
            sim = 1.0 - h.distance
            print(f"Rank {rank}: Chunk ID={h.chunk_id} | Doc='{h.document_title}' | Page={h.page_number} | Sec='{h.section_title}' | Distance={h.distance:.4f} | Sim={sim:.4f}")
            print(f"  Text preview: {repr(h.chunk_text[:150])}")
            
        print(f"\n--- FULL-TEXT (FTS) SEARCH HITS (top_k={settings.TOP_K}) ---")
        for rank, h in enumerate(ft_hits, 1):
            score = 1.0 - h.distance
            print(f"Rank {rank}: Chunk ID={h.chunk_id} | Doc='{h.document_title}' | Page={h.page_number} | Sec='{h.section_title}' | Score={score:.4f}")
            print(f"  Text preview: {repr(h.chunk_text[:150])}")
            
        candidate_results = rank_hybrid_rrf(sem_hits, ft_hits, similarity_threshold=settings.SIMILARITY_THRESHOLD)[:settings.TOP_K]
        
        print(f"\n--- HYBRID / RRF CANDIDATES BEFORE RERANKING (Count: {len(candidate_results)}) ---")
        for r in candidate_results:
            print(f"Rank {r.rank}: Chunk ID={r.chunk_id} | Doc='{r.document_title}' | Page={r.page_number} | Sec='{r.section_title}' | RRF Score={r.similarity_score:.4f}")
            print(f"  First 500 chars:\n{r.chunk_text[:500]}\n")

        print("--- AFTER RERANKING (rerank_cross_encoder) ---")
        final_top_k = getattr(settings, "FINAL_CONTEXT", 5)
        reranked_results = rerank_cross_encoder(question, candidate_results, final_top_k=final_top_k)
        for r in reranked_results:
            print(f"Rank {r.rank}: Chunk ID={r.chunk_id} | Doc='{r.document_title}' | Page={r.page_number} | Sec='{r.section_title}' | Composite Score={r.similarity_score:.4f}")
            print(f"  First 500 chars:\n{r.chunk_text[:500]}\n")

        print("\n==================================================")
        print("3. PROMPT & LLM GENERATION TRACE")
        print("==================================================")
        pb = PromptBuilder()
        prompt = pb.build(question, reranked_results)
        print(f"Included Chunks in Context: {len(prompt.retrieved_chunks)}")
        print("\n--- USER PROMPT SENT TO OLLAMA ---")
        print(prompt.user_prompt)
        print("\n--- SYSTEM PROMPT ---")
        print(prompt.system_prompt)
        
        llm = OllamaLLMClient()
        resp = await llm.generate(prompt.system_prompt, prompt.user_prompt)
        print("\n--- OLLAMA ANSWER ---")
        print(resp.answer)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_audit())
