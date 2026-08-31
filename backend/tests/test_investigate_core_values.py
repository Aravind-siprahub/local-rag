import pytest
import asyncio
from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_version import DocumentVersion
from app.models.embedding import Embedding
from app.rag.query_normalizer import normalize_query
from app.rag.intent_router import classify
from app.rag.service import RAGService
from app.services.chat_session_service import ChatSessionService
from app.services.chat_message_service import ChatMessageService
from app.retrieval.retriever import Retriever
from app.retrieval.search import SearchFilters, search_similar, search_fulltext
from app.llm.sanitize import sanitize_response
from app.llm.factory import get_llm_client
from app.core.config import get_settings

@pytest.mark.asyncio
async def test_step1_and_step2_document_and_db_index():
    async with AsyncSessionLocal() as session:
        print("\n=== STEP 1 & STEP 2: VERIFY SOURCE DOCUMENTS & DB CHUNKS ===")
        doc_stmt = select(Document).where(Document.deleted_at.is_(None))
        docs = (await session.execute(doc_stmt)).scalars().all()
        print(f"\nFound {len(docs)} active documents in DB:")
        for d in docs:
            print(f" - Doc ID: {d.id} | Title: {d.title} | Status: {d.status} | User: {d.user_id} | CurrentVer: {d.current_version_id}")

        keywords = ["Core Values", "SipraHub", "Integrity", "Accountability", "Collaboration", "Excellence", "Respect"]
        
        for kw in keywords:
            chunk_stmt = (
                select(DocumentChunk, Document.title)
                .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
                .join(Document, DocumentVersion.document_id == Document.id)
                .where(DocumentChunk.content.ilike(f"%{kw}%"))
                .where(Document.deleted_at.is_(None))
            )
            res = (await session.execute(chunk_stmt)).all()
            print(f"\nKeyword: '{kw}' -> Found {len(res)} matching chunks:")
            for chunk, doc_title in res:
                print(f"  [MATCH] Chunk ID: {chunk.id} | Doc Title: '{doc_title}' | Page: {chunk.page_number} | Section: '{chunk.section_title}'")
                print(f"  Snippet: {repr(chunk.content[:120])}...")

        core_val_stmt = (
            select(DocumentChunk, Document.title, Document.id.label("doc_id"))
            .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
            .join(Document, DocumentVersion.document_id == Document.id)
            .where(DocumentChunk.content.ilike("%Integrity%") & DocumentChunk.content.ilike("%Accountability%"))
            .where(Document.deleted_at.is_(None))
        )
        core_chunks = (await session.execute(core_val_stmt)).all()
        print("\n--- EXACT CORE VALUES CHUNK SEARCH (Integrity & Accountability) ---")
        assert len(core_chunks) > 0, "Core Values chunk MUST exist in database!"
        for chunk, doc_title, doc_id in core_chunks:
            emb_stmt = select(Embedding).where(Embedding.chunk_id == chunk.id)
            emb = (await session.execute(emb_stmt)).scalars().first()
            dim = len(emb.embedding) if emb and emb.embedding is not None else 0
            emb_model = emb.model_name if emb else "N/A"
            
            print(f"\nChunk ID: {chunk.id}")
            print(f"Doc ID: {doc_id}")
            print(f"Doc Title: {doc_title}")
            print(f"Page: {chunk.page_number}")
            print(f"Section: {chunk.section_title}")
            print(f"Embedding Dimension: {dim} | Model: {emb_model}")
            print(f"Content:\n{chunk.content}\n")

@pytest.mark.asyncio
async def test_full_pipeline_step3_to_step16():
    async with AsyncSessionLocal() as session:
        print("\n============================================================")
        print("FULL PIPELINE TRACE & DIAGNOSTICS FOR POSSESSIVE / PHRASE QUERIES")
        print("============================================================\n")

        doc_stmt = select(Document).where(Document.deleted_at.is_(None))
        docs = (await session.execute(doc_stmt)).scalars().all()
        assert len(docs) > 0, "Must have active documents in DB"
        
        user_id = docs[0].user_id
        doc_titles = [d.title for d in docs]
        print(f"User ID: {user_id}")
        print(f"Uploaded Document Titles: {doc_titles}\n")

        queries = [
            "What are SipraHub's core values?",
            "What are SipraHub core values?",
            "What are SipraHub's values?",
            "core values of SipraHub",
            "What values does SipraHub have?",
            "What are Microsoft's core values?",
            "What are Acme's core values?",
            "What are the project's core values?",
            "What values does the company follow?",
        ]

        retriever = Retriever(session=session)
        llm_client = get_llm_client()
        settings = get_settings()

        for q in queries:
            print(f"\n================ QUERY: '{q}' ================")
            
            norm_res = normalize_query(q)
            print(f"[STEP 3/4] Original:   '{norm_res.original_query}'")
            print(f"[STEP 3/4] Normalized: '{norm_res.normalized_query}'")
            print(f"[STEP 3/4] Retrieval:  '{norm_res.retrieval_query}'")

            route = classify(q, document_titles=doc_titles)
            print(f"[STEP 13] Intent Route: {route}")

            msg_svc = ChatMessageService(session)
            sess_svc = ChatSessionService(session)
            rag_svc = RAGService(session, message_service=msg_svc, session_service=sess_svc)
            resolved_filters = await rag_svc._resolve_entity_filters(user_id, q, SearchFilters())
            print(f"[STEP 8] Entity Filter document_id: {resolved_filters.document_id}")
            print(f"[STEP 8] Entity Filter document_ids: {getattr(resolved_filters, 'document_ids', None)}")

            sq_emb = await retriever.client.embed(q)
            print(f"[STEP 5] Embedding Dim: {len(sq_emb)}")
            vec_hits = await search_similar(
                session,
                query_embedding=sq_emb,
                filters=SearchFilters(user_id=user_id),
                top_k=10,
                model_name=settings.EMBEDDING_MODEL,
            )
            print(f"[STEP 5] Vector Search Top Hits (count={len(vec_hits)}):")
            for rank, hit in enumerate(vec_hits[:5], 1):
                sim = round(1.0 - hit.distance, 4)
                print(f"   Rank {rank}: sim={sim:.4f} | doc='{hit.document_title}' | section='{hit.section_title}' | text={repr(hit.chunk_text[:80])}")

            fts_hits = await search_fulltext(
                session,
                query_text=q,
                filters=SearchFilters(user_id=user_id),
                top_k=10,
            )
            print(f"[STEP 6] Full Text Search Top Hits (count={len(fts_hits)}):")
            for rank, hit in enumerate(fts_hits[:5], 1):
                sim = round(1.0 - hit.distance, 4)
                print(f"   Rank {rank}: sim={sim:.4f} | doc='{hit.document_title}' | section='{hit.section_title}' | text={repr(hit.chunk_text[:80])}")

            final_ranked = await retriever.retrieve(q, filters=SearchFilters(user_id=user_id), top_k=5, similarity_threshold=0.15)
            print(f"[STEP 7/10] RRF Hybrid Retrieved Chunks (count={len(final_ranked)}):")
            for rank, r in enumerate(final_ranked, 1):
                print(f"   Rank {rank}: score={r.similarity_score} | doc='{r.document_title}' | section='{r.section_title}' | text={repr(r.chunk_text[:100])}")

            if final_ranked:
                prompt_bundle = rag_svc.prompt_builder.build(q, final_ranked)
                print(f"[STEP 11] Context Length: {len(prompt_bundle.user_prompt)} chars")
                print(f"[STEP 11] Context Snippet:\n{prompt_bundle.user_prompt[:250]}...\n")
                
                raw_llm_res = await llm_client.generate(prompt_bundle.system_prompt, prompt_bundle.user_prompt)
                raw_text = raw_llm_res.content if hasattr(raw_llm_res, "content") else str(raw_llm_res)
                sanitized_ans = sanitize_response(raw_text)
                print(f"[STEP 14] RAW LLM Response:\n{raw_text}")
                print(f"[STEP 14] SANITIZED LLM Response:\n{sanitized_ans}")
            else:
                print("[STEP 11] NO CHUNKS RETRIEVED! Context was empty!\n")
            
            print("-" * 70)
