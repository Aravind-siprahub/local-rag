import asyncio
import sys
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG_DIAGNOSTIC")

# Add backend to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.db.session import AsyncSessionLocal
from sqlalchemy import text
from app.rag.service import RAGService, _filter_relevant_chunks
from app.rag.query_normalizer import normalize_query
from app.rag.intent_router import classify
from app.rag.query_understanding import extract_query_intent
from app.retrieval.search import SearchFilters
from app.retrieval.retriever import Retriever
from app.retrieval.ranking import rerank_cross_encoder
from app.prompting.builder import PromptBuilder
from app.llm.sanitize import sanitize_response
from app.rag.verifier import verify_answer

async def run_diagnostics():
    question = "What frontend and backend frameworks are used by talk to my data"
    print(f"\n=======================================================")
    print(f"STEP 1: QUERY NORMALIZATION")
    print(f"=======================================================")
    orig_q, norm_q, ret_q = normalize_query(question)
    print(f"Original Q:  {orig_q!r}")
    print(f"Normalized Q: {norm_q!r}")
    print(f"Retrieval Q:  {ret_q!r}")

    print(f"\n=======================================================")
    print(f"STEP 2: QUERY INTENT & ROUTING")
    print(f"=======================================================")
    intent = extract_query_intent(question)
    print(f"Intent Entity:     {intent.entity}")
    print(f"Intent Category:   {intent.category}")
    print(f"Intent Attributes: {intent.attributes}")
    
    route = classify(question)
    print(f"Classified Route:  {route.value}")

    async with AsyncSessionLocal() as session:
        print(f"\n=======================================================")
        print(f"STEP 3: USERS & DOCUMENTS IN DATABASE")
        print(f"=======================================================")
        users = (await session.execute(text("SELECT id, email FROM users"))).fetchall()
        for u in users:
            print(f"User: id={u[0]} email={u[1]}")
            
        docs = (await session.execute(text("SELECT id, user_id, title, status FROM documents WHERE deleted_at IS NULL"))).fetchall()
        print(f"\nFound {len(docs)} active documents:")
        for d in docs:
            print(f"  Doc id={d[0]} user_id={d[1]} title={d[2]!r} status={d[3]}")
            
        if not users or not docs:
            print("ERROR: No users or documents found in DB!")
            return

        user_id = docs[0][1]
        
        print(f"\n=======================================================")
        print(f"STEP 4: ENTITY FILTER RESOLUTION")
        print(f"=======================================================")
        rag_service = RAGService(session)
        base_filters = SearchFilters(user_id=user_id)
        resolved_filters = await rag_service._resolve_entity_filters(user_id, question, base_filters)
        print(f"Base Filters:     {base_filters}")
        print(f"Resolved Filters: {resolved_filters}")

        print(f"\n=======================================================")
        print(f"STEP 5: RETRIEVAL & VECTOR SEARCH")
        print(f"=======================================================")
        retriever = Retriever(session)
        retrieved_chunks = await retriever.retrieve(ret_q or question, filters=resolved_filters)
        print(f"Retrieved Chunks Count: {len(retrieved_chunks)}")
        for idx, c in enumerate(retrieved_chunks, 1):
            print(f"\n--- Chunk #{idx} ---")
            print(f"  Doc Title: {getattr(c, 'document_title', '?')}")
            print(f"  Section:   {getattr(c, 'section_title', '?')}")
            print(f"  Sim Score: {c.similarity_score:.4f}")
            print(f"  Snippet:   {repr(c.chunk_text[:150])}")

        print(f"\n=======================================================")
        print(f"STEP 6: POST-RERANKING RELEVANCE FILTERING")
        print(f"=======================================================")
        filtered_chunks = _filter_relevant_chunks(ret_q or question, retrieved_chunks)
        print(f"Filtered Chunks Count: {len(filtered_chunks)}")
        for idx, c in enumerate(filtered_chunks, 1):
            print(f"  Filtered Chunk #{idx}: {getattr(c, 'document_title', '?')} | score={c.similarity_score:.4f} | snippet={repr(c.chunk_text[:100])}")

        if not filtered_chunks:
            print("\n🚨 CRITICAL FAILURE AT STEP 6: 0 chunks passed relevance filter!")
            return

        print(f"\n=======================================================")
        print(f"STEP 7: PROMPT BUILDING")
        print(f"=======================================================")
        builder = PromptBuilder()
        prompt = builder.build(question, filtered_chunks)
        print(f"SYSTEM PROMPT:\n{prompt.system_prompt}\n")
        print(f"USER PROMPT:\n{prompt.user_prompt}\n")

        print(f"\n=======================================================")
        print(f"STEP 8: LLM GENERATION & SANITIZATION & VERIFICATION")
        print(f"=======================================================")
        from app.llm.ollama_client import get_global_ollama_client
        llm = get_global_ollama_client()
        try:
            llm_resp = await llm.generate(prompt.system_prompt, prompt.user_prompt, num_predict=150)
            print(f"Raw LLM Answer:\n{llm_resp.answer!r}\n")
            
            clean_ans = sanitize_response(llm_resp.answer, question=question)
            print(f"Sanitized Answer:\n{clean_ans!r}\n")
            
            v_res = verify_answer(clean_ans, filtered_chunks, intent)
            print(f"Verification Result: is_valid={v_res.is_valid} reason={v_res.reason}")
        except Exception as e:
            print(f"LLM Call Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_diagnostics())
