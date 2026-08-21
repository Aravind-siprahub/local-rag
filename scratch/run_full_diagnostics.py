import asyncio
import sys
import os
import json
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.db.session import AsyncSessionLocal
from sqlalchemy import text
from app.rag.service import RAGService, _filter_relevant_chunks
from app.rag.query_normalizer import normalize_query
from app.rag.intent_router import classify
from app.rag.query_understanding import extract_query_intent
from app.rag.attribute_detector import detect_requested_attributes
from app.retrieval.search import SearchFilters
from app.retrieval.retriever import Retriever
from app.prompting.builder import PromptBuilder
from app.llm.sanitize import sanitize_response
from app.rag.verifier import verify_answer
from app.repositories.user_repository import UserRepository

test_questions = [
    "What process manager is used by SipraOne?",
    "What reverse proxy is used by SipraOne?",
    "What ports are used by SipraOne?",
    "What frontend and backend frameworks are used by SipraOne?",
    "What is the deployment environment used by SipraOne?",
    "What frontend and backend frameworks are used by Talk to My Data?",
    "What frontend and backend frameworks are used by AIRIS?",
]

async def run_diagnostics():
    async with AsyncSessionLocal() as session:
        user_repo = UserRepository(session)
        users = await user_repo.list_active(limit=10)
        user_id = users[0].id if users else None

        docs = (await session.execute(text("SELECT id, title, user_id FROM documents WHERE deleted_at IS NULL"))).fetchall()
        print(f"=== DATABASE DIAGNOSTICS ===")
        print(f"Active Users Count: {len(users)}")
        print(f"Active Documents Count: {len(docs)}")
        for d in docs:
            print(f"  Doc: id={d[0]} title={d[1]!r} user_id={d[2]}")
        print("="*60)

        rag_service = RAGService(session)
        retriever = Retriever(session)

        for idx, q in enumerate(test_questions, 1):
            print(f"\n=======================================================")
            print(f"DIAGNOSTIC TEST #{idx}: {q!r}")
            print(f"=======================================================")

            # Phase 6: Query Normalization
            orig_q, norm_q, ret_q = normalize_query(q)
            print(f"[PHASE 6: NORMALIZATION]")
            print(f"  Original Query:   {orig_q!r}")
            print(f"  Normalized Query: {norm_q!r}")
            print(f"  Retrieval Query:  {ret_q!r}")

            # Phase 5: Attribute Detection & Intent Router Classification
            attrs = detect_requested_attributes(q)
            intent = extract_query_intent(q)
            doc_titles = [d[1] for d in docs]
            route = classify(q, document_titles=doc_titles)
            print(f"\n[PHASE 5: ATTRIBUTE DETECTION & ROUTING]")
            print(f"  Detected Attributes: {[a.name for a in attrs]}")
            print(f"  Intent Category:     {intent.category}")
            print(f"  Intent Entity:       {intent.entity}")
            print(f"  Classified Route:    {route.value}")

            # Phase 1 & 2: Retrieval & Search Filters
            base_filters = SearchFilters(user_id=user_id)
            if user_id is not None:
                resolved_filters = await rag_service._resolve_entity_filters(user_id, q, base_filters)
            else:
                resolved_filters = base_filters
            print(f"\n[PHASE 1 & 2: RETRIEVAL & SEARCH FILTERS]")
            print(f"  Resolved Filters: {resolved_filters}")

            retrieved_candidates = await retriever.retrieve(ret_q or q, filters=resolved_filters)
            print(f"  Retrieved Candidates Count: {len(retrieved_candidates)}")
            for c_idx, cand in enumerate(retrieved_candidates, 1):
                print(f"    Candidate #{c_idx}:")
                print(f"      Doc Title:   {getattr(cand, 'document_title', '?')}")
                print(f"      Section:     {getattr(cand, 'section_title', '?')}")
                print(f"      Chunk ID:    {cand.chunk_id}")
                print(f"      Rerank Score:{cand.similarity_score:.4f}")
                print(f"      Snippet:     {repr(cand.chunk_text[:120])}")

            # Phase 3 & 4: Relevance Filtering
            print(f"\n[PHASE 3 & 4: RELEVANCE FILTERING - _filter_relevant_chunks]")
            filtered_chunks = _filter_relevant_chunks(q, retrieved_candidates)
            print(f"  Filtered Chunks Remaining Count: {len(filtered_chunks)}")
            for f_idx, cand in enumerate(filtered_chunks, 1):
                print(f"    Surviving Chunk #{f_idx}: {cand.chunk_id} [{getattr(cand, 'document_title', '?')}] score={cand.similarity_score:.4f}")

            # Phase 7 & 8: Fallback Condition & LLM Execution
            print(f"\n[PHASE 7 & 8: FALLBACK & END-TO-END EXECUTION]")
            if not filtered_chunks:
                print(f"  RESULT: Fallback triggered -> 'I could not find this information in the uploaded documents.'")
            else:
                top_chunk = filtered_chunks[0]
                print(f"  Top Surviving Chunk ID:    {top_chunk.chunk_id}")
                print(f"  Top Surviving Doc Title:   {getattr(top_chunk, 'document_title', '?')}")
                print(f"  Top Surviving Rerank Score:{top_chunk.similarity_score:.4f}")
                print(f"  Top Surviving Snippet:     {repr(top_chunk.chunk_text[:150])}")

                try:
                    builder = PromptBuilder()
                    prompt = builder.build(q, filtered_chunks)
                    from app.llm.ollama_client import get_global_ollama_client
                    llm = get_global_ollama_client()
                    resp = await llm.generate(prompt.system_prompt, prompt.user_prompt, num_predict=150)
                    clean_ans = sanitize_response(resp.answer, question=q)
                    v_res = verify_answer(clean_ans, filtered_chunks, intent)
                    print(f"  Raw LLM Output:       {resp.answer!r}")
                    print(f"  Sanitized Answer:     {clean_ans!r}")
                    print(f"  Verification Result:  is_valid={v_res.is_valid} reason={v_res.reason}")
                except Exception as exc:
                    print(f"  LLM Generation Error: {exc}")

if __name__ == "__main__":
    asyncio.run(run_diagnostics())
