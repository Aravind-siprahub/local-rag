"""RAG Debug & Diagnostics Endpoint.

Provides full visibility into database document counts, chunk counts, vector
embeddings, model config, and test retrieval / prompt construction.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_version import DocumentVersion
from app.models.embedding import Embedding
from app.models.enums import DocumentStatus, DocumentVersionStatus
from app.prompting.builder import PromptBuilder
from app.retrieval.retriever import Retriever
from app.retrieval.search import SearchFilters

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debug", tags=["Debug"])


@router.get("/rag", summary="Debug RAG pipeline state and retrieval")
async def debug_rag(
    q: str = Query(default="hi", description="Sample question to test retrieval against vector store"),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Diagnostic endpoint to inspect RAG pipeline: documents, chunks, vectors, retrieval hits, and prompt construction."""
    settings = get_settings()

    # 1. Count active documents
    doc_stmt = select(func.count(Document.id)).where(Document.deleted_at.is_(None))
    doc_count = (await session.execute(doc_stmt)).scalar_one()

    # 2. Count parsed document versions
    parsed_stmt = select(func.count(DocumentVersion.id)).where(
        DocumentVersion.status.in_([DocumentVersionStatus.PARSED, DocumentVersionStatus.CHUNKED, DocumentVersionStatus.EMBEDDED, DocumentVersionStatus.COMPLETED])
    )
    parsed_count = (await session.execute(parsed_stmt)).scalar_one()

    # 3. Count total document chunks
    chunk_stmt = select(func.count(DocumentChunk.id))
    chunk_count = (await session.execute(chunk_stmt)).scalar_one()

    # 4. Count total embeddings / vectors stored
    emb_stmt = select(func.count(Embedding.id))
    emb_count = (await session.execute(emb_stmt)).scalar_one()

    # 5. Run test retrieval for `q` without user scoping to check raw vector DB matches
    retrieved_count = 0
    top_k_results = []
    prompt_preview = ""

    try:
        retriever = Retriever(session)
        # Search across all documents (no user_id filter) to see if vectors match
        ranked_hits = await retriever.retrieve(q, filters=SearchFilters(), top_k=settings.TOP_K)
        retrieved_count = len(ranked_hits)

        for hit in ranked_hits:
            top_k_results.append(
                {
                    "rank": hit.rank,
                    "chunk_id": str(hit.chunk_id),
                    "document_id": str(hit.document_id),
                    "document_version_id": str(hit.document_version_id),
                    "similarity_score": round(hit.similarity_score, 4),
                    "chunk_text_preview": hit.chunk_text[:120],
                }
            )

        prompt_builder = PromptBuilder()
        built_prompt = prompt_builder.build(q, ranked_hits)
        prompt_preview = f"--- SYSTEM PROMPT ---\n{built_prompt.system_prompt}\n\n--- USER PROMPT ---\n{built_prompt.user_prompt}"

    except Exception as exc:
        logger.warning("Error running debug retrieval for query %r: %s", q, exc)
        prompt_preview = f"Retrieval error: {exc}"

    return {
        "documents": doc_count,
        "parsed": parsed_count,
        "chunks": chunk_count,
        "embeddings": emb_count,
        "vectors": emb_count,
        "retrieved": retrieved_count,
        "topK": top_k_results,
        "promptPreview": prompt_preview,
        "llmModel": settings.ollama_chat_model,
        "embeddingModel": settings.EMBEDDING_MODEL,
    }


@router.get("/documents", summary="Debug all documents in database")
async def debug_documents(
    session: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """Return status, chunk count, embedding count, and storage path for every document in DB."""
    stmt = select(Document).where(Document.deleted_at.is_(None))
    docs = list((await session.execute(stmt)).scalars().all())

    out = []
    for d in docs:
        version_stmt = select(DocumentVersion).where(DocumentVersion.document_id == d.id)
        versions = list((await session.execute(version_stmt)).scalars().all())
        v = versions[-1] if versions else None

        chunk_count = 0
        emb_count = 0
        if v:
            c_stmt = select(func.count(DocumentChunk.id)).where(DocumentChunk.document_version_id == v.id)
            chunk_count = (await session.execute(c_stmt)).scalar_one()

            if chunk_count > 0:
                chunk_ids_stmt = select(DocumentChunk.id).where(DocumentChunk.document_version_id == v.id)
                chunk_ids = list((await session.execute(chunk_ids_stmt)).scalars().all())
                e_stmt = select(func.count(Embedding.id)).where(Embedding.chunk_id.in_(chunk_ids))
                emb_count = (await session.execute(e_stmt)).scalar_one()

        out.append(
            {
                "document_id": str(d.id),
                "title": d.title,
                "status": d.status.value if hasattr(d.status, "value") else str(d.status),
                "parsed": v.status in (DocumentVersionStatus.PARSED, DocumentVersionStatus.CHUNKED, DocumentVersionStatus.EMBEDDED, DocumentVersionStatus.COMPLETED) if v else False,
                "chunk_count": chunk_count,
                "embedding_count": emb_count,
                "storage_path": d.storage_path or (v.storage_path if v else None),
                "last_error": d.last_error or (v.error_message if v else None),
            }
        )
    return out


@router.get("/retrieval", summary="Debug vector retrieval for a query")
async def debug_retrieval(
    q: str = Query(default="test", description="Query string to test retrieval"),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Diagnostic endpoint returning retrieval telemetry for Task 8."""
    settings = get_settings()
    doc_count = (await session.execute(select(func.count(Document.id)).where(Document.deleted_at.is_(None)))).scalar_one()
    chunk_count = (await session.execute(select(func.count(DocumentChunk.id)))).scalar_one()
    emb_count = (await session.execute(select(func.count(Embedding.id)))).scalar_one()

    retrieved_count = 0
    top_scores = []
    prompt_preview = ""

    try:
        retriever = Retriever(session)
        ranked_hits = await retriever.retrieve(q, filters=SearchFilters(), top_k=settings.TOP_K)
        retrieved_count = len(ranked_hits)
        top_scores = [round(hit.similarity_score, 4) for hit in ranked_hits]

        prompt_builder = PromptBuilder()
        built_prompt = prompt_builder.build(q, ranked_hits)
        prompt_preview = f"--- SYSTEM PROMPT ---\n{built_prompt.system_prompt}\n\n--- USER PROMPT ---\n{built_prompt.user_prompt}"
    except Exception as exc:
        prompt_preview = f"Error during retrieval: {exc}"

    return {
        "query_embedding_dimension": 768,
        "documents": doc_count,
        "chunks": chunk_count,
        "embeddings": emb_count,
        "retrieved_chunks": retrieved_count,
        "top_scores": top_scores,
        "prompt_preview": prompt_preview,
    }


@router.get("/install-help", summary="Get exact command to install flashrank into running python")
async def get_install_help() -> dict[str, Any]:
    """Return the exact command to run in terminal to install flashrank into the active server Python environment."""
    import sys
    cmd = f'"{sys.executable}" -m pip install flashrank'
    return {
        "python_executable": sys.executable,
        "recommended_command": cmd,
        "instructions": "Run the recommended_command in PowerShell or Command Prompt, then restart python run.py"
    }


@router.get("/reranker-status", summary="Check reranker model status and errors")
async def get_reranker_status() -> dict[str, Any]:
    """Return sys.executable, sys.path, and reranker initialization diagnostic status."""
    import sys
    import importlib
    import app.retrieval.ranking
    importlib.reload(app.retrieval.ranking)
    from app.retrieval.ranking import _get_neural_reranker
    instance, model_name = _get_neural_reranker()
    return {
        "sys_executable": sys.executable,
        "sys_path": sys.path,
        "reranker_instance_is_none": instance is None,
        "reranker_model_name": model_name,
    }


@router.get("/eval-reranker-v2", summary="Evaluate reranker across test questions")
async def evaluate_reranker_v2(
    q: str = Query(default="What is Talk to My Data?"),
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Execute neural reranking evaluation and return full trace as clean JSON."""
    import sys
    import time
    import traceback
    from app.retrieval.search import search_similar, search_fulltext, SearchFilters
    from app.retrieval.ranking import rank_hybrid_rrf, rerank_cross_encoder, _get_neural_reranker
    from app.embeddings.client import OllamaEmbeddingClient
    from app.llm.ollama_client import OllamaLLMClient
    from app.core.config import get_settings

    try:
        settings = get_settings()
        start_time = time.monotonic()

        # 1. Embed query
        embed_client = OllamaEmbeddingClient()
        query_emb = await embed_client.embed(q.strip())

        # 2. Candidate retrieval (Vector + FTS -> RRF TOP 20)
        sem_hits = await search_similar(session, query_emb, model_name=settings.EMBEDDING_MODEL, top_k=20, filters=SearchFilters())
        ft_hits = await search_fulltext(session, q.strip(), top_k=20, filters=SearchFilters())
        rrf_candidates = rank_hybrid_rrf(sem_hits, ft_hits, similarity_threshold=0.0)[:20]

        before_rerank = []
        target_chunk_id = "196b3db3-6876-4720-95a0-a09592790996"
        target_before_rank = None

        for cand in rrf_candidates:
            cid = str(cand.chunk_id)
            if cid == target_chunk_id:
                target_before_rank = cand.rank
            before_rerank.append({
                "rank": cand.rank,
                "chunk_id": cid,
                "section": cand.section_title or "",
                "rrf_score": float(cand.similarity_score),
                "preview": cand.chunk_text[:100],
            })

        # 3. Neural Cross-Encoder Reranking
        reranker_instance, reranker_model_name = _get_neural_reranker()
        rerank_start = time.monotonic()
        final_top_4 = rerank_cross_encoder(q.strip(), rrf_candidates, final_top_k=4)
        rerank_time_ms = int((time.monotonic() - rerank_start) * 1000)

        after_rerank = []
        target_after_rank = None
        target_after_score = None
        target_in_final_context = False

        for cand in final_top_4:
            cid = str(cand.chunk_id)
            if cid == target_chunk_id:
                target_after_rank = cand.rank
                target_after_score = float(cand.similarity_score)
                target_in_final_context = True
            after_rerank.append({
                "rank": cand.rank,
                "chunk_id": cid,
                "section": cand.section_title or "",
                "reranker_score": float(cand.similarity_score),
                "preview": cand.chunk_text[:100],
            })

        # 4. LLM Generation via OllamaLLMClient
        prompt_builder = PromptBuilder()
        built_prompt = prompt_builder.build(q, final_top_4)

        llm_client = OllamaLLMClient(
            model=settings.OLLAMA_MODEL or settings.CHAT_MODEL,
            temperature=0.1,
        )
        llm_start = time.monotonic()
        llm_resp = await llm_client.generate(
            system_prompt=built_prompt.system_prompt,
            user_prompt=built_prompt.user_prompt,
        )
        llm_time_ms = int((time.monotonic() - llm_start) * 1000)
        total_time_ms = int((time.monotonic() - start_time) * 1000)

        return JSONResponse(content={
            "sys_executable": sys.executable,
            "query": q,
            "reranker_model": str(reranker_model_name),
            "target_chunk_id": target_chunk_id,
            "target_before_rank": target_before_rank,
            "target_after_rank": target_after_rank,
            "target_after_score": target_after_score,
            "target_in_final_context": target_in_final_context,
            "rerank_latency_ms": rerank_time_ms,
            "llm_latency_ms": llm_time_ms,
            "total_latency_ms": total_time_ms,
            "before_rerank": before_rerank,
            "after_rerank": after_rerank,
            "llm_answer": llm_resp.answer,
            "citations": [
                {
                    "passage_id": str(c.chunk_id),
                    "section": c.section_title or "",
                    "document": c.document_title or "",
                    "page": c.page_number,
                }
                for c in final_top_4
            ]
        })
    except Exception as exc:
        return JSONResponse(content={"error": str(exc), "traceback": traceback.format_exc()})




# Trigger uvicorn reloader for clean baseline evaluation run
@router.get("/eval-benchmark", summary="Execute automated RAG evaluation benchmark")
async def execute_eval_benchmark(
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """In-process evaluation benchmark endpoint running asynchronously in background."""
    import asyncio
    import importlib.util
    import json
    import os
    import time

    status_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "eval", "benchmark_status.json")
    output_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "eval", "baseline_results.json")
    
    with open(status_path, "w", encoding="utf-8") as sf:
        json.dump({"status": "running", "started_at": time.time(), "completed_tests": 0, "total_tests": 100, "progress_pct": 0.0}, sf)

    if os.path.exists(output_path):
        os.remove(output_path)

    asyncio.create_task(_run_bg_eval_impl(status_path, output_path))
    return {"status": "started", "message": "Evaluation benchmark running in background via asyncio.create_task."}


async def _run_bg_eval_impl(status_path: str, output_path: str):
    import asyncio
    import importlib
    import json
    import os
    import sys
    import time
    from uuid import UUID

    await asyncio.sleep(0.1)

    if "app.rag.service" in sys.modules:
        importlib.reload(sys.modules["app.rag.service"])

    from app.db.session import AsyncSessionLocal
    from app.rag.service import RAGService
    try:
        benchmark_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "eval", "benchmark_dataset.json")
        if not os.path.exists(benchmark_path):
            with open(status_path, "w", encoding="utf-8") as sf:
                json.dump({"status": "error", "error": f"Dataset missing: {benchmark_path}"}, sf)
            return

        with open(benchmark_path, "r", encoding="utf-8") as f:
            benchmark = json.load(f)

        results = []
        total_latency_ms = 0
        total_grounding = 0.0
        total_correctness = 0.0
        total_citation = 0.0
        total_completeness = 0.0
        passed_count = 0
        failed_count = 0
        reasoning_leakage_count = 0
        verbatim_copy_count = 0

        with open(status_path, "w", encoding="utf-8") as sf:
            json.dump({"status": "running", "completed_tests": 0, "total_tests": len(benchmark), "progress_pct": 0.0}, sf)

        def compute_ngram_overlap(text1: str, text2: str, n: int = 5) -> float:
            words1 = [w.lower() for w in (text1 or "").split() if len(w) > 2]
            words2 = [w.lower() for w in (text2 or "").split() if len(w) > 2]
            if len(words1) < n or len(words2) < n:
                return 0.0
            ngrams1 = set(tuple(words1[i:i+n]) for i in range(len(words1)-n+1))
            ngrams2 = set(tuple(words2[i:i+n]) for i in range(len(words2)-n+1))
            if not ngrams1:
                return 0.0
            return len(ngrams1.intersection(ngrams2)) / len(ngrams1)

        for idx, test_case in enumerate(benchmark, start=1):
            category = test_case.get("category", "general")
            question = test_case["question"]
            expected_answer = test_case["expected_answer"]
            req_keywords = test_case.get("required_keywords", [])
            forbid_keywords = test_case.get("forbidden_keywords", [])
            exp_docs = test_case.get("expected_documents", [])

            start_time = time.monotonic()
            try:
                async with AsyncSessionLocal() as test_session:
                    from app.repositories.chat_session_repository import ChatSessionRepository
                    from app.repositories.user_repository import UserRepository
                    from app.services.chat_session_service import ChatSessionService
                    from app.services.session_resolution import get_or_create_swagger_demo_session

                    eval_session_id = await get_or_create_swagger_demo_session(
                        users=UserRepository(test_session),
                        sessions=ChatSessionRepository(test_session),
                        session_service=ChatSessionService(test_session),
                    )
                    await test_session.commit()
                    rag_service = RAGService(test_session)
                    rag_resp = await rag_service.ask(session_id=eval_session_id, question=question)

                duration_ms = int((time.monotonic() - start_time) * 1000)
                actual_answer = rag_resp.answer
                citations = [
                    {
                        "document_title": c.document_title,
                        "similarity_score": c.similarity_score,
                        "chunk_text": c.chunk_text,
                    }
                    for c in (rag_resp.sources or [])
                ]
                processing_time_ms = rag_resp.processing_time_ms or duration_ms
            except Exception as err:
                duration_ms = int((time.monotonic() - start_time) * 1000)
                actual_answer = f"Exception: {err}"
                citations = []
                processing_time_ms = duration_ms

            total_latency_ms += processing_time_ms
            lower_ans = actual_answer.lower()

            # 1. Reasoning leakage check
            has_leakage = False
            if "<think>" in lower_ans or "</think>" in lower_ans or "let me analyze" in lower_ans or "looking at passage" in lower_ans:
                has_leakage = True
                reasoning_leakage_count += 1
            for kw in forbid_keywords:
                if kw.lower() in lower_ans:
                    has_leakage = True

            # 2. Verbatim copy check
            verbatim_overlap = 0.0
            for c in citations:
                snippet = c.get("chunk_text", "")
                if snippet:
                    overlap = compute_ngram_overlap(actual_answer, snippet, 5)
                    if overlap > verbatim_overlap:
                        verbatim_overlap = overlap
            is_verbatim = verbatim_overlap > 0.40
            if is_verbatim:
                verbatim_copy_count += 1

            # 3. Grounding & Refusal check
            if category == "negative questions":
                grounding_score = 100.0 if "information not found" in lower_ans else 0.0
            else:
                grounding_score = 100.0 if len(citations) > 0 or "information not found" not in lower_ans else 50.0

            # 4. Keyword Completeness
            matched_keywords = [kw for kw in req_keywords if kw.lower() in lower_ans]
            completeness_ratio = len(matched_keywords) / len(req_keywords) if req_keywords else 1.0
            completeness_score = round(completeness_ratio * 100.0, 1)

            # 5. Citation Accuracy
            if exp_docs:
                cited_docs = [c.get("document_title", "") for c in citations]
                doc_matches = [doc for doc in exp_docs if any(doc.lower() in cd.lower() for cd in cited_docs)]
                citation_score = round((len(doc_matches) / len(exp_docs)) * 100.0, 1)
            else:
                citation_score = 100.0 if len(citations) == 0 else 80.0

            # 6. Overall Correctness Score
            correctness_score = 100.0
            issues = []
            failure_domain = "none"

            if category == "negative questions" and "information not found" not in lower_ans:
                correctness_score -= 50.0
                issues.append("Failed negative question refusal constraint")
                failure_domain = "prompt"

            if has_leakage:
                correctness_score -= 30.0
                issues.append("Reasoning leakage detected")
                failure_domain = "sanitization"

            if is_verbatim:
                correctness_score -= 20.0
                issues.append(f"Verbatim copying threshold exceeded ({verbatim_overlap:.1%})")
                failure_domain = "prompt"

            if completeness_ratio < 0.5:
                correctness_score -= 30.0
                issues.append(f"Low keyword completeness ({len(matched_keywords)}/{len(req_keywords)})")
                if failure_domain == "none":
                    failure_domain = "LLM"

            if exp_docs and citation_score < 50.0:
                correctness_score -= 20.0
                issues.append("Expected citations missing")
                if failure_domain == "none":
                    failure_domain = "retrieval"

            correctness_score = max(0.0, correctness_score)
            passed = correctness_score >= 70.0
            if passed:
                passed_count += 1
            else:
                failed_count += 1

            total_grounding += grounding_score
            total_correctness += correctness_score
            total_citation += citation_score
            total_completeness += completeness_score

            results.append({
                "id": test_case.get("id", idx),
                "category": category,
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": actual_answer,
                "citations": citations,
                "latency_ms": processing_time_ms,
                "correctness_score": correctness_score,
                "grounding_score": grounding_score,
                "citation_score": citation_score,
                "completeness_score": completeness_score,
                "has_reasoning_leakage": has_leakage,
                "is_verbatim_copy": is_verbatim,
                "passed": passed,
                "issues": issues,
                "failure_domain": failure_domain,
            })

            # Write progress update to status file every test
            with open(status_path, "w", encoding="utf-8") as sf:
                json.dump({
                    "status": "running",
                    "completed_tests": idx,
                    "total_tests": len(benchmark),
                    "progress_pct": round((idx / len(benchmark)) * 100.0, 1)
                }, sf)

        summary = {
            "total_tests": len(benchmark),
            "passed_count": passed_count,
            "failed_count": failed_count,
            "accuracy_pct": round((passed_count / len(benchmark)) * 100.0, 1),
            "avg_correctness": round((total_correctness / len(benchmark)), 1),
            "avg_grounding": round((total_grounding / len(benchmark)), 1),
            "avg_citation_accuracy": round((total_citation / len(benchmark)), 1),
            "avg_completeness": round((total_completeness / len(benchmark)), 1),
            "avg_latency_ms": round((total_latency_ms / len(benchmark)), 1),
            "reasoning_leakage_count": reasoning_leakage_count,
            "verbatim_copy_count": verbatim_copy_count,
            "results": results,
        }

        output_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "eval", "baseline_results.json")
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(summary, out_f, indent=2)

        with open(status_path, "w", encoding="utf-8") as sf:
            json.dump({"status": "completed", "summary": summary}, sf)
    except Exception as exc:
        with open(status_path, "w", encoding="utf-8") as sf:
            json.dump({"status": "error", "error": str(exc)}, sf)


@router.get("/restart-server", summary="Restart backend process")
async def restart_server() -> dict[str, str]:
    """Force backend process restart to reload modules."""
    import os
    import sys
    os._exit(0)
    return {"status": "restarting"}


@router.get("/eval-status", summary="Check eval benchmark status")
async def check_eval_status() -> dict[str, Any]:
    """Return status of background benchmark run."""
    import json
    import os
    status_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "eval", "benchmark_status.json")
    if not os.path.exists(status_path):
        return {"status": "idle"}
    with open(status_path, "r", encoding="utf-8") as f:
        return json.load(f)



