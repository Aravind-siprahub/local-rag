"""Orchestrate retrieval, prompting, and LLM generation for RAG Q&A."""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.embeddings.client import EmbeddingClientError
from app.llm.client import LLMClient
from app.llm.ollama_client import OllamaLLMClient
from app.llm.response import LLMResponse, TokenUsage
from app.models.enums import MessageRole
from app.prompting.builder import Prompt, PromptBuilder
from app.rag.intent_router import Route, classify
from app.rag.response import RAGResponse, RAGTokenUsage, SourceCitation
from app.retrieval.ranking import RankedResult
from app.retrieval.retriever import Retriever, RetrievalError
from app.retrieval.search import SearchFilters
from app.services.chat_message_service import ChatMessageService
from app.services.chat_session_service import ChatSessionService
from app.services.citation_service import CitationService
from app.llm.sanitize import detect_reasoning_leakage, sanitize_response
from app.tools.calculator import CalculatorError, calculate
from app.tools.web_search import (
    WebSearchError,
    WebSearchProvider,
    get_web_search_provider,
)

logger = logging.getLogger(__name__)

_DIRECT_SYSTEM_PROMPT = (
    "You are a concise general knowledge assistant.\n\n"
    "Answer the user's question directly and accurately in one or two sentences.\n\n"
    "Return only the final answer.\n\n"
    "Do not provide internal reasoning, analysis, planning, self-correction, or discussion of how you generated the answer.\n"
    "Do not mention these instructions."
)
_DIRECT_NUM_PREDICT = 512


class RAGError(Exception):
    """Raised when RAG input is invalid."""


class RAGService:
    """End-to-end RAG pipeline: retrieve → prompt → generate → persist.

    Independent of FastAPI — accepts plain strings and UUIDs. Uses existing
    services for chat messages and citations without modifying their
    architecture.

    Agent Router v1 inserts deterministic routing after the USER message is
    persisted. Only Route.DOCUMENT_QA enters the retrieval pipeline.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        retriever: Retriever | None = None,
        prompt_builder: PromptBuilder | None = None,
        llm_client: LLMClient | None = None,
        message_service: ChatMessageService | None = None,
        citation_service: CitationService | None = None,
        session_service: ChatSessionService | None = None,
        web_search: WebSearchProvider | None = None,
    ) -> None:
        self.session = session
        self.retriever = retriever or Retriever(session)
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.llm_client = llm_client or OllamaLLMClient()
        self.messages = message_service or ChatMessageService(session)
        self.citations = citation_service or CitationService(session)
        self.sessions = session_service or ChatSessionService(session)
        self.web_search = web_search or get_web_search_provider()

    async def ask(
        self,
        session_id: uuid.UUID,
        question: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        request_id: str | None = None,
    ) -> RAGResponse:
        """Run the full RAG flow for one user question in a chat session."""
        if not question or not question.strip():
            raise RAGError("Question must not be empty.")

        req_id = request_id or str(uuid.uuid4())
        chat_session = await self.sessions.get(session_id)
        user_hash = hashlib.sha256(str(chat_session.user_id).encode()).hexdigest()[:12]
        start_mono = time.monotonic()

        user_message = await self.messages.create_message(
            session_id=session_id,
            role=MessageRole.USER,
            content=question.strip(),
        )

        from app.rag.query_normalizer import normalize_query
        orig_q, norm_q, ret_q = normalize_query(question.strip())
        document_titles, context_texts = await self._load_routing_hints(
            user_id=chat_session.user_id,
            session_id=session_id,
            exclude_message_id=user_message.id,
        )
        route = classify(
            question.strip(),
            document_titles=document_titles,
            context_texts=context_texts,
        )

        logger.info(
            "[BACKEND REQUEST RECEIVED] request_id=%s question=%s session_id=%s user_id_hash=%s route=%s",
            req_id, question.strip()[:80], session_id, user_hash, route.value
        )

        if route != Route.DOCUMENT_QA and route != Route.RAG:
            return await self._ask_non_rag(
                route=route,
                session_id=session_id,
                question=question.strip(),
                user_message_id=user_message.id,
                start_mono=start_mono,
                request_id=req_id,
                user_hash=user_hash,
                norm_q=norm_q,
            )

        logger.info("[AI ROUTER] DOCUMENT_QA selected, entering retrieval pipeline")

        retrieval_filters = filters or SearchFilters()
        if retrieval_filters.user_id is None:
            retrieval_filters = SearchFilters(
                user_id=chat_session.user_id,
                document_id=retrieval_filters.document_id,
                document_version_id=retrieval_filters.document_version_id,
            )

        # Detect document title references in user question if document_id is not set
        if retrieval_filters.document_id is None and self.session is not None:
            try:
                from app.models.document import Document
                from sqlalchemy import select
                stmt_docs = (
                    select(Document)
                    .where(Document.deleted_at.is_(None))
                    .where(Document.user_id == chat_session.user_id)
                )
                all_docs = list((await self.session.execute(stmt_docs)).scalars().all())
                if not all_docs:
                    stmt_all_docs = (
                        select(Document)
                        .where(Document.deleted_at.is_(None))
                    )
                    all_docs = list((await self.session.execute(stmt_all_docs)).scalars().all())

                q_lower = question.strip().lower()
                for d in all_docs:
                    d_title_lower = d.title.lower()
                    d_stem = d_title_lower.rsplit(".", 1)[0] if "." in d_title_lower else d_title_lower
                    stem_clean = d_stem.replace("_", " ").replace("-", " ").strip()
                    if len(stem_clean) >= 3 and (
                        d_title_lower in q_lower
                        or d_stem in q_lower
                        or stem_clean in q_lower
                    ):
                        logger.info("[DOCUMENT TITLE DETECTED] Question references document '%s' (%s)", d.title, d.id)
                        retrieval_filters = SearchFilters(
                            user_id=retrieval_filters.user_id,
                            document_id=d.id,
                            document_version_id=retrieval_filters.document_version_id,
                        )
                        break
            except Exception as d_exc:
                logger.warning("[DOCUMENT TITLE MATCH FAILED] %s", d_exc)

        retrieval_start = time.monotonic()
        logger.info("[RAG STAGE 2: RETRIEVAL START] orig=%s norm=%s ret=%s filters=%s top_k=%s", orig_q[:60], norm_q[:60], ret_q[:60], retrieval_filters, top_k)
        retrieved_chunks = await self._retrieve_safely(
            ret_q or question.strip(),
            filters=retrieval_filters,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        # Fallback retrieval pass 2 with normalized query if initial pass yielded 0 chunks
        if not retrieved_chunks and norm_q and norm_q != ret_q:
            logger.info("[RAG FALLBACK RETRIEVAL] Primary search yielded 0 hits. Retrying with normalized query: %s", norm_q)
            retrieved_chunks = await self._retrieve_safely(
                norm_q,
                filters=retrieval_filters,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )

        # Fallback retrieval pass 3 without user_id filter if initial passes yielded 0 chunks
        if not retrieved_chunks and retrieval_filters.user_id is not None:
            logger.info("[RAG FALLBACK RETRIEVAL] Retrying without user_id filter")
            fallback_filters = SearchFilters(
                document_id=retrieval_filters.document_id,
                document_version_id=retrieval_filters.document_version_id,
            )
            retrieved_chunks = await self._retrieve_safely(
                ret_q or question.strip(),
                filters=fallback_filters,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )

        retrieval_ms = int((time.monotonic() - retrieval_start) * 1000)
        logger.info("[RETRIEVAL FINISHED] retrieval_ms=%d hits=%d", retrieval_ms, len(retrieved_chunks))

        # Relevance Gate: If zero chunks pass similarity threshold
        if not retrieved_chunks:
            total_ms = int((time.monotonic() - start_mono) * 1000)
            fallback_answer = "I could not find this information in your uploaded documents."
            logger.warning("[NO RELEVANT DOCUMENTS FOUND] 0 chunks passed relevance gate. Returning fallback response.")

            self._log_structured_trace(
                request_id=req_id,
                user_hash=user_hash,
                session_id=session_id,
                orig_q=orig_q,
                norm_q=norm_q,
                route=route,
                retrieval_ms=retrieval_ms,
                context_ms=0,
                llm_ms=0,
                total_ms=total_ms,
                chunks_retrieved=0,
                top_similarity=0.0,
                status="SUCCESS",
                error_type=None,
            )

            assistant_message = await self.messages.create_message(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=fallback_answer,
                model_used=getattr(self.llm_client, "model", "ollama"),
                latency_ms=total_ms,
                generation_time_ms=0,
            )
            return RAGResponse(
                answer=fallback_answer,
                sources=[],
                token_usage=RAGTokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                model=getattr(self.llm_client, "model", "ollama"),
                processing_time_ms=total_ms,
                user_message_id=user_message.id,
                assistant_message_id=assistant_message.id,
            )

        # Fetch recent chat history only to enrich the retrieval question when needed.
        # PromptBuilder currently accepts question + chunks (no chat_history kwarg).
        prompt_start = time.monotonic()
        prompt = self.prompt_builder.build(
            question.strip(),
            retrieved_chunks,
        )
        context_ms = int((time.monotonic() - prompt_start) * 1000)

        top_sim = retrieved_chunks[0].similarity_score if retrieved_chunks else 0.0

        settings = get_settings()
        num_predict = settings.OLLAMA_NUM_PREDICT

        llm_start = time.monotonic()
        logger.info("[LLM GENERATION STARTED] model=%s num_predict=%d", getattr(self.llm_client, "model", "ollama"), num_predict)
        llm_response = await self.llm_client.generate(prompt.system_prompt, prompt.user_prompt, num_predict=num_predict)
        llm_ms = int((time.monotonic() - llm_start) * 1000)
        logger.info("[LLM GENERATION FINISHED] llm_ms=%d", llm_ms)

        # Truncation check: only retry when the first pass produced no usable answer.
        if getattr(llm_response, "finish_reason", None) == "length":
            first_pass = sanitize_response(llm_response.answer)
            if not first_pass or len(first_pass.strip()) < 20:
                logger.warning(
                    "[LLM TRUNCATION DETECTED] finish_reason=length and answer unusable. "
                    "Retrying once with num_predict *= 2 (%d)",
                    num_predict * 2,
                )
                num_predict *= 2
                llm_response = await self.llm_client.generate(
                    prompt.system_prompt,
                    prompt.user_prompt,
                    num_predict=num_predict,
                )
            else:
                logger.info(
                    "[LLM TRUNCATION DETECTED] finish_reason=length but usable answer present; skipping retry"
                )

        # Sanitize answer from thinking tags or monologue prefixes
        clean_ans = sanitize_response(llm_response.answer)
        clean_ans_lower = clean_ans.lower().strip()
        if not clean_ans or clean_ans_lower == "i could not find this information in the uploaded documents." or clean_ans_lower == "i could not find this information in your uploaded documents." or clean_ans_lower == "information not found in document excerpts." or clean_ans_lower == "information not found":
            answer_text = "I could not find this information in your uploaded documents."
        else:
            answer_text = clean_ans

        # Citation Validation: Strict threshold enforcement & deduplication
        effective_threshold = similarity_threshold if similarity_threshold is not None else settings.SIMILARITY_THRESHOLD
        if answer_text == "I could not find this information in your uploaded documents.":
            sources = []
        else:
            raw_sources = _sources_from_prompt(prompt)
            sources = _validate_and_deduplicate_sources(
                raw_sources,
                effective_threshold,
                max_sources=getattr(settings, "FINAL_CONTEXT", 3),
            )

        total_ms = int((time.monotonic() - start_mono) * 1000)

        self._log_structured_trace(
            request_id=req_id,
            user_hash=user_hash,
            session_id=session_id,
            orig_q=orig_q,
            norm_q=norm_q,
            route=route,
            retrieval_ms=retrieval_ms,
            context_ms=context_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
            chunks_retrieved=len(retrieved_chunks),
            top_similarity=top_sim,
            status="SUCCESS",
            error_type=None,
        )

        token_usage = llm_response.token_usage
        assistant_message = await self.messages.create_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=answer_text,
            model_used=llm_response.model_name,
            prompt_tokens=token_usage.prompt_tokens if token_usage else None,
            completion_tokens=token_usage.completion_tokens if token_usage else None,
            latency_ms=total_ms,
            generation_time_ms=llm_ms,
        )

        if sources:
            await self.citations.create_citations_for_message(
                assistant_message.id,
                [
                    {
                        "chunk_id": source.chunk_id,
                        "rank": source.rank,
                        "similarity_score": source.similarity_score,
                    }
                    for source in sources
                ],
            )

        processing_time_ms = int((time.monotonic() - start_mono) * 1000)

        return RAGResponse(
            answer=answer_text,
            sources=sources,
            token_usage=_map_token_usage(token_usage),
            model=llm_response.model_name,
            processing_time_ms=processing_time_ms,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
        )

    async def ask_stream(
        self,
        session_id: uuid.UUID,
        question: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ):
        """Run RAG pipeline and yield Server-Sent Events (SSE) formatting for real-time streaming."""
        import json

        if not question or not question.strip():
            raise RAGError("Question must not be empty.")

        chat_session = await self.sessions.get(session_id)
        start_mono = time.monotonic()

        user_message = await self.messages.create_message(
            session_id=session_id,
            role=MessageRole.USER,
            content=question.strip(),
        )

        from app.rag.intent_router import Route, classify
        from app.rag.query_normalizer import normalize_query
        orig_q, norm_q, ret_q = normalize_query(question.strip())
        document_titles, context_texts = await self._load_routing_hints(
            user_id=chat_session.user_id,
            session_id=session_id,
            exclude_message_id=user_message.id,
        )
        route = classify(
            question.strip(),
            document_titles=document_titles,
            context_texts=context_texts,
        )

        if route != Route.DOCUMENT_QA and route != Route.RAG:
            res = await self._ask_non_rag(
                route=route,
                session_id=session_id,
                question=question.strip(),
                user_message_id=user_message.id,
                start_mono=start_mono,
                norm_q=norm_q,
            )
            yield f"data: {json.dumps({'type': 'meta', 'sources': [], 'user_message_id': str(user_message.id)})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'content': res.answer})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'assistant_message_id': str(res.assistant_message_id), 'processing_time_ms': res.processing_time_ms})}\n\n"
            return

        retrieval_filters = filters or SearchFilters()
        if retrieval_filters.user_id is None:
            retrieval_filters = SearchFilters(
                user_id=chat_session.user_id,
                document_id=retrieval_filters.document_id,
                document_version_id=retrieval_filters.document_version_id,
            )

        retrieved_chunks = await self._retrieve_safely(
            question.strip(),
            filters=retrieval_filters,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        if not retrieved_chunks and retrieval_filters.user_id is not None:
            fallback_chunks = await self._retrieve_safely(
                question.strip(),
                filters=SearchFilters(
                    document_id=retrieval_filters.document_id,
                    document_version_id=retrieval_filters.document_version_id,
                ),
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            if fallback_chunks:
                retrieved_chunks = fallback_chunks

        settings = get_settings()
        effective_threshold = similarity_threshold if similarity_threshold is not None else settings.SIMILARITY_THRESHOLD
        valid_stream_chunks = [
            c for c in retrieved_chunks
            if c.similarity_score >= effective_threshold
        ]
        seen_keys: set[tuple[uuid.UUID, Any, Any]] = set()
        deduped_chunks = []
        for c in valid_stream_chunks:
            key = (c.document_id, getattr(c, "page_number", None), getattr(c, "section_title", None))
            if key not in seen_keys:
                seen_keys.add(key)
                deduped_chunks.append(c)
                if len(deduped_chunks) >= getattr(settings, "FINAL_CONTEXT", 3):
                    break

        sources_data = [
            {
                "chunk_id": str(c.chunk_id),
                "chunk_text": c.chunk_text,
                "document_id": str(c.document_id),
                "document_version_id": str(c.document_version_id),
                "similarity_score": round(c.similarity_score, 4),
                "rank": c.rank,
                "document_title": getattr(c, "document_title", "Unknown"),
            }
            for c in deduped_chunks
        ]

        yield f"data: {json.dumps({'type': 'meta', 'sources': sources_data, 'user_message_id': str(user_message.id)})}\n\n"

        if not retrieved_chunks:
            fallback_ans = "Information not found in document excerpts."
            total_ms = int((time.monotonic() - start_mono) * 1000)
            assistant_msg = await self.messages.create_message(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=fallback_ans,
                model_used=getattr(self.llm_client, "model", "ollama"),
                latency_ms=total_ms,
            )
            yield f"data: {json.dumps({'type': 'token', 'content': fallback_ans})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'assistant_message_id': str(assistant_msg.id), 'processing_time_ms': total_ms})}\n\n"
            return

        prompt = self.prompt_builder.build(
            question.strip(),
            retrieved_chunks,
        )

        full_answer_chunks: list[str] = []
        try:
            if hasattr(self.llm_client, "generate_stream"):
                async for token in self.llm_client.generate_stream(prompt.system_prompt, prompt.user_prompt):
                    full_answer_chunks.append(token)
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            else:
                resp = await self.llm_client.generate(prompt.system_prompt, prompt.user_prompt)
                safe = sanitize_response(resp.answer)
                full_answer_chunks.append(safe)
                yield f"data: {json.dumps({'type': 'token', 'content': safe})}\n\n"
        except Exception as exc:
            logger.error("Streaming error: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

        full_answer = sanitize_response("".join(full_answer_chunks).strip()) or "Information not found in document excerpts."
        total_ms = int((time.monotonic() - start_mono) * 1000)

        assistant_message = await self.messages.create_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=full_answer,
            model_used=getattr(self.llm_client, "model", "ollama"),
            latency_ms=total_ms,
        )

        if sources_data:
            await self.citations.create_citations_for_message(
                assistant_message.id,
                [
                    {
                        "chunk_id": uuid.UUID(s["chunk_id"]) if isinstance(s["chunk_id"], str) else s["chunk_id"],
                        "rank": s["rank"],
                        "similarity_score": s["similarity_score"],
                    }
                    for s in sources_data
                ],
            )

        yield f"data: {json.dumps({'type': 'done', 'assistant_message_id': str(assistant_message.id), 'processing_time_ms': total_ms})}\n\n"

    async def _ask_non_rag(
        self,
        *,
        route: Route,
        session_id: uuid.UUID,
        question: str,
        user_message_id: uuid.UUID,
        start_mono: float,
        request_id: str | None = None,
        user_hash: str | None = None,
        norm_q: str | None = None,
    ) -> RAGResponse:
        """Handle DOCUMENT_LIST / DOCUMENT_METADATA / WEB / CALCULATOR / DIRECT without vector retrieval."""
        res: RAGResponse
        if route == Route.DOCUMENT_LIST:
            chat_session = await self.sessions.get(session_id)
            res = await self._ask_document_list(
                session_id=session_id,
                user_id=chat_session.user_id,
                user_message_id=user_message_id,
                start_mono=start_mono,
            )
        elif route == Route.DOCUMENT_METADATA:
            chat_session = await self.sessions.get(session_id)
            res = await self._ask_document_metadata(
                session_id=session_id,
                user_id=chat_session.user_id,
                user_message_id=user_message_id,
                start_mono=start_mono,
                question=question,
            )
        elif route == Route.WEB:
            res = await self._ask_web(
                session_id=session_id,
                question=question,
                user_message_id=user_message_id,
                start_mono=start_mono,
            )
        elif route == Route.CALCULATOR:
            res = await self._ask_calculator(
                session_id=session_id,
                question=question,
                user_message_id=user_message_id,
                start_mono=start_mono,
            )
        else:
            res = await self._ask_direct(
                session_id=session_id,
                question=question,
                user_message_id=user_message_id,
                start_mono=start_mono,
                norm_q=norm_q,
            )

        total_ms = int((time.monotonic() - start_mono) * 1000)
        self._log_structured_trace(
            request_id=request_id or str(uuid.uuid4()),
            user_hash=user_hash or "anonymous",
            session_id=session_id,
            orig_q=question,
            norm_q=norm_q or question,
            route=route,
            retrieval_ms=0,
            context_ms=0,
            llm_ms=res.processing_time_ms,
            total_ms=total_ms,
            chunks_retrieved=0,
            top_similarity=0.0,
            status="SUCCESS",
            error_type=None,
        )
        return res

    async def _ask_document_metadata(
        self,
        *,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        user_message_id: uuid.UUID,
        start_mono: float,
        question: str,
    ) -> RAGResponse:
        from app.models.document import Document
        from sqlalchemy import select

        stmt = (
            select(Document)
            .where(Document.deleted_at.is_(None))
            .where(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
        )
        docs = list((await self.session.execute(stmt)).scalars().all())

        q_lower = question.lower()
        matched_docs = [
            d for d in docs
            if d.title.lower() in q_lower
            or (d.title.rsplit(".", 1)[0].lower() in q_lower if "." in d.title else False)
        ]
        target_docs = matched_docs if matched_docs else docs

        if target_docs:
            lines = []
            for d in target_docs:
                created_str = d.created_at.strftime("%Y-%m-%d %H:%M UTC") if d.created_at else "Unknown date"
                size_kb = round(d.file_size_bytes / 1024.0, 1) if d.file_size_bytes else 0.0
                status_str = d.status.value if hasattr(d.status, "value") else str(d.status)
                lines.append(f"• **{d.title}** — Uploaded on {created_str} | Size: {size_kb} KB | Status: {status_str}")
            answer_text = "Here is the document metadata:\n\n" + "\n".join(lines)
        else:
            answer_text = "You currently have no uploaded documents to inspect metadata for."

        total_ms = int((time.monotonic() - start_mono) * 1000)
        logger.info("[DOCUMENT METADATA DIRECT] Found %d documents for user_id=%s", len(target_docs), user_id)
        assistant_msg = await self.messages.create_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=answer_text,
            model_used="database-metadata",
            latency_ms=total_ms,
            generation_time_ms=total_ms,
        )
        return RAGResponse(
            answer=answer_text,
            sources=[],
            token_usage=None,
            model="database-metadata",
            processing_time_ms=total_ms,
            user_message_id=user_message_id,
            assistant_message_id=assistant_msg.id,
        )

    def _log_structured_trace(
        self,
        *,
        request_id: str,
        user_hash: str,
        session_id: uuid.UUID,
        orig_q: str,
        norm_q: str,
        route: Route,
        retrieval_ms: int,
        context_ms: int,
        llm_ms: int,
        total_ms: int,
        chunks_retrieved: int,
        top_similarity: float,
        status: str,
        error_type: str | None = None,
    ) -> None:
        trace_payload = {
            "request_id": request_id,
            "user_id_hash": user_hash,
            "conversation_id": str(session_id),
            "original_query": orig_q[:100],
            "normalized_query": norm_q[:100],
            "intent": route.value,
            "route": route.value,
            "embedding_ms": 0,
            "retrieval_ms": retrieval_ms,
            "context_ms": context_ms,
            "llm_ms": llm_ms,
            "total_ms": total_ms,
            "retrieved_chunks": chunks_retrieved,
            "top_similarity": round(top_similarity, 4),
            "response_status": status,
            "error_type": error_type,
        }
        logger.info("[ENTERPRISE RAG TRACE] %s", json.dumps(trace_payload))

    async def _ask_document_list(
        self,
        *,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        user_message_id: uuid.UUID,
        start_mono: float,
    ) -> RAGResponse:
        from app.models.document import Document
        from app.models.enums import DocumentStatus
        from sqlalchemy import select

        stmt = (
            select(Document.title)
            .where(Document.deleted_at.is_(None))
            .where(Document.status.in_([DocumentStatus.READY, DocumentStatus.PROCESSING, DocumentStatus.UPLOADED, "ready", "processing", "uploaded"]))
        )
        if user_id is not None:
            stmt_user = stmt.where(Document.user_id == user_id).order_by(Document.title)
            titles = list((await self.session.execute(stmt_user)).scalars().all())
        else:
            titles = []

        if not titles:
            titles = list((await self.session.execute(stmt.order_by(Document.title))).scalars().all())

        if titles:
            formatted_list = (
                f"You have {len(titles)} uploaded document{'s' if len(titles) != 1 else ''}:\n\n"
                + "\n".join(f"{idx}. {t}" for idx, t in enumerate(titles, 1))
            )
        else:
            formatted_list = "You currently have no documents uploaded. Please upload a document (PDF, DOCX, TXT, MD, CSV) to get started."

        total_ms = int((time.monotonic() - start_mono) * 1000)
        logger.info("[DOCUMENT LIST DIRECT] Found %d documents for user_id=%s", len(titles), user_id)
        assistant_msg = await self.messages.create_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=formatted_list,
            model_used="database-direct",
            latency_ms=total_ms,
            generation_time_ms=total_ms,
        )
        return RAGResponse(
            answer=formatted_list,
            sources=[],
            token_usage=None,
            model="database-direct",
            processing_time_ms=total_ms,
            user_message_id=user_message_id,
            assistant_message_id=assistant_msg.id,
        )

    async def _ask_web(
        self,
        *,
        session_id: uuid.UUID,
        question: str,
        user_message_id: uuid.UUID,
        start_mono: float,
    ) -> RAGResponse:
        try:
            result = await self.web_search.search(question)
            answer_text = result.concise_answer()
            model_name = f"web-search:{result.provider}"
        except WebSearchError as exc:
            logger.error("[AI ROUTER] web search error: %s", exc, exc_info=True)
            answer_text = (
                "I could not find reliable web results for that question right now. "
                "Please try again shortly."
            )
            model_name = "web-search:error"
        except Exception as exc:
            logger.error("[AI ROUTER] unexpected web search exception: %s", exc, exc_info=True)
            answer_text = (
                "I could not find reliable web results for that question right now. "
                "Please try again shortly."
            )
            model_name = "web-search:error"

        total_ms = int((time.monotonic() - start_mono) * 1000)
        assistant_message = await self.messages.create_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=answer_text,
            model_used=model_name,
            latency_ms=total_ms,
            generation_time_ms=total_ms,
        )
        logger.info("[RESPONSE RETURNED] total_ms=%d route=WEB", total_ms)
        return RAGResponse(
            answer=answer_text,
            sources=[],
            token_usage=None,
            model=model_name,
            processing_time_ms=total_ms,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message.id,
        )

    async def _ask_calculator(
        self,
        *,
        session_id: uuid.UUID,
        question: str,
        user_message_id: uuid.UUID,
        start_mono: float,
    ) -> RAGResponse:
        try:
            result = calculate(question)
            answer_text = f"The result is {result.display}."
            model_name = "calculator"
        except CalculatorError:
            logger.warning("[AI ROUTER] calculator failed for question_len=%d", len(question))
            answer_text = (
                "I could not evaluate that as a safe arithmetic expression. "
                "Please rephrase with a clear calculation."
            )
            model_name = "calculator:error"

        total_ms = int((time.monotonic() - start_mono) * 1000)
        assistant_message = await self.messages.create_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=answer_text,
            model_used=model_name,
            latency_ms=total_ms,
            generation_time_ms=total_ms,
        )
        logger.info("[RESPONSE RETURNED] total_ms=%d route=CALCULATOR", total_ms)
        return RAGResponse(
            answer=answer_text,
            sources=[],
            token_usage=None,
            model=model_name,
            processing_time_ms=total_ms,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message.id,
        )

    async def _ask_direct(
        self,
        *,
        session_id: uuid.UUID,
        question: str,
        user_message_id: uuid.UUID,
        start_mono: float,
        norm_q: str | None = None,
    ) -> RAGResponse:
        llm_start = time.monotonic()
        query_text = norm_q.strip() if (norm_q and norm_q.strip()) else question.strip()

        llm_response = await self.llm_client.generate(
            _DIRECT_SYSTEM_PROMPT,
            query_text,
            num_predict=_DIRECT_NUM_PREDICT,
        )
        llm_ms = int((time.monotonic() - llm_start) * 1000)
        answer_text = sanitize_response(llm_response.answer).strip()

        # Validate answer — retry once if response is empty, truncated, or contains CoT monologue
        if not _is_valid_direct_answer(answer_text):
            logger.warning("[DIRECT ANSWER REJECTED] invalid/truncated answer=%r. Retrying once.", answer_text)
            retry_start = time.monotonic()
            retry_prompt = (
                "You are a concise general knowledge assistant. Answer the user's question directly and accurately in 1 or 2 sentences.\n\n"
                "Return ONLY the final answer. Do not include internal thoughts, commentary, or greetings."
            )
            llm_response = await self.llm_client.generate(
                retry_prompt,
                query_text,
                num_predict=_DIRECT_NUM_PREDICT,
            )
            llm_ms += int((time.monotonic() - retry_start) * 1000)
            answer_text = sanitize_response(llm_response.answer).strip()

        if not answer_text or not _is_valid_direct_answer(answer_text):
            answer_text = "I could not generate an answer right now."

        total_ms = int((time.monotonic() - start_mono) * 1000)
        token_usage = llm_response.token_usage
        assistant_message = await self.messages.create_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=answer_text,
            model_used=llm_response.model_name,
            prompt_tokens=token_usage.prompt_tokens if token_usage else None,
            completion_tokens=token_usage.completion_tokens if token_usage else None,
            latency_ms=total_ms,
            generation_time_ms=llm_ms,
        )
        logger.info("[RESPONSE RETURNED] total_ms=%d route=DIRECT", total_ms)
        return RAGResponse(
            answer=answer_text,
            sources=[],
            token_usage=_map_token_usage(token_usage),
            model=llm_response.model_name,
            processing_time_ms=total_ms,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message.id,
        )

    async def _load_routing_hints(
        self,
        *,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        exclude_message_id: uuid.UUID | None = None,
    ) -> tuple[list[str], list[str]]:
        """Load user document titles + recent chat turns for corpus-aware routing."""
        from app.models.document import Document
        from sqlalchemy import select

        titles: list[str] = []
        try:
            stmt = (
                select(Document.title)
                .where(Document.deleted_at.is_(None))
                .where(Document.user_id == user_id)
            )
            titles = [str(t) for t in (await self.session.execute(stmt)).scalars().all() if t]
        except Exception as exc:
            logger.warning("[ROUTING HINTS] failed to load document titles: %s", exc)

        context_texts: list[str] = []
        try:
            # Recent messages only — enough to resolve anaphora, not full history.
            recent = await self.messages.list_by_session(session_id, limit=8, offset=0)
            for msg in recent:
                if exclude_message_id is not None and msg.id == exclude_message_id:
                    continue
                content = (msg.content or "").strip()
                if content:
                    context_texts.append(content)
            # Keep chronological order if repository returns newest-first.
            if len(context_texts) >= 2 and recent and recent[0].created_at and recent[-1].created_at:
                if recent[0].created_at > recent[-1].created_at:
                    context_texts = list(reversed(context_texts))
            context_texts = context_texts[-6:]
        except Exception as exc:
            logger.warning("[ROUTING HINTS] failed to load conversation context: %s", exc)

        return titles, context_texts

    async def _retrieve_safely(
        self,
        question: str,
        *,
        filters: SearchFilters,
        top_k: int | None,
        similarity_threshold: float | None,
    ) -> list[RankedResult]:
        try:
            return await self.retriever.retrieve(
                question,
                filters=filters,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
        except Exception as exc:
            logger.exception("Retrieval failed in _retrieve_safely: %s", exc)
            raise

    async def close(self) -> None:
        await self.retriever.close()
        await self.llm_client.close()
        close_web = getattr(self.web_search, "close", None)
        if close_web is not None:
            await close_web()


def _sources_from_prompt(prompt: Prompt) -> list[SourceCitation]:
    return [
        SourceCitation(
            chunk_id=chunk.chunk_id,
            chunk_text=chunk.chunk_text,
            document_id=chunk.document_id,
            document_version_id=chunk.document_version_id,
            similarity_score=chunk.similarity_score,
            rank=chunk.rank,
        )
        for chunk in prompt.retrieved_chunks
    ]


def _validate_and_deduplicate_sources(
    sources: list[SourceCitation],
    similarity_threshold: float,
    max_sources: int = 3,
) -> list[SourceCitation]:
    """Filter, deduplicate, and limit valid source citations."""
    valid = [s for s in sources if s.similarity_score >= similarity_threshold]
    if not valid:
        return []

    seen: set[uuid.UUID] = set()
    deduped: list[SourceCitation] = []
    for s in valid:
        if s.chunk_id in seen:
            continue
        seen.add(s.chunk_id)
        deduped.append(s)
        if len(deduped) >= max_sources:
            break
    return deduped


def _is_valid_direct_answer(ans: str) -> bool:
    """Check if the generated direct answer is non-empty, complete, and free of reasoning leakage."""
    if not ans or not isinstance(ans, str) or not ans.strip():
        return False
    clean = ans.strip()
    if len(clean) < 2:
        return False
    if detect_reasoning_leakage(clean):
        return False
    low = clean.lower()
    reasoning_keywords = (
        "okay, the core question",
        "that phrasing is",
        "the user seems",
        "i recall that",
        "checks reliable knowledge",
        "wait, the instruction",
        "so the cleanest response",
        "let me unpack",
        "mixing up concepts",
        "first, i'll",
        "first, i will",
        "hmm, the user",
        "checks requirements",
        "i need to respond",
    )
    if any(kw in low for kw in reasoning_keywords):
        return False
    if low in ("earth is one", "or just", "the user", "wait", "okay"):
        return False
    return True


def _map_token_usage(token_usage: TokenUsage | None) -> RAGTokenUsage | None:
    if token_usage is None:
        return None
    return RAGTokenUsage(
        prompt_tokens=token_usage.prompt_tokens,
        completion_tokens=token_usage.completion_tokens,
        total_tokens=token_usage.total_tokens,
    )
