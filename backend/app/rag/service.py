"""Orchestrate retrieval, prompting, and LLM generation for RAG Q&A."""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import replace

from sqlalchemy.ext.asyncio import AsyncSession

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
from app.llm.sanitize import sanitize_response
from app.tools.calculator import CalculatorError, calculate
from app.tools.web_search import (
    WebSearchError,
    WebSearchProvider,
    get_web_search_provider,
)

logger = logging.getLogger(__name__)

_DIRECT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer clearly and concisely. "
    "Do not invent citations. Do not include chain-of-thought."
)
_DIRECT_NUM_PREDICT = 128


class RAGError(Exception):
    """Raised when RAG input is invalid."""


class RAGService:
    """End-to-end RAG pipeline: retrieve → prompt → generate → persist.

    Independent of FastAPI — accepts plain strings and UUIDs. Uses existing
    services for chat messages and citations without modifying their
    architecture.

    Agent Router v1 inserts deterministic routing after the USER message is
    persisted. Only Route.RAG enters the retrieval pipeline.
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
    ) -> RAGResponse:
        """Run the full RAG flow for one user question in a chat session."""
        if not question or not question.strip():
            raise RAGError("Question must not be empty.")

        chat_session = await self.sessions.get(session_id)
        start_mono = time.monotonic()

        user_message = await self.messages.create_message(
            session_id=session_id,
            role=MessageRole.USER,
            content=question.strip(),
        )

        logger.info("[BACKEND REQUEST RECEIVED] question=%s session_id=%s user_id=%s", question.strip(), session_id, chat_session.user_id)

        route = classify(question.strip())
        if route != Route.RAG:
            return await self._ask_non_rag(
                route=route,
                session_id=session_id,
                question=question.strip(),
                user_message_id=user_message.id,
                start_mono=start_mono,
            )

        logger.info("[AI ROUTER] RAG selected, entering retrieval pipeline")

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
                q_lower = question.strip().lower()
                for d in all_docs:
                    d_title_lower = d.title.lower()
                    d_stem = d_title_lower.rsplit(".", 1)[0] if "." in d_title_lower else d_title_lower
                    stem_clean = d_stem.replace("_", " ").replace("-", " ").strip()
                    if len(stem_clean) >= 4 and (
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

        from app.rag.query_normalizer import normalize_query
        orig_q, norm_q, ret_q = normalize_query(question.strip())

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

        retrieval_ms = int((time.monotonic() - start_mono) * 1000)
        logger.info("[RETRIEVAL FINISHED] retrieval_ms=%d hits=%d", retrieval_ms, len(retrieved_chunks))

        # If no chunks exist after primary and fallback retrieval
        if not retrieved_chunks:
            total_ms = int((time.monotonic() - start_mono) * 1000)
            fallback_answer = "I could not find this information in the uploaded documents."
            logger.warning("[NO DOCUMENTS FOUND] 0 chunks retrieved. Returning fallback response.")

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

        # Fetch recent chat history before adding current message
        raw_history = await self.messages.list_by_session(session_id, limit=6)
        formatted_history = [
            {"role": m.role.value if hasattr(m.role, "value") else str(m.role), "content": m.content}
            for m in raw_history
            if m.id != user_message.id
        ]

        prompt_start = time.monotonic()
        prompt = self.prompt_builder.build(
            question.strip(),
            retrieved_chunks,
            chat_history=formatted_history,
        )
        prompt_ms = int((time.monotonic() - prompt_start) * 1000)

        top_sim = retrieved_chunks[0].similarity_score if retrieved_chunks else 0.0
        logger.info(
            "[RAG DEBUG TRACE]\n"
            "  question=%s\n"
            "  user_id=%s\n"
            "  session_id=%s\n"
            "  intent=%s\n"
            "  chunks_retrieved=%d\n"
            "  top_similarity=%.4f\n"
            "  context_length=%d\n"
            "  llm_model=%s",
            question.strip(),
            chat_session.user_id,
            session_id,
            route.value,
            len(retrieved_chunks),
            top_sim,
            len(prompt.user_prompt),
            getattr(self.llm_client, "model", "ollama"),
        )

        from app.llm.sanitize import detect_reasoning_leakage
        from app.core.config import get_settings

        settings = get_settings()
        num_predict = settings.OLLAMA_NUM_PREDICT

        llm_start = time.monotonic()
        logger.info("[LLM GENERATION STARTED] model=%s num_predict=%d", getattr(self.llm_client, "model", "ollama"), num_predict)
        llm_response = await self.llm_client.generate(prompt.system_prompt, prompt.user_prompt, num_predict=num_predict)
        llm_ms = int((time.monotonic() - llm_start) * 1000)
        logger.info("[LLM GENERATION FINISHED] llm_ms=%d", llm_ms)

        # Truncation check (done_reason == "length") -> Retry once with num_predict *= 2
        if getattr(llm_response, "finish_reason", None) == "length":
            logger.warning("[LLM TRUNCATION DETECTED] finish_reason=length. Retrying once with num_predict *= 2 (%d)", num_predict * 2)
            num_predict *= 2
            llm_response = await self.llm_client.generate(prompt.system_prompt, prompt.user_prompt, num_predict=num_predict)

        # Sanitize answer from thinking tags or monologue prefixes
        from app.llm.sanitize import sanitize_response
        validation_start = time.monotonic()
        clean_ans = sanitize_response(llm_response.answer)
        clean_ans_lower = clean_ans.lower().strip()
        if not clean_ans or clean_ans_lower == "i could not find this information in the uploaded documents." or clean_ans_lower == "information not found in document excerpts." or clean_ans_lower == "information not found":
            answer_text = "I could not find this information in the uploaded documents."
        else:
            answer_text = clean_ans
        validation_ms = int((time.monotonic() - validation_start) * 1000)

        sources = _sources_from_prompt(prompt)

        # Append human-readable source footnotes when answer is factual
        if answer_text and answer_text != "I could not find this information in the uploaded documents." and sources:
            sources_lines = ["\n\n---\n**Sources:**"]
            for idx, src in enumerate(sources, 1):
                doc_name = src.document_title or "Document"
                sec = src.section_title or "General"
                pg = src.page_number or 1
                sources_lines.append(f"{idx}. *{doc_name}* — Page {pg}, {sec}")
            answer_text += "\n".join(sources_lines)

        total_ms = int((time.monotonic() - start_mono) * 1000)
        user_hash = hashlib.sha256(str(chat_session.user_id).encode()).hexdigest()[:12]
        trace_payload = {
            "trace_id": str(uuid.uuid4()),
            "session_id": str(session_id),
            "user_id_hash": user_hash,
            "intent": route.value,
            "question": question.strip()[:80],
            "chunks_retrieved": len(retrieved_chunks),
            "top_similarity": round(top_sim, 4),
            "total_ms": total_ms,
            "retrieval_ms": retrieval_ms,
            "prompt_ms": prompt_ms,
            "llm_ms": llm_ms,
            "prompt_tokens": llm_response.token_usage.prompt_tokens if llm_response.token_usage else 0,
            "completion_tokens": llm_response.token_usage.completion_tokens if llm_response.token_usage else 0,
            "model": llm_response.model_name,
            "status": "SUCCESS",
        }
        logger.info("[ENTERPRISE RAG TRACE] %s", json.dumps(trace_payload))

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
            for c in retrieved_chunks
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

        raw_history = await self.messages.list_by_session(session_id, limit=6)
        formatted_history = [
            {"role": m.role.value if hasattr(m.role, "value") else str(m.role), "content": m.content}
            for m in raw_history
            if m.id != user_message.id
        ]

        prompt = self.prompt_builder.build(
            question.strip(),
            retrieved_chunks,
            chat_history=formatted_history,
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
    ) -> RAGResponse:
        """Handle DOCUMENT_LIST / WEB / CALCULATOR / DIRECT without vector retrieval."""
        if route == Route.DOCUMENT_LIST:
            chat_session = await self.sessions.get(session_id)
            return await self._ask_document_list(
                session_id=session_id,
                user_id=chat_session.user_id,
                user_message_id=user_message_id,
                start_mono=start_mono,
            )
        if route == Route.WEB:
            return await self._ask_web(
                session_id=session_id,
                question=question,
                user_message_id=user_message_id,
                start_mono=start_mono,
            )
        if route == Route.CALCULATOR:
            return await self._ask_calculator(
                session_id=session_id,
                question=question,
                user_message_id=user_message_id,
                start_mono=start_mono,
            )
        return await self._ask_direct(
            session_id=session_id,
            question=question,
            user_message_id=user_message_id,
            start_mono=start_mono,
        )

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
            .where(Document.user_id == user_id)
            .where(Document.status.in_([DocumentStatus.READY, DocumentStatus.PROCESSING, DocumentStatus.UPLOADED, "ready", "processing", "uploaded"]))
            .order_by(Document.title)
        )
        titles = list((await self.session.execute(stmt)).scalars().all())

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
    ) -> RAGResponse:
        llm_start = time.monotonic()
        llm_response = await self.llm_client.generate(
            _DIRECT_SYSTEM_PROMPT,
            question,
            num_predict=_DIRECT_NUM_PREDICT,
        )
        llm_ms = int((time.monotonic() - llm_start) * 1000)
        answer_text = (
            sanitize_response(llm_response.answer).strip()
            if llm_response.answer
            else "I could not generate an answer right now."
        )
        if not answer_text:
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
            document_title=getattr(chunk, "document_title", None),
            section_title=getattr(chunk, "section_title", None),
            page_number=getattr(chunk, "page_number", None),
        )
        for chunk in prompt.retrieved_chunks
    ]


def _map_token_usage(token_usage: TokenUsage | None) -> RAGTokenUsage | None:
    if token_usage is None:
        return None
    return RAGTokenUsage(
        prompt_tokens=token_usage.prompt_tokens,
        completion_tokens=token_usage.completion_tokens,
        total_tokens=token_usage.total_tokens,
    )
