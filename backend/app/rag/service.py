"""Orchestrate retrieval, prompting, and LLM generation for RAG Q&A."""
from __future__ import annotations

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
from app.rag.response import RAGResponse, RAGTokenUsage, SourceCitation
from app.retrieval.ranking import RankedResult
from app.retrieval.retriever import Retriever, RetrievalError
from app.retrieval.search import SearchFilters
from app.services.chat_message_service import ChatMessageService
from app.services.chat_session_service import ChatSessionService
from app.services.citation_service import CitationService
from app.llm.sanitize import sanitize_response

logger = logging.getLogger(__name__)


class RAGError(Exception):
    """Raised when RAG input is invalid."""


class RAGService:
    """End-to-end RAG pipeline: retrieve → prompt → generate → persist.

    Independent of FastAPI — accepts plain strings and UUIDs. Uses existing
    services for chat messages and citations without modifying their
    architecture.
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
    ) -> None:
        self.session = session
        self.retriever = retriever or Retriever(session)
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.llm_client = llm_client or OllamaLLMClient()
        self.messages = message_service or ChatMessageService(session)
        self.citations = citation_service or CitationService(session)
        self.sessions = session_service or ChatSessionService(session)

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

        # Check if user asks for a list of uploaded/available documents
        q_lower = question.strip().lower()
        has_doc_word = any(w in q_lower for w in ["document", "documents", "file", "files", "pdf", "docx"])
        has_list_intent = any(w in q_lower for w in ["list", "show", "what", "which", "available", "uploaded", "have", "exist", "all"])
        is_specific_inquiry = any(w in q_lower for w in ["inside", "content", "summary", "summarize", "about", "detail", "explain", "how", "why"])

        if (has_doc_word and has_list_intent and not is_specific_inquiry) or any(kw in q_lower for kw in ["list all uploaded documents", "list uploaded documents", "what documents are available", "list documents", "show all documents", "list what document u have"]):

            from app.models.document import Document
            from app.models.enums import DocumentStatus
            from sqlalchemy import select

            stmt = (
                select(Document.title)
                .where(Document.deleted_at.is_(None))
                .where(Document.status.in_([DocumentStatus.READY, DocumentStatus.PROCESSING, DocumentStatus.UPLOADED, "ready", "processing", "uploaded"]))
                .order_by(Document.title)
            )
            # Try scoped by user_id first
            scoped_stmt = stmt.where(Document.user_id == chat_session.user_id)
            titles = list((await self.session.execute(scoped_stmt)).scalars().all())
            if not titles:
                # Fallback across knowledge base if user_id yields 0
                titles = list((await self.session.execute(stmt)).scalars().all())

            if titles:
                total_ms = int((time.monotonic() - start_mono) * 1000)
                formatted_list = "The uploaded documents available in the system are:\n" + "\n".join(f"- {t}" for t in titles)
                logger.info("[RAG DOCUMENT LIST BYPASS] Found %d documents, returning direct answer", len(titles))
                assistant_msg = await self.messages.create_message(
                    session_id=session_id,
                    role=MessageRole.ASSISTANT,
                    content=formatted_list,
                    model_used="database-direct",
                    latency_ms=total_ms,
                    generation_time_ms=total_ms,
                )
                logger.info("[RESPONSE RETURNED] total_ms=%d (Document List Direct Query)", total_ms)
                return RAGResponse(
                    answer=formatted_list,
                    sources=[],
                    token_usage=None,
                    model="database-direct",
                    processing_time_ms=total_ms,
                    user_message_id=user_message.id,
                    assistant_message_id=assistant_msg.id,
                )

        retrieval_start = time.monotonic()

        retrieval_filters = filters or SearchFilters()
        if retrieval_filters.user_id is None:
            retrieval_filters = SearchFilters(
                user_id=chat_session.user_id,
                document_id=retrieval_filters.document_id,
                document_version_id=retrieval_filters.document_version_id,
            )

        logger.info("[RAG STAGE 2: RETRIEVAL START] filters=%s top_k=%s similarity_threshold=%s", retrieval_filters, top_k, similarity_threshold)
        retrieved_chunks = await self._retrieve_safely(
            question.strip(),
            filters=retrieval_filters,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        # Fallback: if chat_session.user_id has 0 embeddings, try searching without user_id restriction
        if not retrieved_chunks and retrieval_filters.user_id is not None:
            logger.info("[RAG STAGE 2: RETRIEVAL FALLBACK] No hits for user_id=%s, searching across entire knowledge base", chat_session.user_id)
            fallback_filters = SearchFilters(
                document_id=retrieval_filters.document_id,
                document_version_id=retrieval_filters.document_version_id,
            )

            fallback_chunks = await self._retrieve_safely(
                question.strip(),
                filters=fallback_filters,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            if fallback_chunks:
                logger.info(
                    "Fallback retrieval succeeded: retrieved %d chunks across knowledge base (user_id mismatch bypass)",
                    len(fallback_chunks),
                )
                retrieved_chunks = fallback_chunks

        retrieval_ms = int((time.monotonic() - retrieval_start) * 1000)
        logger.info("[RETRIEVAL FINISHED] retrieval_ms=%d hits=%d", retrieval_ms, len(retrieved_chunks))

        # If no chunks exist after primary and fallback retrieval
        if not retrieved_chunks:
            total_ms = int((time.monotonic() - start_mono) * 1000)
            fallback_answer = "Information not found in document excerpts."
            logger.warning("[NO DOCUMENTS FOUND] 0 chunks retrieved. Returning early without calling LLM.")

            assistant_message = await self.messages.create_message(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=fallback_answer,
                model_used=self.llm_client.model,
                latency_ms=total_ms,
                generation_time_ms=0,
            )
            return RAGResponse(
                answer=fallback_answer,
                sources=[],
                token_usage=RAGTokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                model=self.llm_client.model,
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
        logger.info("[PROMPT BUILT] prompt_ms=%d context_chunks=%d", prompt_ms, len(prompt.retrieved_chunks))

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

        # Reasoning leakage check -> Discard response and Retry ONCE (NO regex hacks)
        if detect_reasoning_leakage(llm_response.answer):
            logger.warning("[REASONING LEAKAGE DETECTED] Response contained unhandled thinking tags. Retrying generation once.")
            llm_response = await self.llm_client.generate(prompt.system_prompt, prompt.user_prompt, num_predict=num_predict)

        validation_start = time.monotonic()
        answer_text = llm_response.answer.strip() if (llm_response.answer and llm_response.answer.strip()) else "Information not found in document excerpts."
        validation_ms = int((time.monotonic() - validation_start) * 1000)

        sources = _sources_from_prompt(prompt)

        # Append human-readable source footnotes when answer is factual
        if answer_text and answer_text != "Information not found in document excerpts." and sources:
            sources_lines = ["\n\n---\n**Sources:**"]
            for idx, src in enumerate(sources, 1):
                doc_name = src.document_title or "Document"
                sec = src.section_title or "General"
                pg = src.page_number or 1
                sources_lines.append(f"{idx}. *{doc_name}* — Page {pg}, {sec}")
            answer_text += "\n".join(sources_lines)

        total_ms = int((time.monotonic() - start_mono) * 1000)

        logger.info("[STAGE PROFILING] total_ms=%d retrieval_ms=%d prompt_ms=%d llm_ms=%d validation_ms=%d", total_ms, retrieval_ms, prompt_ms, llm_ms, validation_ms)
        logger.info("[RESPONSE RETURNED] total_duration_ms=%d", total_ms)

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
                        "chunk_id": s["chunk_id"],
                        "rank": s["rank"],
                        "similarity_score": s["similarity_score"],
                    }
                    for s in sources_data
                ],
            )

        yield f"data: {json.dumps({'type': 'done', 'assistant_message_id': str(assistant_message.id), 'processing_time_ms': total_ms})}\n\n"

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
