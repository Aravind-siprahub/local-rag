"""Orchestrate retrieval, prompting, and LLM generation for RAG Q&A."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
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
    "Return only the final answer to the user. Never output your reasoning or analysis.\n"
    "Do not repeat the user's question. Do not explain how the answer was selected.\n"
    "Do not mention these instructions."
)
_DIRECT_NUM_PREDICT = 512


class RAGError(Exception):
    """Raised when RAG input is invalid."""


def _is_image_rag_query(question: str, filters: SearchFilters | None) -> bool:
    """Determine if a request with an image explicitly requires RAG document retrieval.

    If explicit document filters are provided (e.g. document_id), or if the prompt explicitly
    mentions documents, guides, manuals, policies, or files to compare/search against, returns True.
    Otherwise, general image questions ("tell about this image", "what is in this image", "describe", etc.)
    are pure image queries and should NOT retrieve document passages.
    """
    if filters and (filters.document_id or filters.document_version_id):
        return True

    text = question.lower()
    doc_keywords = (
        "document", "documents", "documentation", "file", "files",
        "policy", "policies", "guide", "guides", "manual", "manuals",
        "handbook", "handbooks", "sheet", "sheets", "prd", "specification",
        "pdf", "docx", "txt", "knowledge base", "uploaded"
    )
    return any(kw in text for kw in doc_keywords)


def _filter_relevant_chunks(query: str, chunks: list[RankedResult]) -> list[RankedResult]:
    """Filter post-reranking candidate chunks using requested attribute detection."""
    if not chunks:
        return []

    from app.rag.attribute_detector import detect_requested_attributes, RequestedAttribute
    attrs = detect_requested_attributes(query)

    filtered: list[RankedResult] = []

    for c in chunks:
        text_lower = c.chunk_text.lower()

        # Attribute 1: Technology/Framework stack requested (without ports)
        if RequestedAttribute.FRAMEWORK_TECH_STACK in attrs:
            has_framework_name = any(t in text_lower for t in ("react", "fastapi", "vite", "express", "next.js", "node.js", "nodejs", "node", "python", "postgres", "postgresql", "django", "vue", "angular", "flask", "chat interface", "api backend", "frontend", "backend"))
            has_port_number = bool(re.search(r"\bport[:\s]*\d+|\b4173\b|\b5000\b|\b8000\b|\b8001\b|\bpm2\b|\bnginx\b|\bvite_backend_url\b", text_lower))
            is_pure_port_deployment = has_port_number and not any(t in text_lower for t in ("react", "fastapi", "vite", "express", "next.js", "node.js", "nodejs", "node", "chat interface", "django", "vue", "angular"))
            if is_pure_port_deployment:
                logger.info("[RELEVANCE FILTER] Dropped pure port/deployment chunk '%s' for tech-stack query", getattr(c, 'document_title', '?'))
                continue

        # Attribute 2: Port/Networking requested (without tech stack)
        elif RequestedAttribute.PORT_NETWORKING in attrs:
            has_port_info = any(p in text_lower for p in ("port", "4173", "5000", "8000", "80", "443", "listening"))
            if not has_port_info:
                logger.info("[RELEVANCE FILTER] Dropped non-port chunk '%s' for port query", getattr(c, 'document_title', '?'))
                continue

        filtered.append(c)

    return filtered


class RAGService:
    """End-to-end RAG pipeline: retrieve -> prompt -> generate -> persist.

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
        from app.llm.ollama_client import get_global_ollama_client
        self.llm_client = llm_client or get_global_ollama_client()
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
        image: bytes | None = None,
        image_name: str | None = None,
        image_mime: str | None = None,
        image_storage_path: str | None = None,
        image_size: int | None = None,
    ) -> RAGResponse:
        """Run the full RAG flow for one user question in a chat session."""
        if not question or not question.strip():
            if image or image_storage_path:
                question = "Describe this image."
            else:
                raise RAGError("Question must not be empty.")

        if image or image_storage_path:
            vision_model = get_settings().ollama_vision_model
            supports_vision_fn = getattr(self.llm_client, "supports_vision", None)
            logger.info('[VISION GATE] checking vision capability model=%s has_fn=%s', vision_model, supports_vision_fn is not None)
            has_vision = await supports_vision_fn(model=vision_model) if supports_vision_fn else False
            logger.info('[VISION GATE] vision_check_result=%s model=%s', has_vision, vision_model)
            if not has_vision:
                raise RAGError(f"Image analysis is unavailable because the configured vision model '{vision_model}' does not support vision.")

        top_sim = 0.0
        req_id = request_id or str(uuid.uuid4())
        chat_session = await self.sessions.get(session_id)
        user_hash = hashlib.sha256(str(chat_session.user_id).encode()).hexdigest()[:12]
        start_mono = time.monotonic()

        from app.rag.query_understanding import extract_query_intent
        intent = extract_query_intent(question)

        attachments = None
        if image_storage_path:
            attachments = [{
                "storage_path": image_storage_path,
                "bucket": get_settings().SUPABASE_STORAGE_BUCKET,
                "mime_type": image_mime or "application/octet-stream",
                "filename": image_name or "upload.png",
                "size": image_size or 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }]
        elif image:
            attachments = [{
                "id": str(uuid.uuid4()),
                "mime_type": image_mime or "image/png",
                "filename": image_name or "upload.png",
                "size": len(image),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }]

        user_message = await self.messages.create_message(
            session_id=session_id,
            role=MessageRole.USER,
            content=question.strip(),
            attachments=attachments,
        )

        if image_storage_path and not image:
            from app.storage import get_storage_service
            storage = get_storage_service(bucket_name=get_settings().SUPABASE_STORAGE_BUCKET)
            try:
                image = await storage.download_file(storage_path=image_storage_path)
                logger.info(
                    '[IMAGE] image_bytes_loaded storage_path=%s size=%d',
                    image_storage_path, len(image)
                )
            except Exception as e:
                logger.error("[IMAGE] image_bytes_load_failed storage_path=%s error=%s", image_storage_path, e)
                raise RAGError(f"Failed to download image from storage: {e}")

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
            request_id=req_id,
        )

        if image or image_storage_path:
            if not _is_image_rag_query(question.strip(), filters):
                route = Route.DIRECT
            else:
                route = Route.DOCUMENT_QA
        elif filters and (filters.document_id or filters.document_version_id):
            route = Route.DOCUMENT_QA

        logger.info(
            "[BACKEND REQUEST RECEIVED] request_id=%s question=%s session_id=%s user_id_hash=%s route=%s",
            req_id, question.strip()[:80], session_id, user_hash, route.value
        )

        # Execute through production Agent Orchestrator
        settings = get_settings()
        from app.agent.orchestrator import AgentOrchestrator
        orchestrator = AgentOrchestrator(self.session)
        agent_state = await orchestrator.run(
            query=question.strip(),
            session_id=session_id,
            user_id=chat_session.user_id,
            document_id=filters.document_id if filters else None,
            document_version_id=filters.document_version_id if filters else None,
            image_bytes=image,
            image_name=image_name,
            request_id=req_id,
            document_titles=document_titles,
        )

        answer_text = agent_state.final_answer or "I could not find this information in the uploaded documents."

        # Citation processing from agent state retrieved documents
        effective_threshold = similarity_threshold if similarity_threshold is not None else settings.SIMILARITY_THRESHOLD
        if answer_text == "I could not find this information in the uploaded documents.":
            sources = []
        else:
            raw_sources = _sources_from_chunks(agent_state.retrieved_documents)
            sources = _validate_and_deduplicate_sources(
                raw_sources,
                effective_threshold,
                max_sources=getattr(settings, "FINAL_CONTEXT", 3),
            )

        total_ms = int((time.monotonic() - start_mono) * 1000)

        assistant_message = await self.messages.create_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=answer_text,
            model_used=agent_state.selected_model or getattr(self.llm_client, "model", "ollama"),
            latency_ms=total_ms,
            generation_time_ms=agent_state.metrics.llm_generation_time_ms,
        )

        if sources:
            try:
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
            except Exception as cit_exc:
                logger.warning("[CITATIONS CREATE FAILED] %s", cit_exc)

        # Update Working Memory Summary on ChatSession
        try:
            from app.rag.memory_summarizer import summarize_session_history
            history_dicts = agent_state.conversation_context + [{"role": "user", "content": question}, {"role": "assistant", "content": answer_text}]
            new_summary = summarize_session_history(
                history_dicts,
                existing_summary=getattr(chat_session, "working_memory_summary", None),
            )
            chat_session.working_memory_summary = new_summary
            if self.session is not None:
                await self.session.commit()
        except Exception as mem_exc:
            logger.warning("[WORKING MEMORY UPDATE FAILED] %s", mem_exc)

        return RAGResponse(
            answer=answer_text,
            sources=sources,
            token_usage=RAGTokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model=agent_state.selected_model or getattr(self.llm_client, "model", "ollama"),
            processing_time_ms=total_ms,
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
        image: bytes | None = None,
        image_name: str | None = None,
        image_mime: str | None = None,
        image_storage_path: str | None = None,
        image_size: int | None = None,
    ):
        """Run RAG pipeline and yield Server-Sent Events (SSE) formatting for real-time streaming."""
        import json

        if not question or not question.strip():
            if image or image_storage_path:
                question = "Describe this image."
            else:
                raise RAGError("Question must not be empty.")

        if image or image_storage_path:
            vision_model = get_settings().ollama_vision_model
            supports_vision_fn = getattr(self.llm_client, "supports_vision", None)
            logger.info('[VISION GATE stream] checking vision capability model=%s has_fn=%s', vision_model, supports_vision_fn is not None)
            has_vision = await supports_vision_fn(model=vision_model) if supports_vision_fn else False
            logger.info('[VISION GATE stream] vision_check_result=%s model=%s', has_vision, vision_model)
            if not has_vision:
                raise RAGError(f"Image analysis is unavailable because the configured vision model '{vision_model}' does not support vision.")

        chat_session = await self.sessions.get(session_id)
        start_mono = time.monotonic()

        attachments = None
        if image_storage_path:
            attachments = [{
                "storage_path": image_storage_path,
                "bucket": get_settings().SUPABASE_STORAGE_BUCKET,
                "mime_type": image_mime or "application/octet-stream",
                "filename": image_name or "upload.png",
                "size": image_size or 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }]
        elif image:
            attachments = [{
                "id": str(uuid.uuid4()),
                "mime_type": image_mime or "image/png",
                "filename": image_name or "upload.png",
                "size": len(image),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }]

        user_message = await self.messages.create_message(
            session_id=session_id,
            role=MessageRole.USER,
            content=question.strip(),
            attachments=attachments,
        )

        if image_storage_path and not image:
            from app.storage import get_storage_service
            storage = get_storage_service(bucket_name=get_settings().SUPABASE_STORAGE_BUCKET)
            try:
                image = await storage.download_file(storage_path=image_storage_path)
                logger.info(
                    '[IMAGE] image_bytes_loaded (stream) storage_path=%s size=%d',
                    image_storage_path, len(image)
                )
            except Exception as e:
                logger.error("[IMAGE] image_bytes_load_failed (stream) storage_path=%s error=%s", image_storage_path, e)
                raise RAGError(f"Failed to download image from storage: {e}")

        from app.rag.intent_router import Route, classify
        from app.rag.query_normalizer import normalize_query
        
        yield f"data: {json.dumps({'type': 'status', 'message': 'Understanding query...'})}\n\n"
        
        orig_q, norm_q, ret_q = normalize_query(question.strip())
        document_titles, context_texts = await self._load_routing_hints(
            user_id=chat_session.user_id,
            session_id=session_id,
            exclude_message_id=user_message.id,
        )
        req_id = str(uuid.uuid4())
        route = classify(
            question.strip(),
            document_titles=document_titles,
            context_texts=context_texts,
            request_id=req_id,
        )
        if image or image_storage_path:
            if not _is_image_rag_query(question.strip(), filters):
                route = Route.DIRECT
            else:
                route = Route.DOCUMENT_QA
        elif filters and (filters.document_id or filters.document_version_id):
            route = Route.DOCUMENT_QA

        if route != Route.DOCUMENT_QA and route != Route.RAG:
            res = await self._ask_non_rag(
                route=route,
                session_id=session_id,
                question=question.strip(),
                user_message_id=user_message.id,
                start_mono=start_mono,
                request_id=req_id,
                norm_q=norm_q,
                image=image,
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

        retrieval_filters = await self._resolve_entity_filters(chat_session.user_id, question, retrieval_filters)

        yield f"data: {json.dumps({'type': 'status', 'message': 'Searching knowledge base...'})}\n\n"
        
        search_query = ret_q or norm_q or question.strip()
        retrieved_chunks = await self._retrieve_safely(
            search_query,
            filters=retrieval_filters,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        has_doc_filter = (retrieval_filters.document_id is not None) or bool(getattr(retrieval_filters, "document_ids", None))
        if not retrieved_chunks and has_doc_filter:
            logger.info("[RETRIEVAL FALLBACK stream] Scoped retrieval for document(s) returned 0 chunks. Retrying across ALL documents for user.")
            fallback_chunks = await self._retrieve_safely(
                search_query,
                filters=SearchFilters(
                    user_id=retrieval_filters.user_id,
                    document_id=None,
                    document_ids=None,
                    document_version_id=None,
                ),
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            if fallback_chunks:
                retrieved_chunks = fallback_chunks

        if not retrieved_chunks and retrieval_filters.user_id is not None:
            logger.info("[RETRIEVAL FALLBACK stream] 0 chunks found for user_id=%s. Retrying globally without user_id filter.", retrieval_filters.user_id)
            unrestricted_chunks = await self._retrieve_safely(
                search_query,
                filters=SearchFilters(
                    user_id=None,
                    document_id=None,
                    document_ids=None,
                    document_version_id=None,
                ),
                top_k=top_k,
                similarity_threshold=0.10,
            )
            if unrestricted_chunks:
                retrieved_chunks = unrestricted_chunks

        # NOTE: Do NOT re-apply cosine similarity_threshold to cross-encoder-reranked scores.
        # The reranker already selected and re-scored the best candidates; scores are not cosine values.
        settings = get_settings()
        seen_keys: set[uuid.UUID] = set()
        deduped_chunks = []
        for c in retrieved_chunks:
            if c.chunk_id not in seen_keys:
                seen_keys.add(c.chunk_id)
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
                "section_title": getattr(c, "section_title", None),
                "page_number": getattr(c, "page_number", None),
            }
            for c in deduped_chunks
        ]

        yield f"data: {json.dumps({'type': 'meta', 'sources': sources_data, 'user_message_id': str(user_message.id)})}\n\n"

        if not deduped_chunks and not image and not image_storage_path:
            fallback_ans = "I could not find this information in the uploaded documents."
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

        # Relevance guard: if reranker scores are all very low (< 0.05), the retrieved
        # chunks are unlikely to answer the question — abstain rather than hallucinate.
        if deduped_chunks and not image and not image_storage_path:
            top_reranker_score = deduped_chunks[0].similarity_score
            # Relevance guard disabled for highly broken English queries.
            # The LLM will now determine relevance itself.
            if top_reranker_score < -999.0:
                logger.warning(
                    "[STREAM RELEVANCE GUARD] Top reranker score=%.4f is below threshold=0.01. Abstaining.",
                    top_reranker_score,
                )
                fallback_ans = "I couldn't find enough information in the uploaded documents to answer this accurately."
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
        if not deduped_chunks and (image or image_storage_path):
            logger.info("[VISION ONLY STREAM] No document chunks found. Routing image-only stream to _ask_direct.")
            res = await self._ask_direct(
                session_id=session_id,
                question=question.strip(),
                user_message_id=user_message.id,
                start_mono=start_mono,
                norm_q=norm_q,
                image=image,
            )
            yield f"data: {json.dumps({'type': 'token', 'content': res.answer})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'assistant_message_id': str(res.assistant_message_id), 'processing_time_ms': res.processing_time_ms})}\n\n"
            return

        prompt = self.prompt_builder.build(
            search_query,
            deduped_chunks,
            is_vision=bool(image or image_storage_path),
        )

        logger.info("=== RETRIEVED CHUNKS START ===")
        for idx, c in enumerate(deduped_chunks, 1):
            logger.info(
                "[RAG RETRIEVED CHUNK] index=%d chunk_id=%s document_id=%s doc_title=%r section=%r score=%.4f full_text=%r",
                idx, str(c.chunk_id), str(c.document_id), getattr(c, 'document_title', '?'), getattr(c, 'section_title', '?'),
                c.similarity_score, c.chunk_text
            )
        logger.info("=== RETRIEVED CHUNKS END ===")
        
        logger.info("=== FINAL LLM CONTEXT START ===")
        logger.info("[RAG FINAL LLM CONTEXT]\nSYSTEM_PROMPT:\n%s\nUSER_PROMPT:\n%s", prompt.system_prompt, prompt.user_prompt)
        logger.info("=== FINAL LLM CONTEXT END ===")

        yield f"data: {json.dumps({'type': 'status', 'message': 'Generating response...'})}\n\n"
        
        full_answer_chunks: list[str] = []
        ttft_ms = None
        start_generation = time.monotonic()
        from app.rag.intent_router import analyze_complexity
        model_name = get_settings().ollama_vision_model if image else getattr(self.llm_client, "model", None)
        dynamic_max_tokens = analyze_complexity(question, model_name=model_name)
        try:
            if hasattr(self.llm_client, "generate_stream"):
                # Buffer ALL tokens first — do NOT stream raw tokens to client.
                # The model may emit chain-of-thought ("Passage 1 mentions...",
                # "Wait...") that must be sanitized before the user sees it.
                async for token in self.llm_client.generate_stream(
                    prompt.system_prompt,
                    prompt.user_prompt,
                    images=[image] if image else None,
                    model=get_settings().ollama_vision_model if image else None,
                    num_predict=dynamic_max_tokens,
                ):
                    if ttft_ms is None:
                        ttft_ms = int((time.monotonic() - start_generation) * 1000)
                    full_answer_chunks.append(token)
            else:
                resp = await self.llm_client.generate(
                    prompt.system_prompt,
                    prompt.user_prompt,
                    images=[image] if image else None,
                    model=get_settings().ollama_vision_model if image else None,
                    num_predict=dynamic_max_tokens,
                )
                safe = sanitize_response(resp.answer, question=question)
                ttft_ms = int((time.monotonic() - start_generation) * 1000)
                full_answer_chunks.append(safe)
        except Exception as exc:
            logger.error("Streaming error: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

        full_answer = sanitize_response("".join(full_answer_chunks).strip(), question=question) or "I could not find this information in the uploaded documents."

        # Now emit the clean, sanitized answer as a single token
        yield f"data: {json.dumps({'type': 'token', 'content': full_answer})}\n\n"
        total_ms = int((time.monotonic() - start_mono) * 1000)
        token_count = len(full_answer.split())  # rough estimate for telemetry

        assistant_message = await self.messages.create_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=full_answer,
            model_used=getattr(self.llm_client, "model", "ollama"),
            latency_ms=total_ms,
        )

        # Filter citations by similarity threshold — mirrors _validate_and_deduplicate_sources() in ask()
        effective_threshold = similarity_threshold if similarity_threshold is not None else settings.SIMILARITY_THRESHOLD
        valid_sources_data = [s for s in sources_data if s.get("similarity_score", 0.0) >= effective_threshold]
        if valid_sources_data:
            await self.citations.create_citations_for_message(
                assistant_message.id,
                [
                    {
                        "chunk_id": uuid.UUID(s["chunk_id"]) if isinstance(s["chunk_id"], str) else s["chunk_id"],
                        "rank": s["rank"],
                        "similarity_score": s["similarity_score"],
                    }
                    for s in valid_sources_data
                ],
            )

        yield f"data: {json.dumps({'type': 'done', 'assistant_message_id': str(assistant_message.id), 'processing_time_ms': total_ms, 'ttft_ms': ttft_ms, 'token_count': token_count})}\n\n"

    async def _resolve_entity_filters(
        self,
        user_id: uuid.UUID,
        question: str,
        base_filters: SearchFilters,
    ) -> SearchFilters:
        """Filter documents strictly by project or document name mentioned in user query."""
        if (base_filters.document_id or getattr(base_filters, "document_ids", None)) or self.session is None:
            return base_filters

        try:
            import re
            from app.models.document import Document
            from sqlalchemy import select
            stmt_docs = (
                select(Document)
                .where(Document.deleted_at.is_(None))
                .where(Document.user_id == user_id)
            )
            all_docs = list((await self.session.execute(stmt_docs)).scalars().all())

            q_lower = question.strip().lower()
            q_clean = re.sub(r"[^\w\s]", "", q_lower)

            matched_doc_ids: list[uuid.UUID] = []

            for d in all_docs:
                d_title_lower = d.title.lower()
                stem = d_title_lower.rsplit(".", 1)[0] if "." in d_title_lower else d_title_lower
                clean_stem = re.sub(r"[_\-]+", " ", stem).strip()
                core_words = [w for w in clean_stem.split() if w not in {"prd", "guide", "deployment", "staging", "combined", "summary", "overview", "doc", "docx", "v1", "v2", "final", "draft"}]
                core_phrase = " ".join(core_words).strip()

                is_match = False
                if "talk to my data" in q_clean:
                    if any(k in d_title_lower for k in ("talk", "data", "ttmd")):
                        is_match = True
                elif any(s in q_clean for s in ("sipraone", "siprahub", "sipra one", "sipra hub", "sipra")):
                    if any(s in d_title_lower for s in ("sipraone", "siprahub", "sipra")):
                        is_match = True
                elif d_title_lower in q_lower or clean_stem in q_clean:
                    is_match = True
                elif core_phrase and len(core_phrase) >= 3 and core_phrase in q_clean:
                    is_match = True

                if is_match:
                    logger.info("[PROJECT/DOCUMENT ENTITY DETECTED] Question references document '%s' (%s)", d.title, d.id)
                    matched_doc_ids.append(d.id)

            if matched_doc_ids:
                return SearchFilters(
                    user_id=base_filters.user_id,
                    document_id=matched_doc_ids[0] if len(matched_doc_ids) == 1 else None,
                    document_ids=tuple(matched_doc_ids),
                    document_version_id=base_filters.document_version_id,
                    search_mode=base_filters.search_mode,
                )
        except Exception as d_exc:
            logger.warning("[PROJECT ENTITY MATCH FAILED] %s", d_exc)

        return base_filters

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
        image: bytes | None = None,
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
        elif route == Route.CALCULATOR:
            res = await self._ask_calculator(
                session_id=session_id,
                question=question,
                user_message_id=user_message_id,
                start_mono=start_mono,
            )
        elif route == Route.WEB:
            res = await self._ask_web(
                session_id=session_id,
                question=question,
                user_message_id=user_message_id,
                start_mono=start_mono,
                request_id=request_id,
            )
        elif route in (Route.GENERAL_KNOWLEDGE, Route.GENERIC_CHAT, Route.DIRECT):
            res = await self._ask_general_knowledge(
                session_id=session_id,
                question=question,
                user_message_id=user_message_id,
                start_mono=start_mono,
                norm_q=norm_q,
                image=image,
            )
        else:
            res = await self._ask_direct(
                session_id=session_id,
                question=question,
                user_message_id=user_message_id,
                start_mono=start_mono,
                norm_q=norm_q,
                image=image,
            )

        total_ms = int((time.monotonic() - start_mono) * 1000)
        status = "SUCCESS"
        error_type = None
        if res.model.startswith("web-search:error:"):
            status = "ERROR"
            error_type = res.model.split(":", 2)[2]
            
        await self._log_structured_trace(
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
            status=status,
            error_type=error_type,
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

    async def _log_structured_trace(
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
        retrieved_chunk_ids: list[str] | None = None,
        retrieved_doc_ids: list[str] | None = None,
        doc_version_ids: list[str] | None = None,
        similarity_scores: list[float] | None = None,
    ) -> None:
        trace_payload = {
            "request_id": request_id,
            "user_id_hash": user_hash,
            "conversation_id": str(session_id),
            "original_query": orig_q[:100],
            "normalized_query": norm_q[:100],
            "intent": route.value,
            "route": route.value,
            "provider": "duckduckgo" if route == Route.GENERAL_KNOWLEDGE else "none",
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

        if self.session:
            try:
                from app.services.trace_service import TraceStore
                trace_store = TraceStore(self.session)
                await trace_store.save_trace_safely(
                    request_id=request_id,
                    session_id=session_id,
                    original_query=orig_q,
                    normalized_query=norm_q,
                    detected_intent=route.name,
                    selected_route=route.value,
                    retrieval_duration_ms=retrieval_ms,
                    retrieved_chunk_ids=retrieved_chunk_ids or [],
                    retrieved_document_ids=retrieved_doc_ids or [],
                    document_version_ids=doc_version_ids or [],
                    similarity_scores=similarity_scores or [top_similarity],
                    llm_duration_ms=llm_ms,
                    total_duration_ms=total_ms,
                    error_type=error_type,
                    status=status,
                )
            except Exception as exc:
                logger.warning("[TRACE PERSISTENCE ERROR] request_id=%s: %s", request_id, exc)

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
        request_id: str | None = None,
    ) -> RAGResponse:
        """Search the web and synthesize a clean answer using the LLM."""
        provider_name = getattr(self.web_search, "provider", "duckduckgo")
        req_id = request_id or "N/A"
        try:
            # Build search query for provider (stripping web search prefixes)
            search_query = question.strip()
            search_query_clean = re.sub(r"(?i)^(?:search\s+the\s+web\s+for|search\s+web\s+for|search\s+for|web\s+search|google|browse)\s+", "", search_query).strip()

            logger.info("[WEB SEARCH TRIGGERED] request_id=%s orig_question=%r search_query=%r", req_id, question, search_query_clean)
            result = await self.web_search.search(search_query_clean or question, request_id=req_id)
            total_ms = int((time.monotonic() - start_mono) * 1000)

            if result.hits:
                logger.info(
                    "[WEB SEARCH SUCCESS] request_id=%s provider=%s query=%r hits_count=%d urls=%s",
                    req_id,
                    result.provider,
                    question,
                    len(result.hits),
                    [getattr(h, "url", "") for h in result.hits[:3]],
                )

                # Format web snippets with title, snippet, and URL
                snippets_list = []
                for h in result.hits[:5]:
                    title = getattr(h, "title", "") or "Web Result"
                    url = getattr(h, "url", "")
                    snippet = h.snippet.strip()
                    if snippet:
                        item_str = f"Source: {title} ({url})\nSnippet: {snippet}" if url else f"Snippet: {snippet}"
                        snippets_list.append(item_str)

                web_context = "\n\n".join(snippets_list)
                web_system_prompt = get_settings().WEB_SEARCH_SYSTEM_PROMPT

                web_user_prompt = (
                    f"Web search results:\n\n{web_context}\n\n"
                    f"User question:\n{question}"
                )

                logger.info("[WEB SEARCH PROMPT SENT] request_id=%s system_prompt=%r user_prompt=%r", req_id, web_system_prompt[:100], web_user_prompt[:150])

                try:
                    llm_resp = await self.llm_client.generate(
                        web_system_prompt,
                        web_user_prompt,
                        num_predict=512,
                        temperature=0.2,
                    )
                    raw_text = llm_resp.answer.strip()
                    logger.info("[WEB SEARCH LLM RAW RESPONSE] request_id=%s %r", req_id, raw_text)
                    answer_text = sanitize_response(raw_text, question=question).strip()

                    if not answer_text or answer_text == "I could not generate an answer right now.":
                        answer_text = result.concise_answer()
                except Exception as llm_exc:
                    logger.error("[WEB SEARCH LLM ERROR] request_id=%s failed to generate answer using LLM: %s", req_id, llm_exc, exc_info=True)
                    answer_text = result.concise_answer()
            else:
                logger.warning("[WEB SEARCH EMPTY] request_id=%s provider=%s query=%r empty results", req_id, provider_name, question)
                answer_text = "I could not find reliable web results for that question right now."
                raise WebSearchError("Web search yielded no results. Please try again.")
            
            model_name = f"web-search:{result.provider}"

        except WebSearchError as exc:
            msg = str(exc)
            if "timeout" in msg.lower():
                error_type = "timeout"
                answer_text = "Web search timed out. Please try again shortly."
            elif "no results" in msg.lower() or "yielded no results" in msg.lower():
                error_type = "empty_results"
                answer_text = "I could not find reliable web results for that question right now."
            elif "unavailable" in msg.lower() or "http" in msg.lower():
                error_type = "provider_unavailable"
                answer_text = "Web search is temporarily unavailable. Please try again shortly."
            elif "parse" in msg.lower():
                error_type = "parser_failure"
                answer_text = "Web search failed to parse results. Please try again shortly."
            else:
                error_type = "search_failed"
                answer_text = "Web search failed. Please try again shortly."
                
            logger.error("[WEB SEARCH FAILURE] request_id=%s provider=%s query=%r error_type=%s reason=%r", req_id, provider_name, question, error_type, msg)
            model_name = f"web-search:error:{error_type}"

        except Exception as exc:
            error_type = "unexpected_error"
            logger.error("[WEB SEARCH FAILURE] request_id=%s provider=%s query=%r error_type=%s reason=%r", req_id, provider_name, question, error_type, str(exc), exc_info=True)
            answer_text = "An unexpected error occurred during web search."
            model_name = f"web-search:error:{error_type}"

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
        image: bytes | None = None,
    ) -> RAGResponse:
        llm_start = time.monotonic()
        query_text = norm_q.strip() if (norm_q and norm_q.strip()) else question.strip()
        if image and not query_text.startswith("Question:"):
            query_text = f"Question:\n\n{query_text}"

        if not image:
            answer_text = "I couldn't find enough information in the uploaded documents to answer this accurately."
            llm_ms = 0
            llm_model_name = "guardrail:out-of-bounds"
            token_usage = None
        else:
            direct_sys_prompt = get_settings().VISION_SYSTEM_PROMPT
            num_pred = 1024
            
            llm_response = await self.llm_client.generate(
                direct_sys_prompt,
                query_text,
                num_predict=num_pred,
                images=[image],
                model=get_settings().ollama_vision_model,
            )
            llm_ms = int((time.monotonic() - llm_start) * 1000)
            answer_text = sanitize_response(llm_response.answer).strip()
            llm_model_name = llm_response.model_name
            token_usage = llm_response.token_usage

        # Validate answer — retry once if response is empty, truncated, or contains CoT monologue
        if image and not _is_valid_direct_answer(answer_text):
            logger.warning("[DIRECT ANSWER REJECTED] invalid/truncated answer=%r. Retrying once.", answer_text)
            retry_start = time.monotonic()
            retry_prompt = (
                get_settings().VISION_SYSTEM_PROMPT + "\n\n"
                "Return ONLY the final direct answer. Do not include internal thoughts, commentary, or greetings."
            )
            llm_response = await self.llm_client.generate(
                retry_prompt,
                query_text,
                num_predict=1024,
                images=[image],
                model=get_settings().ollama_vision_model,
            )
            llm_ms += int((time.monotonic() - retry_start) * 1000)
            answer_text = sanitize_response(llm_response.answer).strip()
            llm_model_name = llm_response.model_name
            token_usage = llm_response.token_usage

        if not answer_text or not _is_valid_direct_answer(answer_text):
            answer_text = "I could not generate an answer right now."

        total_ms = int((time.monotonic() - start_mono) * 1000)
        assistant_message = await self.messages.create_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=answer_text,
            model_used=llm_model_name,
            prompt_tokens=token_usage.prompt_tokens if token_usage else None,
            completion_tokens=token_usage.completion_tokens if token_usage else None,
            latency_ms=total_ms,
            generation_time_ms=llm_ms,
        )
        logger.info("[RESPONSE RETURNED] total_ms=%d route=DIRECT", total_ms)
        return RAGResponse(
            answer=answer_text,
            sources=[],
            token_usage=_map_token_usage(token_usage) if token_usage else None,
            model=llm_model_name,
            processing_time_ms=total_ms,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message.id,
        )

    async def _ask_general_knowledge(
        self,
        *,
        session_id: uuid.UUID,
        question: str,
        user_message_id: uuid.UUID,
        start_mono: float,
        norm_q: str | None = None,
        image: bytes | None = None,
    ) -> RAGResponse:
        llm_start = time.monotonic()
        query_text = norm_q.strip() if (norm_q and norm_q.strip()) else question.strip()
        if image and not query_text.startswith("Question:"):
            query_text = f"Question:\n\n{query_text}"

        if image:
            direct_sys_prompt = get_settings().VISION_SYSTEM_PROMPT
        else:
            direct_sys_prompt = get_settings().GENERAL_CHAT_SYSTEM_PROMPT

        llm_response = await self.llm_client.generate(
            direct_sys_prompt,
            query_text,
            num_predict=512,
            images=[image] if image else None,
            model=get_settings().ollama_vision_model if image else None,
        )
        llm_ms = int((time.monotonic() - llm_start) * 1000)
        answer_text = sanitize_response(llm_response.answer, question=question).strip()
        llm_model_name = llm_response.model_name
        token_usage = llm_response.token_usage

        if not answer_text:
            answer_text = "I could not generate an answer right now."

        total_ms = int((time.monotonic() - start_mono) * 1000)
        assistant_message = await self.messages.create_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=answer_text,
            model_used=llm_model_name,
            prompt_tokens=token_usage.prompt_tokens if token_usage else None,
            completion_tokens=token_usage.completion_tokens if token_usage else None,
            latency_ms=total_ms,
            generation_time_ms=llm_ms,
        )
        logger.info("[RESPONSE RETURNED] total_ms=%d route=GENERAL_KNOWLEDGE", total_ms)
        return RAGResponse(
            answer=answer_text,
            sources=[],
            token_usage=_map_token_usage(token_usage) if token_usage else None,
            model=llm_model_name,
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
        if self.session is not None:
            try:
                from app.repositories.document_repository import DocumentRepository
                repo = DocumentRepository(self.session)
                docs = await repo.list_by_user(user_id)
                titles = [d.title for d in docs if d.title]
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


def _sources_from_chunks(chunks: list[RankedResult]) -> list[SourceCitation]:
    return [
        SourceCitation(
            chunk_id=chunk.chunk_id,
            chunk_text=chunk.chunk_text,
            document_id=chunk.document_id,
            document_version_id=chunk.document_version_id,
            similarity_score=chunk.similarity_score,
            rank=getattr(chunk, "rank", 1),
            document_title=getattr(chunk, "document_title", None),
            section_title=getattr(chunk, "section_title", None),
            page_number=getattr(chunk, "page_number", None),
        )
        for chunk in chunks
    ]


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
