"""Orchestrate retrieval, prompting, and LLM generation for RAG Q&A."""
from __future__ import annotations

import logging
import time
import uuid

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

        retrieval_start = time.monotonic()
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

        # Fallback: if chat_session.user_id has 0 embeddings, try searching without user_id restriction
        if not retrieved_chunks and retrieval_filters.user_id is not None and filters is None:
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

        prompt_start = time.monotonic()
        prompt = self.prompt_builder.build(question.strip(), retrieved_chunks)
        prompt_ms = int((time.monotonic() - prompt_start) * 1000)

        llm_start = time.monotonic()
        llm_response = await self.llm_client.generate(prompt.system_prompt, prompt.user_prompt)
        llm_ms = int((time.monotonic() - llm_start) * 1000)

        total_ms = int((time.monotonic() - start_mono) * 1000)
        logger.info(
            "RAG Pipeline Latency Breakdown: Retrieval=%d ms | Prompt=%d ms | LLM=%d ms | Total=%d ms",
            retrieval_ms,
            prompt_ms,
            llm_ms,
            total_ms,
        )

        token_usage = llm_response.token_usage
        assistant_message = await self.messages.create_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=llm_response.answer,
            model_used=llm_response.model_name,
            prompt_tokens=token_usage.prompt_tokens if token_usage else None,
            completion_tokens=token_usage.completion_tokens if token_usage else None,
            latency_ms=total_ms,
            generation_time_ms=llm_ms,
        )

        sources = _sources_from_prompt(prompt)
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
            answer=llm_response.answer,
            sources=sources,
            token_usage=_map_token_usage(token_usage),
            model=llm_response.model_name,
            processing_time_ms=processing_time_ms,
            user_message_id=user_message.id,
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
        except (RetrievalError, EmbeddingClientError) as exc:
            logger.warning("Retrieval failed; continuing without document context: %s", exc)
            return []

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
