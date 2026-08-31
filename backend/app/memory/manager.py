"""Memory Manager — high-level orchestrator for the Chat Memory subsystem.

Responsibilities:
1. `before_query`: Retrieve relevant long-term memories → build memory context string.
2. `after_response`: Extract new memories from the conversation turn → persist them.

Performance:
- `after_response` supports async background execution (via asyncio.create_task)
  when MEMORY_ASYNC_EXTRACTION=true, so it never blocks the response path.
- Extraction errors are logged but not propagated — memory failures are
  non-fatal (the app still works without memory).

Observability:
- Structured log lines for every significant event.
- No sensitive memory content is logged.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.memory.context_builder import MemoryContextBuilder
from app.memory.extractor import MemoryExtractor
from app.memory.long_term_store import LongTermMemoryStore
from app.memory.types import ExtractionCandidate, MemoryEntry

logger = logging.getLogger(__name__)

_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


async def shutdown_memory_tasks(timeout: float = 5.0) -> None:
    """Await all active background memory extraction tasks during application shutdown."""
    if not _BACKGROUND_TASKS:
        return

    # Suppress asyncio's default exception handler to avoid 'I/O operation on closed file'
    # errors when background tasks are destroyed after logging streams have closed.
    try:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, _ctx: None)
    except RuntimeError:
        pass

    logger.info("[MEMORY MANAGER] Shutting down %d background task(s)...", len(_BACKGROUND_TASKS))
    tasks = list(_BACKGROUND_TASKS)
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("[MEMORY MANAGER] Timed out waiting for background tasks; cancelling remaining...")
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    _BACKGROUND_TASKS.clear()


class MemoryManager:
    """Orchestrates retrieval and extraction for the Chat Memory subsystem.

    Thread safety: each instance is bound to a single SQLAlchemy AsyncSession
    (request-scoped). Do not share instances across requests.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._store = LongTermMemoryStore(session)
        self._extractor = MemoryExtractor()
        self._builder = MemoryContextBuilder()
        settings = get_settings()
        self._enabled = settings.MEMORY_ENABLED
        self._async_extraction = getattr(settings, "MEMORY_ASYNC_EXTRACTION", True)

    # ------------------------------------------------------------------
    # Before-query step: retrieve relevant memories
    # ------------------------------------------------------------------

    async def before_query(
        self,
        user_id: uuid.UUID,
        query: str,
        *,
        top_k: int | None = None,
    ) -> tuple[str, list[MemoryEntry]]:
        """Retrieve relevant long-term memories and build a context section.

        Returns:
            (memory_context_string, retrieved_memories_list)
            memory_context_string is empty when memory is disabled or no memories found.
        """
        if not self._enabled:
            return "", []

        t0 = time.monotonic()
        try:
            memories = await self._store.retrieve(user_id, query, top_k=top_k)
        except Exception as exc:
            logger.warning(
                "[MEMORY MANAGER] before_query retrieval_failed user=%s error=%s",
                user_id,
                exc,
            )
            return "", []

        retrieval_ms = int((time.monotonic() - t0) * 1000)
        memory_ids = [str(m.id) for m in memories]

        logger.info(
            "[MEMORY MANAGER] before_query user=%s retrieved=%d ids=%s latency_ms=%d",
            user_id,
            len(memories),
            memory_ids,
            retrieval_ms,
        )

        section = self._builder.build_memory_section(memories)
        return section, memories

    # ------------------------------------------------------------------
    # After-response step: extract and persist new memories
    # ------------------------------------------------------------------

    def schedule_extraction(
        self,
        user_id: uuid.UUID,
        question: str,
        answer: str,
        conversation_id: uuid.UUID | None,
        existing_memories: list[MemoryEntry],
    ) -> None:
        """Schedule memory extraction after a response.

        When MEMORY_ASYNC_EXTRACTION=true, extraction runs in a background
        asyncio task and does NOT block the response path.

        When false, callers should await `extract_and_store` directly.
        """
        if not self._enabled:
            return

        if self._async_extraction:
            task = asyncio.create_task(
                self._extract_and_store_safe(
                    user_id=user_id,
                    question=question,
                    answer=answer,
                    conversation_id=conversation_id,
                    existing_memories=existing_memories,
                ),
                name=f"MemoryExtraction-{user_id}",
            )
            _BACKGROUND_TASKS.add(task)
            task.add_done_callback(_BACKGROUND_TASKS.discard)
        else:
            # Synchronous extraction — caller must await this separately
            logger.info(
                "[MEMORY MANAGER] async_extraction=false; call extract_and_store directly"
            )

    async def extract_and_store(
        self,
        user_id: uuid.UUID,
        question: str,
        answer: str,
        conversation_id: uuid.UUID | None,
        existing_memories: list[MemoryEntry],
    ) -> list[Any]:
        """Extract memories from a conversation turn and persist them.

        Returns list of created LongTermMemory ORM objects.
        """
        if not self._enabled:
            return []

        t0 = time.monotonic()
        try:
            candidates = self._extractor.extract(
                user_id=user_id,
                question=question,
                answer=answer,
                conversation_id=conversation_id,
                existing_memories=existing_memories,
            )
        except Exception as exc:
            logger.warning(
                "[MEMORY MANAGER] extraction_failed user=%s error=%s", user_id, exc
            )
            await self._session.rollback()
            return []

        if not candidates:
            await self._session.rollback()
            return []

        created = []
        superseded_count = 0
        rejected_count = 0

        for candidate in candidates:
            try:
                result = await self._persist_candidate(
                    candidate, user_id, conversation_id
                )
                if result is not None:
                    created.append(result)
                    if candidate.conflicts_with:
                        superseded_count += 1
                else:
                    rejected_count += 1
            except Exception as exc:
                logger.warning(
                    "[MEMORY MANAGER] persist_failed user=%s candidate_type=%s error=%s",
                    user_id,
                    candidate.memory_type.value,
                    exc,
                )
                rejected_count += 1

        # Commit in the session-scoped transaction
        try:
            await self._session.commit()
        except Exception as exc:
            logger.warning("[MEMORY MANAGER] commit_failed user=%s error=%s", user_id, exc)
            await self._session.rollback()
            return []

        extraction_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "[MEMORY MANAGER] after_response user=%s created=%d superseded=%d rejected=%d latency_ms=%d",
            user_id,
            len(created),
            superseded_count,
            rejected_count,
            extraction_ms,
        )
        return created

    async def _extract_and_store_safe(self, **kwargs: Any) -> None:
        """Wrapper that creates a fresh database session and swallows all errors.

        Uses a raw AsyncSession without the async context manager so that session
        cleanup is done via the synchronous sync_session.close() path, which avoids
        SQLAlchemy's internal asyncio.shield() call that produces orphan
        'Task was destroyed but it is pending!' noise on event loop teardown.
        """
        from app.db.session import AsyncSessionLocal

        try:
            async with AsyncSessionLocal() as bg_session:
                mgr = MemoryManager(bg_session)
                await mgr.extract_and_store(**kwargs)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("[MEMORY MANAGER] background_extraction_error: %s", exc)

    async def _persist_candidate(
        self,
        candidate: ExtractionCandidate,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID | None,
    ) -> Any | None:
        """Persist one candidate: create new memory (and supersede old if conflict)."""
        settings = get_settings()

        if candidate.importance < settings.MEMORY_MIN_IMPORTANCE:
            logger.info(
                "[MEMORY MANAGER] rejected low_importance=%.2f type=%s",
                candidate.importance,
                candidate.memory_type.value,
            )
            return None

        # Create the new memory
        new_mem = await self._store.create(
            user_id=user_id,
            memory_type=candidate.memory_type,
            content=candidate.content,
            importance=candidate.importance,
            confidence=candidate.confidence,
            source_conversation_id=conversation_id,
            metadata=candidate.metadata,
        )

        # Supersede the conflicting memory (if detected)
        if candidate.conflicts_with:
            try:
                await self._store.supersede(
                    old_memory_id=candidate.conflicts_with,
                    new_memory_id=new_mem.id,
                    user_id=user_id,
                )
            except Exception as exc:
                logger.warning(
                    "[MEMORY MANAGER] supersede_failed old=%s new=%s error=%s",
                    candidate.conflicts_with,
                    new_mem.id,
                    exc,
                )

        return new_mem
