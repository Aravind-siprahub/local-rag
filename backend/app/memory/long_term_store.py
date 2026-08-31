"""Long-term memory store — CRUD + semantic retrieval.

Semantic retrieval uses the existing OllamaEmbeddingClient (nomic-embed-text)
to embed the query and all active memories, then ranks by:
    score = 0.6 * cosine_similarity + 0.3 * importance + 0.1 * recency_score

This approach:
- Reuses the existing embedding infrastructure (no new clients).
- Avoids a VECTOR column in the DB (no pgvector dependency for memory).
- Is fast enough for expected memory set sizes (< 1,000 per user).
"""
from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.memory.types import MemoryEntry, MemoryType
from app.models.long_term_memory import LongTermMemory
from app.repositories.long_term_memory_repository import LongTermMemoryRepository

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _recency_score(memory: LongTermMemory, now: datetime) -> float:
    """Return a recency score in [0, 1] — newer memories score higher.

    Decay: score = 1 / (1 + days_old / 30).
    A memory created today scores ~1.0; one from 30 days ago ~0.5.
    """
    ref_time = memory.last_accessed_at or memory.created_at
    if ref_time.tzinfo is None:
        ref_time = ref_time.replace(tzinfo=timezone.utc)
    days = max(0.0, (now - ref_time).total_seconds() / 86400)
    return 1.0 / (1.0 + days / 30.0)


class LongTermMemoryStore:
    """Persistent long-term memory store with semantic retrieval.

    Public interface:
        create(...)            → LongTermMemory (ORM object)
        retrieve(...)          → list[MemoryEntry]  (ranked by relevance)
        update(...)            → LongTermMemory | None
        supersede(...)         → bool
        soft_delete(...)       → bool
        delete_all(...)        → int
        list_all(...)          → list[MemoryEntry]
    """

    def __init__(self, session: AsyncSession) -> None:
        self._repo = LongTermMemoryRepository(session)
        self._session = session

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def create(
        self,
        *,
        user_id: uuid.UUID,
        memory_type: MemoryType,
        content: str,
        importance: float = 0.5,
        confidence: float = 0.5,
        source_conversation_id: uuid.UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LongTermMemory:
        """Persist a new long-term memory and return the ORM object."""
        settings = get_settings()
        if not settings.MEMORY_ENABLED:
            raise RuntimeError("Memory is disabled via MEMORY_ENABLED=false.")

        now = datetime.now(timezone.utc)
        mem = await self._repo.create(
            user_id=user_id,
            memory_type=memory_type.value,
            content=content.strip(),
            importance=max(0.0, min(1.0, importance)),
            confidence=max(0.0, min(1.0, confidence)),
            source_conversation_id=source_conversation_id,
            is_active=True,
            created_at=now,
            updated_at=now,
            metadata_=metadata or {},
        )

        logger.info(
            "[MEMORY STORE] created id=%s user_id=%s type=%s importance=%.2f",
            mem.id,
            user_id,
            memory_type.value,
            importance,
        )
        return mem

    async def update(
        self,
        memory_id: uuid.UUID,
        user_id: uuid.UUID,
        **fields: Any,
    ) -> LongTermMemory | None:
        """Update writable fields on an existing memory."""
        # Remove read-only fields if accidentally passed
        fields.pop("id", None)
        fields.pop("user_id", None)
        fields.pop("created_at", None)

        updated = await self._repo.update_memory(memory_id, user_id, **fields)
        if updated:
            logger.info(
                "[MEMORY STORE] updated id=%s user_id=%s fields=%s",
                memory_id,
                user_id,
                list(fields.keys()),
            )
        return updated

    async def supersede(
        self,
        old_memory_id: uuid.UUID,
        new_memory_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> bool:
        """Mark `old_memory_id` as superseded by `new_memory_id`."""
        ok = await self._repo.supersede(old_memory_id, new_memory_id, user_id)
        if ok:
            logger.info(
                "[MEMORY STORE] superseded old=%s new=%s user=%s",
                old_memory_id,
                new_memory_id,
                user_id,
            )
        return ok

    async def soft_delete(
        self, memory_id: uuid.UUID, user_id: uuid.UUID
    ) -> bool:
        """Soft-delete a specific memory (is_active=False)."""
        ok = await self._repo.soft_delete(memory_id, user_id)
        if ok:
            logger.info(
                "[MEMORY STORE] soft_deleted id=%s user=%s", memory_id, user_id
            )
        return ok

    async def delete_all(self, user_id: uuid.UUID) -> int:
        """Hard-delete ALL memories for a user. Returns count removed."""
        count = await self._repo.delete_all_for_user(user_id)
        logger.info(
            "[MEMORY STORE] delete_all user=%s count=%d", user_id, count
        )
        return count

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def list_all(
        self,
        user_id: uuid.UUID,
        *,
        limit: int = 200,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        """List all active memories for a user (for UI inspection)."""
        rows = await self._repo.list_by_user(user_id, is_active=True, limit=limit, offset=offset)
        return [MemoryEntry.from_orm(r) for r in rows]

    async def list_by_conversation(
        self, user_id: uuid.UUID, conversation_id: uuid.UUID
    ) -> list[MemoryEntry]:
        """List memories sourced from a specific conversation."""
        rows = await self._repo.list_by_conversation(user_id, conversation_id)
        return [MemoryEntry.from_orm(r) for r in rows]

    async def retrieve(
        self,
        user_id: uuid.UUID,
        query: str,
        *,
        top_k: int | None = None,
        min_importance: float | None = None,
        similarity_threshold: float | None = None,
    ) -> list[MemoryEntry]:
        """Retrieve top-k most relevant long-term memories for a query.

        Ranking combines cosine similarity + importance + recency.
        Never injects the entire memory database — bounded by top_k.

        Args:
            user_id: The user whose memories to search.
            query: The current user query text.
            top_k: Max memories to return. Defaults to MEMORY_TOP_K.
            min_importance: Filter out memories below this importance.
            similarity_threshold: Filter out memories below this similarity.

        Returns:
            List of MemoryEntry, ranked best-first.
        """
        settings = get_settings()
        if not settings.MEMORY_ENABLED:
            return []

        effective_top_k = top_k if top_k is not None else settings.MEMORY_TOP_K
        effective_min_imp = (
            min_importance if min_importance is not None else settings.MEMORY_MIN_IMPORTANCE
        )
        effective_sim_thresh = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.MEMORY_SIMILARITY_THRESHOLD
        )

        # Fetch all active memories for the user
        rows = await self._repo.list_by_user(
            user_id, is_active=True, limit=1000
        )
        if not rows:
            return []

        # Filter by minimum importance before doing expensive embedding
        candidates = [r for r in rows if r.importance >= effective_min_imp]
        if not candidates:
            return []

        # Embed the query using the existing embedding client
        query_embedding = await self._embed_text(query)
        if not query_embedding:
            # Embedding unavailable — fall back to importance-only ranking
            ranked = sorted(candidates, key=lambda r: r.importance, reverse=True)
            entries = [MemoryEntry.from_orm(r) for r in ranked[:effective_top_k]]
            logger.warning(
                "[MEMORY RETRIEVE] embedding_unavailable user=%s fallback=importance_rank returned=%d",
                user_id,
                len(entries),
            )
            return entries

        # Embed each candidate and compute composite score
        now = datetime.now(timezone.utc)
        scored: list[tuple[float, LongTermMemory]] = []

        for mem in candidates:
            try:
                mem_embedding = await self._embed_text(mem.content)
            except Exception:
                mem_embedding = []

            sim = _cosine_similarity(query_embedding, mem_embedding)
            if len(candidates) > 10 and sim < effective_sim_thresh and effective_sim_thresh > 0:
                continue  # Below threshold for large memory sets — skip

            recency = _recency_score(mem, now)
            composite = 0.6 * sim + 0.3 * mem.importance + 0.1 * recency
            scored.append((composite, mem))

        # Sort best-first, take top_k
        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:effective_top_k]

        # Build MemoryEntry list with similarity scores attached
        results: list[MemoryEntry] = []
        ids_to_touch: list[uuid.UUID] = []
        for composite_score, mem in top:
            entry = MemoryEntry.from_orm(mem)
            entry.similarity_score = composite_score
            results.append(entry)
            ids_to_touch.append(mem.id)

        # Update last_accessed_at (fire-and-forget — non-critical)
        for mid in ids_to_touch:
            try:
                await self._repo.touch_accessed(mid)
            except Exception:
                pass  # Non-critical

        logger.info(
            "[MEMORY RETRIEVE] user=%s query=%r candidates=%d returned=%d top_k=%d",
            user_id,
            query[:50],
            len(candidates),
            len(results),
            effective_top_k,
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _embed_text(self, text: str) -> list[float]:
        """Embed text using the existing OllamaEmbeddingClient. Returns [] on failure."""
        try:
            from app.embeddings.client import OllamaEmbeddingClient
            client = OllamaEmbeddingClient()
            return await client.embed(text)
        except Exception as exc:
            logger.warning("[MEMORY STORE] embedding_failed text=%r error=%s", text[:50], exc)
            return []
