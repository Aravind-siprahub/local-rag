"""Repository for the `long_term_memories` table.

Follows the same patterns as every other repository in the project:
- Extends `BaseRepository` for generic CRUD.
- Adds domain-specific query methods.
- Flushes but never commits — caller owns the transaction boundary.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.long_term_memory import LongTermMemory
from app.repositories.base_repository import BaseRepository


class LongTermMemoryRepository(BaseRepository[LongTermMemory, uuid.UUID]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, LongTermMemory)

    async def list_by_user(
        self,
        user_id: uuid.UUID,
        *,
        is_active: bool = True,
        limit: int = 200,
        offset: int = 0,
    ) -> list[LongTermMemory]:
        """Fetch all active (or all) memories for a user, ordered by creation time desc."""
        stmt = (
            select(LongTermMemory)
            .where(
                LongTermMemory.user_id == user_id,
                LongTermMemory.is_active == is_active,
            )
            .order_by(LongTermMemory.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_by_user_and_type(
        self,
        user_id: uuid.UUID,
        memory_type: str,
        *,
        is_active: bool = True,
        limit: int = 100,
    ) -> list[LongTermMemory]:
        """Fetch memories filtered by type for a user."""
        stmt = (
            select(LongTermMemory)
            .where(
                LongTermMemory.user_id == user_id,
                LongTermMemory.memory_type == memory_type,
                LongTermMemory.is_active == is_active,
            )
            .order_by(LongTermMemory.importance.desc(), LongTermMemory.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_by_conversation(
        self,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
    ) -> list[LongTermMemory]:
        """Fetch all memories sourced from a specific conversation."""
        stmt = (
            select(LongTermMemory)
            .where(
                LongTermMemory.user_id == user_id,
                LongTermMemory.source_conversation_id == conversation_id,
            )
            .order_by(LongTermMemory.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_user(
        self, memory_id: uuid.UUID, user_id: uuid.UUID
    ) -> LongTermMemory | None:
        """Fetch a specific memory only if it belongs to the given user (ownership check)."""
        stmt = select(LongTermMemory).where(
            LongTermMemory.id == memory_id,
            LongTermMemory.user_id == user_id,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def soft_delete(
        self, memory_id: uuid.UUID, user_id: uuid.UUID
    ) -> bool:
        """Soft-delete (deactivate) a memory. Returns True if row was found and updated."""
        mem = await self.get_by_user(memory_id, user_id)
        if mem is None:
            return False
        mem.is_active = False
        mem.updated_at = datetime.now(timezone.utc)
        await self.session.flush()
        return True

    async def delete_all_for_user(self, user_id: uuid.UUID) -> int:
        """Hard-delete ALL memories for a user (privacy wipe). Returns count deleted."""
        stmt = (
            select(LongTermMemory)
            .where(LongTermMemory.user_id == user_id)
        )
        result = await self.session.execute(stmt)
        memories = list(result.scalars().all())
        count = len(memories)
        for mem in memories:
            await self.session.delete(mem)
        await self.session.flush()
        return count

    async def supersede(
        self,
        old_memory_id: uuid.UUID,
        new_memory_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> bool:
        """Mark `old_memory_id` as superseded by `new_memory_id`.

        The old memory is deactivated and linked to the new one for audit purposes.
        Returns True if the old memory was found and updated.
        """
        old = await self.get_by_user(old_memory_id, user_id)
        if old is None:
            return False
        old.is_active = False
        old.superseded_by = new_memory_id
        old.updated_at = datetime.now(timezone.utc)
        await self.session.flush()
        return True

    async def touch_accessed(self, memory_id: uuid.UUID) -> None:
        """Update last_accessed_at to now. Non-critical — errors are swallowed by caller."""
        stmt = (
            update(LongTermMemory)
            .where(LongTermMemory.id == memory_id)
            .values(last_accessed_at=datetime.now(timezone.utc))
        )
        await self.session.execute(stmt)
        await self.session.flush()

    async def update_memory(
        self, memory_id: uuid.UUID, user_id: uuid.UUID, **values: Any
    ) -> LongTermMemory | None:
        """Update fields on an existing memory. Returns the updated object or None."""
        mem = await self.get_by_user(memory_id, user_id)
        if mem is None:
            return None
        values["updated_at"] = datetime.now(timezone.utc)
        for key, val in values.items():
            setattr(mem, key, val)
        await self.session.flush()
        return mem
