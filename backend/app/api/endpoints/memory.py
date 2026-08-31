"""REST endpoints for long-term memory management.

Follows the project's API design standards:
- User isolation enforced via authentication (`get_current_user` or `resolve_owner_user_id`).
- Proper HTTP status codes (200, 201, 204, 404).
- Standard JSON response models defined in `app.schemas.memory`.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import PaginationParams, get_current_user, get_db, get_long_term_store
from app.memory.long_term_store import LongTermMemoryStore
from app.memory.types import MemoryType
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.memory import (
    MemoryCreate,
    MemoryListResponse,
    MemoryPurgeResponse,
    MemoryResponse,
    MemoryUpdate,
)
from app.services.owner_resolution import resolve_owner_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["Memory"])


async def _resolve_user(
    current_user: User | None,
    x_user_id: str | None,
    session: AsyncSession,
) -> uuid.UUID:
    """Helper to resolve the effective owner user UUID."""
    if current_user is not None and getattr(current_user, "id", None):
        return current_user.id

    parsed_uuid: uuid.UUID | None = None
    if x_user_id and x_user_id.strip() and x_user_id.strip().lower() not in ("undefined", "null", "none"):
        try:
            parsed_uuid = uuid.UUID(x_user_id.strip())
        except ValueError:
            pass

    user_repo = UserRepository(session)
    return await resolve_owner_user_id(parsed_uuid, user_repo)


@router.get(
    "",
    response_model=MemoryListResponse,
    status_code=status.HTTP_200_OK,
    summary="List long-term memories for the current user",
    description="Returns active (or all) long-term memories for the authenticated user.",
)
async def list_memories(
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    memory_type: MemoryType | None = Query(default=None, description="Filter by memory category."),
    is_active: bool = Query(default=True, description="Filter by active status."),
    pagination: PaginationParams = Depends(),
    store: LongTermMemoryStore = Depends(get_long_term_store),
    session: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
) -> MemoryListResponse:
    """Fetch paginated memories for the user."""
    user_id = await _resolve_user(current_user, x_user_id, session)

    all_memories = await store.list_all(user_id, limit=500)
    filtered = [
        m for m in all_memories
        if (is_active is None or m.is_active == is_active)
        and (memory_type is None or m.memory_type == memory_type)
    ]

    total = len(filtered)
    page = filtered[pagination.offset : pagination.offset + pagination.limit]

    items = [
        MemoryResponse(
            id=m.id,
            user_id=m.user_id,
            memory_type=m.memory_type.value,
            content=m.content,
            importance=m.importance,
            confidence=m.confidence,
            is_active=m.is_active,
            created_at=m.created_at,
            updated_at=m.updated_at,
            last_accessed_at=m.last_accessed_at,
            source_conversation_id=m.source_conversation_id,
            metadata=m.metadata,
        )
        for m in page
    ]

    return MemoryListResponse(items=items, total=total)


@router.post(
    "",
    response_model=MemoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Manually create a long-term memory",
    description="Allows users or administrative tools to add a explicit memory fact.",
)
async def create_memory(
    payload: MemoryCreate,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    store: LongTermMemoryStore = Depends(get_long_term_store),
    session: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
) -> MemoryResponse:
    """Manually add a long-term memory."""
    user_id = await _resolve_user(current_user, x_user_id, session)

    mem = await store.create(
        user_id=user_id,
        memory_type=payload.memory_type,
        content=payload.content,
        importance=payload.importance,
        confidence=payload.confidence,
        metadata=payload.metadata,
    )

    return MemoryResponse(
        id=mem.id,
        user_id=mem.user_id,
        memory_type=mem.memory_type,
        content=mem.content,
        importance=mem.importance,
        confidence=mem.confidence,
        is_active=mem.is_active,
        created_at=mem.created_at,
        updated_at=mem.updated_at,
        last_accessed_at=mem.last_accessed_at,
        source_conversation_id=mem.source_conversation_id,
        metadata=mem.metadata_ or {},
    )


@router.delete(
    "",
    response_model=MemoryPurgeResponse,
    status_code=status.HTTP_200_OK,
    summary="Purge ALL memories for the current user",
    description="Permanently deletes all long-term memories for privacy compliance (GDPR right to be forgotten).",
)
async def purge_all_memories(
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    store: LongTermMemoryStore = Depends(get_long_term_store),
    session: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
) -> MemoryPurgeResponse:
    """Hard delete all memories for the current user."""
    user_id = await _resolve_user(current_user, x_user_id, session)

    count = await store.delete_all(user_id)
    return MemoryPurgeResponse(
        deleted_count=count,
        message=f"Successfully purged {count} long-term memories.",
    )


@router.patch(
    "/{memory_id}",
    response_model=MemoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Update a specific memory",
)
async def update_memory(
    memory_id: uuid.UUID,
    payload: MemoryUpdate,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    store: LongTermMemoryStore = Depends(get_long_term_store),
    session: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
) -> MemoryResponse:
    """Update fields on a long-term memory."""
    user_id = await _resolve_user(current_user, x_user_id, session)

    updates = payload.model_dump(exclude_unset=True)
    if "metadata" in updates and updates["metadata"] is not None:
        updates["metadata_"] = updates.pop("metadata")

    updated = await store.update(memory_id, user_id, **updates)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {memory_id} not found.",
        )

    return MemoryResponse(
        id=updated.id,
        user_id=updated.user_id,
        memory_type=updated.memory_type,
        content=updated.content,
        importance=updated.importance,
        confidence=updated.confidence,
        is_active=updated.is_active,
        created_at=updated.created_at,
        updated_at=updated.updated_at,
        last_accessed_at=updated.last_accessed_at,
        source_conversation_id=updated.source_conversation_id,
        metadata=updated.metadata_ or {},
    )


@router.delete(
    "/{memory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete (deactivate) a specific memory",
)
async def delete_memory(
    memory_id: uuid.UUID,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    store: LongTermMemoryStore = Depends(get_long_term_store),
    session: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(get_current_user),
) -> Response:
    """Soft-delete a memory by setting is_active = False."""
    user_id = await _resolve_user(current_user, x_user_id, session)

    ok = await store.soft_delete(memory_id, user_id)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {memory_id} not found.",
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)
