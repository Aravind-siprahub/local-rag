import uuid
from datetime import datetime, timezone
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.memory.long_term_store import LongTermMemoryStore, _cosine_similarity, _recency_score
from app.memory.types import MemoryType
from app.models.long_term_memory import LongTermMemory


def test_cosine_similarity():
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _cosine_similarity([], [1.0]) == 0.0
    assert _cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_recency_score():
    now = datetime.now(timezone.utc)
    mem = MagicMock()
    mem.last_accessed_at = now
    mem.created_at = now
    score_now = _recency_score(mem, now)
    assert score_now > 0.99


@pytest.mark.asyncio
async def test_store_create():
    session = AsyncMock()
    store = LongTermMemoryStore(session)

    mock_row = MagicMock(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        memory_type="preference",
        content="User prefers local models.",
        importance=0.8,
        confidence=0.9,
        source_conversation_id=None,
        last_accessed_at=None,
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        metadata_={},
    )
    store._repo.create = AsyncMock(return_value=mock_row)

    created = await store.create(
        user_id=mock_row.user_id,
        memory_type=MemoryType.PREFERENCE,
        content="User prefers local models.",
        importance=0.8,
    )
    assert created.content == "User prefers local models."


@pytest.mark.asyncio
async def test_store_retrieve_importance_fallback():
    session = AsyncMock()
    store = LongTermMemoryStore(session)

    user_id = uuid.uuid4()
    m1 = MagicMock(
        id=uuid.uuid4(),
        user_id=user_id,
        memory_type="preference",
        content="Prefers dark mode",
        importance=0.9,
        confidence=0.8,
        source_conversation_id=None,
        last_accessed_at=None,
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        metadata_={},
    )
    store._repo.list_by_user = AsyncMock(return_value=[m1])

    # Mock embedding failure to test fallback
    with patch.object(store, "_embed_text", new_callable=AsyncMock, return_value=[]):
        results = await store.retrieve(user_id, "dark mode", top_k=5)
        assert len(results) == 1
        assert results[0].content == "Prefers dark mode"
