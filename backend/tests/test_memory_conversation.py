import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.memory.conversation_memory import ConversationMemory
from app.models.enums import MessageRole


@pytest.mark.asyncio
async def test_conversation_memory_windowing():
    session = AsyncMock()
    conv_mem = ConversationMemory(session)

    # Mock list_by_session returning 5 messages
    m1 = MagicMock(id=uuid.uuid4(), role=MessageRole.USER, content="Hello")
    m2 = MagicMock(id=uuid.uuid4(), role=MessageRole.ASSISTANT, content="Hi there!")
    m3 = MagicMock(id=uuid.uuid4(), role=MessageRole.USER, content="I use Qwen 3 8B.")
    m4 = MagicMock(id=uuid.uuid4(), role=MessageRole.ASSISTANT, content="Great model choice!")
    m5 = MagicMock(id=uuid.uuid4(), role=MessageRole.USER, content="What is RAG?")

    conv_mem._service.list_by_session = AsyncMock(return_value=[m1, m2, m3, m4, m5])

    # Test retrieving with limit=3 excluding m5
    result = await conv_mem.get_recent_messages(
        conversation_id=uuid.uuid4(),
        limit=3,
        exclude_message_id=m5.id,
    )

    assert len(result) == 3
    assert result[0]["content"] == "Hi there!"
    assert result[1]["content"] == "I use Qwen 3 8B."
    assert result[2]["content"] == "Great model choice!"
