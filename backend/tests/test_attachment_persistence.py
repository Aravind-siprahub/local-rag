import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.models.enums import MessageRole
from app.rag.service import RAGService

@pytest.mark.asyncio
async def test_attachment_persistence_in_user_message():
    """Verify that multi-file attachments passed to RAGService are saved in user_message.attachments."""
    session = AsyncMock()
    
    # Mock message service
    messages_mock = AsyncMock()
    created_user_msg = MagicMock(id=uuid.uuid4())
    messages_mock.create_message.return_value = created_user_msg
    
    sessions_mock = AsyncMock()
    chat_session_mock = MagicMock(user_id=uuid.uuid4())
    sessions_mock.get.return_value = chat_session_mock

    rag_service = RAGService(
        session=session,
        message_service=messages_mock,
        session_service=sessions_mock,
    )

    test_attachments = [
        {"id": "att-1", "filename": "report.pdf", "mime_type": "application/pdf", "size": 1024},
        {"id": "att-2", "filename": "diagram.png", "mime_type": "image/png", "size": 2048},
    ]

    # Call RAGService.ask with attachments
    try:
        await rag_service.ask(
            session_id=uuid.uuid4(),
            question="Analyze these documents",
            attachments=test_attachments,
        )
    except Exception:
        pass

    assert messages_mock.create_message.called
    user_call_kwargs = messages_mock.create_message.call_args_list[0].kwargs
    assert user_call_kwargs["role"] == MessageRole.USER
    assert user_call_kwargs["attachments"] == test_attachments
