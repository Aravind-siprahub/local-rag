"""Dedicated regression test suite for IMAGE_ANALYSIS functionality."""
from __future__ import annotations

import uuid
import pytest
from unittest.mock import AsyncMock, patch

from app.models.user import User
from app.models.chat_session import ChatSession
from app.models.document import Document
from app.models.enums import DocumentStatus
from app.rag.service import RAGService, RAGError
from app.rag.intent_router import Route, classify
from app.llm.response import LLMResponse, TokenUsage


@pytest.mark.asyncio
async def test_basic_image_analysis_routes_and_executes(db_session) -> None:
    """Requirement 12, Test 1: Basic image analysis with zero RAG calls."""
    user_id = uuid.uuid4()
    user = User(id=user_id, email=f"user-{user_id}@example.com", is_active=True)
    db_session.add(user)
    
    session_id = uuid.uuid4()
    chat_session = ChatSession(id=session_id, user_id=user_id, title="Image Test Session")
    db_session.add(chat_session)
    await db_session.commit()

    service = RAGService(db_session)
    attachments = [{
        "id": str(uuid.uuid4()),
        "name": "sample_chart.png",
        "file_path": "user/session/sample_chart.png",
        "mime_type": "image/png",
    }]

    with patch("app.storage.get_storage_service") as mock_storage, \
         patch.object(service.retriever, "retrieve", new_callable=AsyncMock) as mock_retrieve, \
         patch.object(service.llm_client, "generate", new_callable=AsyncMock) as mock_generate:

        mock_instance = AsyncMock()
        mock_instance.download_file.return_value = b"sample_image_content_bytes"
        mock_storage.return_value = mock_instance

        mock_generate.return_value = LLMResponse(
            answer="This image shows a bar chart with financial performance metrics.",
            model_name="qwen3:8b",
            token_usage=TokenUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
        )

        response = await service.ask(session_id, "tell about this image", attachments=attachments)

        assert "bar chart" in response.answer
        assert "Information not found in document excerpts." not in response.answer
        assert mock_generate.called
        assert mock_retrieve.call_count == 0  # Requirement 12, Test 3: RAG calls == 0


@pytest.mark.asyncio
async def test_image_mime_types(db_session) -> None:
    """Requirement 12, Test 2: image/png, image/jpeg, image/webp all route to vision analysis."""
    user_id = uuid.uuid4()
    user = User(id=user_id, email=f"user-{user_id}@example.com", is_active=True)
    db_session.add(user)
    
    session_id = uuid.uuid4()
    chat_session = ChatSession(id=session_id, user_id=user_id, title="MIME Test Session")
    db_session.add(chat_session)
    await db_session.commit()

    service = RAGService(db_session)

    mimes = ["image/png", "image/jpeg", "image/webp"]
    for mime_str in mimes:
        attachments = [{
            "id": str(uuid.uuid4()),
            "name": f"test_file.{mime_str.split('/')[-1]}",
            "file_path": f"user/session/file.{mime_str.split('/')[-1]}",
            "mime_type": mime_str,
        }]

        with patch("app.storage.get_storage_service") as mock_storage, \
             patch.object(service.llm_client, "generate", new_callable=AsyncMock) as mock_generate:

            mock_instance = AsyncMock()
            mock_instance.download_file.return_value = b"image_raw_bytes"
            mock_storage.return_value = mock_instance

            mock_generate.return_value = LLMResponse(
                answer="Analysis of image input.",
                model_name="qwen3:8b",
                token_usage=TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )

            res = await service.ask(session_id, "describe this image", attachments=attachments)
            assert res.answer == "Analysis of image input."
            assert mock_generate.called


@pytest.mark.asyncio
async def test_correct_image_bytes_reach_model(db_session) -> None:
    """Requirement 12, Test 4: Vision model receives exact image bytes."""
    user_id = uuid.uuid4()
    user = User(id=user_id, email=f"user-{user_id}@example.com", is_active=True)
    db_session.add(user)
    
    session_id = uuid.uuid4()
    chat_session = ChatSession(id=session_id, user_id=user_id, title="Bytes Test Session")
    db_session.add(chat_session)
    await db_session.commit()

    service = RAGService(db_session)
    raw_payload = b"EXPECTED_IMAGE_BYTES_12345"
    attachments = [{
        "id": str(uuid.uuid4()),
        "name": "photo.jpg",
        "file_path": "path/photo.jpg",
        "mime_type": "image/jpeg",
    }]

    with patch("app.storage.get_storage_service") as mock_storage, \
         patch.object(service.llm_client, "generate", new_callable=AsyncMock) as mock_generate:

        mock_instance = AsyncMock()
        mock_instance.download_file.return_value = raw_payload
        mock_storage.return_value = mock_instance

        mock_generate.return_value = LLMResponse(
            answer="Photo description.",
            model_name="qwen3:8b",
        )

        await service.ask(session_id, "what is shown in this picture", attachments=attachments)
        
        assert mock_generate.called
        kwargs = mock_generate.call_args.kwargs
        assert kwargs.get("images") == [raw_payload]


@pytest.mark.asyncio
async def test_image_loading_failure_returns_image_error(db_session) -> None:
    """Requirement 12, Test 5: Image load failure returns image error and NOT RAG fallback."""
    user_id = uuid.uuid4()
    user = User(id=user_id, email=f"user-{user_id}@example.com", is_active=True)
    db_session.add(user)
    
    session_id = uuid.uuid4()
    chat_session = ChatSession(id=session_id, user_id=user_id, title="Error Test Session")
    db_session.add(chat_session)
    await db_session.commit()

    service = RAGService(db_session)
    attachments = [{
        "id": str(uuid.uuid4()),
        "name": "corrupt.png",
        "file_path": "path/corrupt.png",
        "mime_type": "image/png",
    }]

    with patch("app.storage.get_storage_service") as mock_storage:
        mock_instance = AsyncMock()
        mock_instance.download_file.side_effect = Exception("Storage file not found")
        mock_storage.return_value = mock_instance

        with pytest.raises(RAGError) as exc_info:
            await service.ask(session_id, "tell about this image", attachments=attachments)

        assert "Failed to download image from storage" in str(exc_info.value)
        assert "Information not found in document excerpts." not in str(exc_info.value)


@pytest.mark.asyncio
async def test_existing_document_regression(db_session) -> None:
    """Requirement 12, Test 6: Non-image document queries still use normal document RAG path."""
    user_id = uuid.uuid4()
    user = User(id=user_id, email=f"user-{user_id}@example.com", is_active=True)
    db_session.add(user)
    
    session_id = uuid.uuid4()
    chat_session = ChatSession(id=session_id, user_id=user_id, title="Doc Regression Session")
    db_session.add(chat_session)
    
    doc = Document(id=uuid.uuid4(), user_id=user_id, title="Architecture_Overview.pdf", status=DocumentStatus.READY, storage_path="p1")
    db_session.add(doc)
    await db_session.commit()

    service = RAGService(db_session)

    with patch.object(service.retriever, "retrieve", new_callable=AsyncMock) as mock_retrieve, \
         patch.object(service.llm_client, "generate", new_callable=AsyncMock) as mock_generate:

        mock_retrieve.return_value = []
        mock_generate.return_value = LLMResponse(
            answer="I could not find this information in the uploaded documents.",
            model_name="qwen3:8b",
        )

        response = await service.ask(session_id, "What does section 3 of Architecture_Overview.pdf say?")
        
        assert mock_retrieve.called
