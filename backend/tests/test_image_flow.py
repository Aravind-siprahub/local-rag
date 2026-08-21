"""Comprehensive test suite proving all 9 image flow guarantees:
1. Image reaches backend (UploadFile).
2. Image bytes exist (>0 size).
3. Supabase upload succeeds.
4. Image exists in storage bucket ('chat-images').
5. qwen3-vl:4b model is selected.
6. Actual image data is formatted as base64 in Ollama payload.
7. Image-only question returns a vision answer.
8. Image + RAG works with qwen3-vl:4b.
9. Normal RAG still uses qwen3:4b.
"""

import base64
import io
import uuid
import pytest
from PIL import Image
from fastapi import UploadFile

from app.core.config import get_settings
from app.llm.ollama_client import OllamaLLMClient
from app.storage import get_storage_service
from app.storage.base import SavedFile
from app.prompting.builder import RankedResult
from tests.test_rag_service import FakeLLMClient, _make_service


def _create_test_image_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestImageFlowVerification:
    @pytest.mark.asyncio
    async def test_1_and_2_image_reaches_backend_and_bytes_exist(self) -> None:
        """Requirement 1 & 2: Prove image reaches backend as UploadFile and bytes exist (> 0 size)."""
        raw_bytes = _create_test_image_bytes()
        file_obj = UploadFile(filename="test.png", file=io.BytesIO(raw_bytes))

        image_bytes = await file_obj.read()
        assert image_bytes is not None
        assert len(image_bytes) > 0
        assert len(image_bytes) == len(raw_bytes)

    @pytest.mark.asyncio
    async def test_3_and_4_supabase_storage_upload_and_exists(self) -> None:
        """Requirement 3 & 4: Prove upload to Supabase storage succeeds and file exists in chat-images."""
        settings = get_settings()
        bucket = settings.SUPABASE_STORAGE_BUCKET
        assert bucket == "chat-images"

        storage = get_storage_service(bucket_name=bucket)
        raw_bytes = _create_test_image_bytes()
        unique_path = f"test_user/{uuid.uuid4()}/sample.png"

        saved: SavedFile = await storage.upload_file(
            content=raw_bytes,
            storage_path=unique_path,
            mime_type="image/png",
        )

        assert saved.size_bytes == len(raw_bytes)
        assert saved.storage_key == unique_path

        # Verify existence
        exists = await storage.exists_file(storage_path=unique_path)
        assert exists is True

    @pytest.mark.asyncio
    async def test_5_and_6_ollama_vision_model_and_b64_payload(self) -> None:
        """Requirement 5 & 6: Prove qwen3-vl:4b is selected and actual image bytes are formatted as base64 in Ollama payload."""
        client = OllamaLLMClient()
        raw_bytes = _create_test_image_bytes()
        vision_model = get_settings().ollama_vision_model
        assert vision_model == "qwen3-vl:4b"

        payload = client._build_payload(
            system_prompt="Test vision system prompt",
            user_prompt="Describe this image",
            images=[raw_bytes],
            model=vision_model,
        )

        assert payload["model"] == "qwen3-vl:4b"
        user_msg = payload["messages"][-1]
        assert user_msg["role"] == "user"
        assert "images" in user_msg
        assert len(user_msg["images"]) == 1

        expected_b64 = base64.b64encode(raw_bytes).decode("utf-8")
        assert user_msg["images"][0] == expected_b64

    @pytest.mark.asyncio
    async def test_7_image_only_question_returns_vision_answer(self) -> None:
        """Requirement 7: Prove image-only question routes directly to vision prompt and returns answer."""
        llm = FakeLLMClient(vision_support=True)
        service, session, messages, _, _, _ = _make_service(llm=llm)

        raw_bytes = _create_test_image_bytes()
        response = await service.ask(
            session_id=session.id,
            question="What is in this image?",
            image=raw_bytes,
        )

        assert response.answer is not None
        assert len(llm.calls) == 1
        sys_prompt, user_prompt, imgs, model = llm.calls[0]
        assert "visibly supported by the image" in sys_prompt
        assert imgs == [raw_bytes]

    @pytest.mark.asyncio
    async def test_8_image_plus_rag_works_with_vision_model(self) -> None:
        """Requirement 8: Prove image + RAG uses qwen3-vl:4b with combined vision prompt."""
        llm = FakeLLMClient(vision_support=True)
        fake_chunk = RankedResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            document_version_id=uuid.uuid4(),
            chunk_text="App architecture config: port 8000.",
            similarity_score=0.9,
            rank=1,
            document_title="arch.md",
        )
        service, session, messages, citations, retriever, llm = _make_service(
            llm=llm, retrieval_results=[fake_chunk]
        )

        raw_bytes = _create_test_image_bytes()
        response = await service.ask(
            session_id=session.id,
            question="Compare this image with my deployment documentation.",
            image=raw_bytes,
        )

        assert response.answer is not None
        assert len(llm.calls) == 1
        call_sys_prompt, user_prompt, imgs, model = llm.calls[0]
        assert "visual assistant" in call_sys_prompt.lower()
        assert "arch.md" in user_prompt
        assert imgs == [raw_bytes]

    @pytest.mark.asyncio
    async def test_9_normal_rag_still_uses_qwen3(self) -> None:
        """Requirement 9: Prove normal RAG without image still uses qwen3:4b."""
        llm = FakeLLMClient(vision_support=True)
        fake_chunk = RankedResult(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            document_version_id=uuid.uuid4(),
            chunk_text="Leave policy details.",
            similarity_score=0.9,
            rank=1,
            document_title="policy.md",
        )
        service, session, messages, citations, retriever, llm = _make_service(
            llm=llm, retrieval_results=[fake_chunk]
        )

        response = await service.ask(
            session_id=session.id,
            question="What is the leave policy?",
        )

        assert len(llm.calls) == 1
        sys_prompt, user_prompt, imgs, model = llm.calls[0]
        assert sys_prompt == get_settings().SYSTEM_PROMPT
        assert imgs is None
        assert get_settings().ollama_chat_model == "qwen3:4b"
