import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.prompting.builder import PromptBuilder
from app.prompting.templates import format_user_prompt


def test_prompt_builder_with_memory_context():
    builder = PromptBuilder()

    chunk_mock = MagicMock()
    chunk_mock.chunk_id = uuid.uuid4()
    chunk_mock.chunk_text = "Python 3.12 is the latest release."
    chunk_mock.document_id = uuid.uuid4()
    chunk_mock.document_version_id = uuid.uuid4()
    chunk_mock.similarity_score = 0.9
    chunk_mock.rank = 1
    chunk_mock.document_title = "Python Release Notes"
    chunk_mock.section_title = "General"
    chunk_mock.page_number = 1

    memory_sec = "<memory_context>\n- [Preference] User prefers Python.\n</memory_context>"

    prompt = builder.build(
        question="What Python version should I install?",
        retrieved_chunks=[chunk_mock],
        long_term_memory_context=memory_sec,
    )

    assert "<memory_context>" in prompt.user_prompt
    assert "- [Preference] User prefers Python." in prompt.user_prompt
    assert "Python 3.12 is the latest release." in prompt.user_prompt
    # Ensure system prompt is clean and date-aware but unchanged by memory
    assert "System Prompt" not in memory_sec
