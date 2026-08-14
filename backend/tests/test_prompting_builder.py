"""Unit tests for `app.prompting.builder`."""
import uuid

import pytest

from app.prompting.builder import PromptBuilder, PromptBuilderError
from app.prompting.templates import format_chunk
from app.retrieval.ranking import RankedResult


def _ranked(text: str, rank: int, score: float = 0.9) -> RankedResult:
    return RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text=text,
        document_id=uuid.uuid4(),
        document_version_id=uuid.uuid4(),
        similarity_score=score,
        rank=rank,
    )


class TestPromptBuilder:
    def test_builds_numbered_context_and_preserves_metadata(self) -> None:
        builder = PromptBuilder(system_prompt="System rules.", max_context_chars=5000)
        chunks = [_ranked("First chunk.", 1), _ranked("Second chunk.", 2)]

        prompt = builder.build("What is the summary?", chunks)

        assert prompt.system_prompt == "System rules."
        assert "First chunk." in prompt.user_prompt
        assert "Second chunk." in prompt.user_prompt
        assert "What is the summary?" in prompt.user_prompt
        assert len(prompt.retrieved_chunks) == 2
        assert prompt.retrieved_chunks[0].context_index == 1
        assert prompt.retrieved_chunks[1].context_index == 2
        assert prompt.retrieved_chunks[0].rank == 1
        assert prompt.retrieved_chunks[0].chunk_text == "First chunk."

    def test_truncates_context_to_max_chars(self) -> None:
        builder = PromptBuilder(max_context_chars=120)
        chunks = [
            _ranked("A" * 40, 1),
            _ranked("B" * 40, 2),
            _ranked("C" * 40, 3),
        ]

        prompt = builder.build("Question?", chunks)

        assert len(prompt.retrieved_chunks) < 3
        context_only = "\n\n".join(
            format_chunk(chunk.context_index, chunk.chunk_text) for chunk in prompt.retrieved_chunks
        )
        assert len(context_only) <= 120

    def test_truncates_single_large_chunk(self) -> None:
        builder = PromptBuilder(max_context_chars=30)
        chunks = [_ranked("X" * 100, 1)]

        prompt = builder.build("Big chunk?", chunks)

        assert len(prompt.retrieved_chunks) == 1
        assert len(prompt.retrieved_chunks[0].chunk_text) < 100
        assert prompt.retrieved_chunks[0].chunk_text.endswith("...")

    def test_empty_retrieval_still_builds_question_prompt(self) -> None:
        builder = PromptBuilder()
        prompt = builder.build("Any updates?", [])

        assert prompt.retrieved_chunks == []
        assert "Any updates?" in prompt.user_prompt
        assert "No document excerpts were available" in prompt.user_prompt

    def test_rejects_empty_question(self) -> None:
        builder = PromptBuilder()
        with pytest.raises(PromptBuilderError, match="empty"):
            builder.build("   ", [_ranked("text", 1)])

    def test_rejects_invalid_max_context_chars(self) -> None:
        builder = PromptBuilder(max_context_chars=0)
        with pytest.raises(PromptBuilderError, match="max_context_chars"):
            builder.build("question", [_ranked("text", 1)])
