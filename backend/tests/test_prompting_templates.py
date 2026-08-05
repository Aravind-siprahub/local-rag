"""Unit tests for `app.prompting.templates`."""
from app.prompting.templates import format_chunk, format_user_prompt


class TestFormatChunk:
    def test_numbers_chunk(self) -> None:
        result = format_chunk(1, "Revenue grew 12%.")
        assert result == "[Chunk 1]\nRevenue grew 12%."


class TestFormatUserPrompt:
    def test_includes_numbered_context_and_question(self) -> None:
        context = "[Chunk 1]\nRevenue grew 12%.\n\n[Chunk 2]\nCosts fell 3%."
        prompt = format_user_prompt(context, "What happened to revenue?")

        assert "## Document Excerpts" in prompt
        assert "[Chunk 1]" in prompt
        assert "[Chunk 2]" in prompt
        assert "## Question" in prompt
        assert "What happened to revenue?" in prompt

    def test_empty_context_omits_excerpt_section(self) -> None:
        prompt = format_user_prompt("", "What is EBITDA?")

        assert "## Document Excerpts" not in prompt
        assert "No document excerpts were available" in prompt
        assert "## Question" in prompt
        assert "What is EBITDA?" in prompt
