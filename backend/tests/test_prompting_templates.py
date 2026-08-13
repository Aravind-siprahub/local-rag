"""Unit tests for `app.prompting.templates`."""
from app.prompting.templates import format_chunk, format_user_prompt


class TestFormatChunk:
    def test_numbers_chunk(self) -> None:
        result = format_chunk(1, "Revenue grew 12%.")
        assert "Document: Unknown" in result
        assert "Revenue grew 12%." in result
        assert "[Chunk 1]" not in result


class TestFormatUserPrompt:
    def test_includes_numbered_context_and_question(self) -> None:
        context = "---\nDocument: MyDoc\n\nRevenue grew 12%.\n\n---\nDocument: MyDoc\n\nCosts fell 3%."
        prompt = format_user_prompt(context, "What happened to revenue?")

        assert "Retrieved Document Context" in prompt
        assert "Revenue grew 12%." in prompt
        assert "Costs fell 3%." in prompt
        assert "Question:" in prompt
        assert "What happened to revenue?" in prompt

    def test_empty_context_omits_excerpt_section(self) -> None:
        prompt = format_user_prompt("", "What is EBITDA?")

        assert "Retrieved Document Context" in prompt
        assert "No document excerpts were available" in prompt
        assert "Question:" in prompt
        assert "What is EBITDA?" in prompt

    def test_with_context_does_not_raise_missing_headers(self) -> None:
        """Regression: USER_PROMPT_WITH_CONTEXT requires context_header/question_header."""
        prompt = format_user_prompt("some context", "hello?")
        assert "some context" in prompt
        assert "hello?" in prompt
