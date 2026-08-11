"""Unit tests for LLM response sanitization and reasoning stripping."""
from __future__ import annotations

import json

import httpx
import pytest

from app.llm.ollama_client import OllamaLLMClient, _parse_chat_response
from app.llm.sanitize import ThinkingStreamFilter, is_reasoning_model, sanitize_response, supports_think_parameter

# Build tag literals the same way as production code (avoids source corruption).
_RED_OPEN = "<" + "redacted_thinking" + ">"
_RED_CLOSE = "</" + "redacted_thinking" + ">"


class TestIsReasoningModel:
    @pytest.mark.parametrize(
        "model",
        [
            "qwen3:8b",
            "deepseek-r1:7b",
            "qwq:32b",
            "qwen3-thinking",
            "openai/o1-preview",
        ],
    )
    def test_detects_reasoning_models(self, model: str) -> None:
        assert is_reasoning_model(model) is True

    @pytest.mark.parametrize(
        "model",
        ["llama3.2", "mistral", "nomic-embed-text", "qwen2.5:7b"],
    )
    def test_normal_models_not_flagged(self, model: str) -> None:
        assert is_reasoning_model(model) is False


class TestSupportsThinkParameter:
    @pytest.mark.parametrize("model", ["qwen3:8b", "qwq:32b", "deepseek-r1:7b"])
    def test_think_param_supported(self, model: str) -> None:
        assert supports_think_parameter(model) is True

    @pytest.mark.parametrize("model", ["llama3.2", "mistral", "openai/o1-preview"])
    def test_think_param_not_sent_to_unsupported(self, model: str) -> None:
        assert supports_think_parameter(model) is False


class TestSanitizeResponse:
    def test_strips_redacted_thinking_block(self) -> None:
        raw = (
            f"{_RED_OPEN}Okay, the user is asking about revenue.{_RED_CLOSE}\n\n"
            "Revenue grew **12%** year over year."
        )
        assert sanitize_response(raw) == "Revenue grew **12%** year over year."

    def test_strips_deepseek_think_tags(self) -> None:
        open_tag = "`" + "think" + "`"
        close_tag = "`" + "/" + "think" + "`"
        raw = (
            f"{open_tag}\nThe user wants financial data.\n{close_tag}\n\n"
            "Net profit was $4.2M in Q3."
        )
        assert sanitize_response(raw) == "Net profit was $4.2M in Q3."

    def test_strips_qwen_think_backtick_block(self) -> None:
        open_tag = "`" + "think" + "`"
        close_tag = "\\" + "`" + "think" + "`"
        raw = (
            f"{open_tag}\nLet me analyze the documents.\n{close_tag}\n\n"
            "The policy allows 20 days of leave."
        )
        assert sanitize_response(raw) == "The policy allows 20 days of leave."

    def test_strips_reasoning_prefix_paragraph(self) -> None:
        raw = (
            "Okay, the user has shared a question about authentication.\n\n"
            "Use JWT access tokens with a 15-minute expiry."
        )
        assert "Okay" not in sanitize_response(raw)
        assert "JWT access tokens" in sanitize_response(raw)

    def test_strips_let_me_think_prefix(self) -> None:
        raw = "Let me think about this carefully.\n\nThe answer is 42."
        assert sanitize_response(raw) == "The answer is 42."

    def test_strips_okay_so_i_need_prefix(self) -> None:
        raw = "Okay, so I need to calculate the total from the excerpts.\n\nRevenue was $5M."
        assert "Okay" not in sanitize_response(raw)
        assert "Revenue was $5M." in sanitize_response(raw)

    def test_strips_okay_lets_tackle_prefix(self) -> None:
        raw = (
            "Okay, let's tackle this problem. The user is asking for revenue.\n\n"
            "Revenue was $5M."
        )
        out = sanitize_response(raw)
        assert "Okay" not in out
        assert "Revenue was $5M." in out

    def test_strips_first_ill_prefix(self) -> None:
        raw = "First, I'll look at the revenue figures in chunk 1.\n\nRevenue was $5M."
        out = sanitize_response(raw)
        assert "First, I'll" not in out
        assert "Revenue was $5M." in out

    def test_strips_chained_reasoning_paragraphs(self) -> None:
        raw = (
            "Okay, the user asked about revenue.\n\n"
            "Hmm, this seems like a financial question.\n\n"
            "Revenue was $5M."
        )
        assert sanitize_response(raw) == "Revenue was $5M."

    def test_reasoning_only_response_returns_empty(self) -> None:
        raw = "Okay, let's tackle this problem. The user is asking for the total revenue."
        assert sanitize_response(raw) == ""

    def test_strips_phrasing_commentary_monologue(self) -> None:
        raw = (
            "That phrasing is a bit odd - they probably meant '2 planets or 3 planets'.\n\n"
            "Okay, let me unpack this. The user seems confused about planetary counts.\n\n"
            "Earth is the 3rd planet from the Sun."
        )
        assert sanitize_response(raw) == "Earth is the 3rd planet from the Sun."

    def test_truncated_reasoning_tail_returns_empty(self) -> None:
        raw = (
            "Okay, let me try to figure this out. The user is asking about revenue.\n\n"
            "Wait"
        )
        assert sanitize_response(raw) == ""

    def test_preserves_normal_chat_response(self) -> None:
        answer = "Here are three key points:\n\n1. Speed\n2. Security\n3. Scale"
        assert sanitize_response(answer) == answer

    def test_empty_response(self) -> None:
        assert sanitize_response("") == ""
        assert sanitize_response("   ") == ""
        assert sanitize_response(None) == ""

    def test_strips_based_on_context_prefix(self) -> None:
        raw = "Based on the context provided, Talk to My Data is a platform."
        assert sanitize_response(raw) == "Talk to My Data is a platform."

    def test_strips_from_chunks_above_prefix(self) -> None:
        raw = "From the chunks above, Talk to My Data is a platform."
        assert sanitize_response(raw) == "Talk to My Data is a platform."

    def test_strips_to_answer_this_question_prefix(self) -> None:
        raw = "To answer this question, I need to review the sources.\n\nTalk to My Data is a platform."
        assert sanitize_response(raw) == "Talk to My Data is a platform."

    def test_strips_looking_at_chunk_prefix(self) -> None:
        raw = "Looking at Chunk 1, I can see that Talk to My Data is a platform."
        assert sanitize_response(raw) == "Talk to My Data is a platform."

    def test_strips_looking_at_document_excerpts_bullet_list(self) -> None:
        raw = (
            "Looking at the document excerpts:\n"
            "- Passage 1 is about conversation history in PRD_Talk_to_My_Data.docx section 15\n"
            "- Passage 2 is about Vanna text-to-SQL mechanics in section 18\n"
            "- Passage 3 is critical - it's from PRD_Talk_to_My_Data.docx section 4\n\n"
            "Important: Must not invent anything. The document excerpt clearly states the definition.\n\n"
            "Talk to My Data is an enterprise AI platform."
        )
        assert sanitize_response(raw) == "Talk to My Data is an enterprise AI platform."

    def test_strips_unclosed_close_think_tag_and_leading_monologue(self) -> None:
        raw = (
            "It doesn't directly mention what \"Talk to My Data\" is.\n\n"
            "Passage 2:\nThis passage discusses Vanna...\n\n"
            "Passage 3:\nThis passage has a key statement...\n\n"
            "Let me check if there's more information in the passages...\n\n"
            "I'll write this in concise Markdown format as requested.\n"
            "</think>\n\n"
            "\"Talk to my data\" is actually two different technical problems."
        )
        assert sanitize_response(raw) == '"Talk to my data" is actually two different technical problems.'

    def test_preserves_first_note_that_legitimate_answer(self) -> None:
        raw = "First, note that Talk to My Data is a platform with three key features."
        assert sanitize_response(raw) == "First, note that Talk to My Data is a platform with three key features."

    def test_non_string_input_returns_empty(self) -> None:
        assert sanitize_response(123) == ""  # type: ignore[arg-type]

    def test_only_thinking_returns_empty(self) -> None:
        raw = f"{_RED_OPEN}Internal planning only.{_RED_CLOSE}"
        assert sanitize_response(raw) == ""


class TestThinkingStreamFilter:
    def test_buffers_until_thinking_block_ends(self) -> None:
        filt = ThinkingStreamFilter()
        assert filt.feed(_RED_OPEN) == ""
        assert filt.feed("Okay, the user") == ""
        assert filt.feed(_RED_CLOSE) == ""
        assert filt.feed("\n\nFinal answer.") == "Final answer."

    def test_handles_split_tags_across_tokens(self) -> None:
        filt = ThinkingStreamFilter()
        partial = _RED_OPEN[:13]
        assert filt.feed(partial) == ""
        assert filt.feed(_RED_OPEN[13:] + "hidden") == ""
        assert filt.feed(_RED_CLOSE) == ""
        assert filt.feed("Visible text") == "Visible text"

    def test_flush_emits_safe_tail(self) -> None:
        filt = ThinkingStreamFilter()
        assert filt.feed("Answer without closing tag") == "Answer without closing tag"
        assert filt.flush() == ""

    def test_holds_reasoning_prefix_until_paragraph_break(self) -> None:
        filt = ThinkingStreamFilter()
        assert filt.feed("Okay, the user asked about leave.") == ""
        assert filt.feed("\n\nEmployees receive 20 days.") == "Employees receive 20 days."
        assert filt.flush() == ""

    def test_releases_direct_answer_without_delay(self) -> None:
        filt = ThinkingStreamFilter()
        assert filt.feed("Revenue was $5M in Q3.") == "Revenue was $5M in Q3."


class TestOllamaParseResponse:
    def test_never_returns_thinking_field_as_answer(self) -> None:
        data = {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "Okay, the user is asking about leave policy.",
            },
            "done": True,
        }
        result = _parse_chat_response(data, fallback_model="qwen3:8b")
        assert "Okay" not in result.answer
        assert result.answer == ""

    def test_strips_inline_thinking_from_content(self) -> None:
        data = {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "content": (
                    f"{_RED_OPEN}Hmm, let me check.{_RED_CLOSE}\n\n"
                    "Employees receive 20 days annual leave."
                ),
            },
            "done": True,
        }
        result = _parse_chat_response(data, fallback_model="qwen3:8b")
        assert result.answer == "Employees receive 20 days annual leave."


@pytest.mark.asyncio
async def test_qwen3_sends_think_false() -> None:
    capture: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        capture["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "message": {"role": "assistant", "content": "Direct answer."},
                "done": True,
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaLLMClient(
            base_url="http://ollama.test",
            model="qwen3:8b",
            max_retries=0,
            client=http_client,
        )
        response = await client.generate("System.", "Question?")
        await client.close()

    assert capture["payload"]["think"] is False
    assert response.answer == "Direct answer."


@pytest.mark.asyncio
async def test_stream_filter_end_to_end_like_ollama() -> None:
    """Simulate streamed tokens that include a full thinking block."""
    filt = ThinkingStreamFilter()
    out: list[str] = []
    for token in [_RED_OPEN, "Hidden", _RED_CLOSE, "\n\n", "Safe."]:
        safe = filt.feed(token)
        if safe:
            out.append(safe)
    tail = filt.flush()
    if tail:
        out.append(tail)
    assert "".join(out) == "Safe."
