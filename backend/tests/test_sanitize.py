"""Unit tests for app.llm.sanitize."""
import pytest

from app.llm.sanitize import (
    ThinkingStreamFilter,
    detect_reasoning_leakage,
    sanitize_response,
    supports_think_parameter,
)


def test_sanitize_response_standard_think_tags() -> None:
    raw = "<think>\nThinking about greeting the user...\n</think>\nHello! How can I help you today?"
    assert sanitize_response(raw) == "Hello! How can I help you today?"


def test_sanitize_response_unopened_think_tag_leakage_screenshot_case_1() -> None:
    raw = (
        "First, I'll acknowledge their greeting. Since they're just saying hi, I don't need to add much. "
        "A simple \"Hello! How can I assist you today?\" would work. That's one sentence, which fits the requirement. "
        "I should make sure not to add any extra info or analysis. Just the direct response. Let me check: it's friendly, "
        "concise, and answers the greeting appropriately. Yes, that's perfect. No need for more than that. </think>\n"
        "Hello! How can I assist you today?"
    )
    assert sanitize_response(raw) == "Hello! How can I assist you today?"


def test_sanitize_response_unopened_think_tag_leakage_screenshot_case_2() -> None:
    raw = (
        "Hmm, the user just said \"helllo\" - that's a casual greeting with a typo (should be \"hello\"). "
        "I need to respond concisely as per instructions: one or two sentences max, no extra fluff. "
        "Since they're greeting me, I'll acknowledge warmly but keep it minimal. The typo is minor so I won't correct it - "
        "just match their casual tone. \"Hello! How can I help you today?\" works perfectly: friendly, direct, and fits the one-sentence limit. "
        "checks requirements again Yep, no analysis needed here - just the response. User seems to be testing if I'm responsive to simple greetings. </think>\n"
        "Hello! How can I help you today?"
    )
    assert sanitize_response(raw) == "Hello! How can I help you today?"


def test_sanitize_response_reasoning_prefix_without_tags() -> None:
    raw = (
        "First, I'll acknowledge the user's greeting.\n"
        "Hello! How can I assist you today?"
    )
    assert sanitize_response(raw) == "Hello! How can I assist you today?"


def test_detect_reasoning_leakage() -> None:
    assert detect_reasoning_leakage("Here is the answer. </think>") is True
    assert detect_reasoning_leakage("<think>thinking...</think>") is True
    assert detect_reasoning_leakage("Clean answer without tags") is False


def test_supports_think_parameter() -> None:
    assert supports_think_parameter("qwen2.5-coder:7b") is True
    assert supports_think_parameter("deepseek-r1:8b") is True
    assert supports_think_parameter("llama3:8b") is False


def test_thinking_stream_filter_with_unopened_closing_tag() -> None:
    sf = ThinkingStreamFilter()
    tokens = ["First, ", "I'll ", "acknowledge ", "greeting. ", "</think>", "Hello! ", "How ", "can ", "I ", "help?"]
    output = []
    for t in tokens:
        res = sf.feed(t)
        if res:
            output.append(res)
    res_flush = sf.flush()
    if res_flush:
        output.append(res_flush)

    full_output = "".join(output)
    assert "First, I'll acknowledge" not in full_output
    assert "Hello! How can I help?" in full_output


def test_sanitize_response_new_reasoning_monologues() -> None:
    raw = (
        "We are given a user query about SipraOne.\n"
        "Let's extract information from the chunks.\n"
        "Chunk 1 says SipraOne is deployed on Azure VM.\n"
        "We have to be careful.\n"
        "Which one to use?\n"
        "Wait...\n"
        "SipraOne was deployed on an Azure Ubuntu VM."
    )
    assert sanitize_response(raw) == "SipraOne was deployed on an Azure Ubuntu VM."


def test_sanitize_response_monologue_with_answers() -> None:
    raw = (
        "We must answer using only the provided context. Let's extract:\n"
        "Chunk 1 says SipraOne uses Node.js.\n"
        "Chunk 2 says PM2 manages the frontend.\n"
        "From the chunks, we can infer that Nginx is used as the reverse proxy."
    )
    expected = (
        "SipraOne uses Node.js.\n"
        "PM2 manages the frontend.\n"
        "Nginx is used as the reverse proxy."
    )
    assert sanitize_response(raw) == expected


