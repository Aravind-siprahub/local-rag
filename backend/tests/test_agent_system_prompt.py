"""Tests for the upgraded production-grade Local RAG Agent System Prompts."""
import pytest
from app.core.config import get_settings


def test_system_prompt_agent_architecture():
    """Verify SYSTEM_PROMPT contains 16-section SipraHub Local RAG System Prompt rules and quality checks."""
    settings = get_settings()
    sys_prompt = settings.SYSTEM_PROMPT

    assert "document-grounded AI assistant for SipraHub" in sys_prompt
    assert "PRIMARY RULE" in sys_prompt
    assert "UNDERSTAND THE USER'S INTENT" in sys_prompt
    assert "WHOLE-DOCUMENT SUMMARY BEHAVIOR" in sys_prompt
    assert "DO NOT CONFUSE RETRIEVAL FAILURE WITH MISSING INFORMATION" in sys_prompt
    assert "REQUIRED SUMMARY STRUCTURE" in sys_prompt
    assert "TELL ME MORE DETAIL" in sys_prompt
    assert "IMPORTANT NUMBERS AND RULES" in sys_prompt
    assert "ANSWER ALL PARTS OF THE QUESTION" in sys_prompt
    assert "HALLUCINATION PREVENTION" in sys_prompt
    assert "RETRIEVAL-AWARE ANSWERING" in sys_prompt
    assert "FINAL VALIDATION BEFORE ANSWERING" in sys_prompt
    assert "GOLDEN RULE" in sys_prompt


def test_web_search_system_prompt_agent_architecture():
    """Verify WEB_SEARCH_SYSTEM_PROMPT contains decision flow, DuckDuckGo rules, and injection defense."""
    settings = get_settings()
    web_prompt = settings.WEB_SEARCH_SYSTEM_PROMPT

    assert "Local RAG Agent" in web_prompt
    assert "OPERATIONAL AGENT DECISION FLOW" in web_prompt
    assert "DuckDuckGo" in web_prompt
    assert "Do not claim that you cannot access the internet" in web_prompt
    assert "UNTRUSTED DATA" in web_prompt
    assert "SECURITY & PROMPT INJECTION DEFENSE" in web_prompt


def test_general_chat_system_prompt_agent_architecture():
    """Verify GENERAL_CHAT_SYSTEM_PROMPT contains decision flow and security rules."""
    settings = get_settings()
    gen_prompt = settings.GENERAL_CHAT_SYSTEM_PROMPT

    assert "Local RAG Agent" in gen_prompt
    assert "OPERATIONAL AGENT DECISION FLOW" in gen_prompt
    assert "UNTRUSTED DATA" in gen_prompt
