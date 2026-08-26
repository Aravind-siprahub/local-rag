"""Tests for the upgraded production-grade Local RAG Agent System Prompts."""
import pytest
from app.core.config import get_settings


def test_system_prompt_agent_architecture():
    """Verify SYSTEM_PROMPT contains decision flow, RAG rules, local tool rules, and injection defense."""
    settings = get_settings()
    sys_prompt = settings.SYSTEM_PROMPT

    assert "Local RAG Agent" in sys_prompt
    assert "OPERATIONAL AGENT DECISION FLOW" in sys_prompt
    assert "CRITICAL GROUNDING & RAG RULES" in sys_prompt
    assert "VERIFICATION & LOCAL TOOL RULES" in sys_prompt
    assert "SECURITY & PROMPT INJECTION DEFENSE" in sys_prompt
    assert "UNTRUSTED DATA" in sys_prompt
    assert "no reasoning, no commentary, no self-talk" in sys_prompt
    assert "multiple values" in sys_prompt
    assert "The requested information is not found in the documents." in sys_prompt


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
