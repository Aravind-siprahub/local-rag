import pytest
from app.core.config import get_settings
from app.prompting.templates import format_user_prompt

def test_system_prompt_forbids_reasoning():
    settings = get_settings()
    sys_prompt = settings.SYSTEM_PROMPT.lower()
    
    assert "no reasoning, no commentary, no self-talk" in sys_prompt

def test_user_prompt_forbids_reasoning():
    user_prompt = format_user_prompt("Mock Context", "What port does the frontend use?")
    user_prompt_lower = user_prompt.lower()
    
    assert "if the context contains no relevant facts" in user_prompt_lower
