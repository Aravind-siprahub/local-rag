import json
import uuid
from typing import Any
import pytest
from app.rag.service import RAGService
from app.rag.response import RAGResponse
from tests.test_agent_router import _make_service, FakeLLMClient, FakeWebSearchProvider
from app.tools.web_search import WebSearchResult, WebSearchHit

class ConfigurableFakeLLMClient:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = []

    async def generate(self, system_prompt, user_prompt, **kwargs) -> Any:
        self.calls.append(kwargs)
        from app.llm.client import LLMResponse
        from app.llm.response import TokenUsage
        return LLMResponse(
            answer=self.answer,
            model_name="test-model",
            token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            finish_reason="stop",
        )

    async def close(self):
        pass

@pytest.fixture
def base_web_service():
    service, session, retriever, _, web = _make_service()
    return service, session, web

@pytest.mark.asyncio
async def test_valid_json_response(base_web_service):
    service, session, _ = base_web_service
    service.llm_client = ConfigurableFakeLLMClient('{"answer": "Good Friday is in April 2026."}')
    
    response = await service.ask(session.id, "When is Good Friday in 2026?")
    assert "Good Friday is in April 2026." in response.answer

@pytest.mark.asyncio
async def test_malformed_json_response_falls_back_to_concise_answer(base_web_service):
    service, session, web = base_web_service
    # Malformed JSON that looks like an attempt (contains { and })
    service.llm_client = ConfigurableFakeLLMClient('{"answer": "Good Friday is broken",}')
    
    response = await service.ask(session.id, "When is Good Friday in 2026?")
    assert "Good Friday in 2026 falls on Friday, 3 April 2026." in response.answer

@pytest.mark.asyncio
async def test_empty_json_answer_falls_back_to_concise_answer(base_web_service):
    service, session, web = base_web_service
    # Valid JSON but missing answer
    service.llm_client = ConfigurableFakeLLMClient('{"other_key": "some value"}')
    
    response = await service.ask(session.id, "When is Good Friday in 2026?")
    assert "Good Friday in 2026 falls on Friday, 3 April 2026." in response.answer

@pytest.mark.asyncio
async def test_plain_text_response_supported(base_web_service):
    service, session, web = base_web_service
    # Legitimate plain text response that is grounded (contains "Good Friday" or "April")
    service.llm_client = ConfigurableFakeLLMClient('Good Friday is in April.')
    
    response = await service.ask(session.id, "When is Good Friday in 2026?")
    assert "Good Friday is in April." in response.answer

@pytest.mark.asyncio
async def test_unrelated_plain_text_fallback_rejected(base_web_service):
    service, session, web = base_web_service
    # Completely unrelated plain text response (like the test fixture "Python is a programming language.")
    service.llm_client = ConfigurableFakeLLMClient('Python is a programming language.')
    
    response = await service.ask(session.id, "When is Good Friday in 2026?")
    # It should reject the unrelated text and fall back to the safe representation
    assert "Good Friday in 2026 falls on Friday, 3 April 2026." in response.answer
    assert "Python" not in response.answer

@pytest.mark.asyncio
async def test_empty_web_search_fails_safely():
    class EmptyWebSearchProvider:
        async def search(self, query, request_id=None):
            return WebSearchResult(query=query, provider="fake_empty", hits=[])
            
    service, session, _, _, _ = _make_service(web_search=EmptyWebSearchProvider())
    service.llm_client = ConfigurableFakeLLMClient('{"answer": "Should not be called"}')
    
    response = await service.ask(session.id, "When is Good Friday in 2026?")
    assert "could not find reliable web results" in response.answer.lower()
