"""Unit tests for OmniRoute LLM client and factory integration."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from app.core.config import get_settings
from app.llm.factory import get_llm_client
from app.llm.openai_client import OmniRouteLLMClient
from app.llm.response import LLMResponse, TokenUsage


@pytest.mark.asyncio
async def test_omniroute_client_generation_and_headers():
    """Verify OmniRoute client request payload, base URL, and headers."""
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "OmniRoute routed response"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 15, "completion_tokens": 30, "total_tokens": 45},
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    client = OmniRouteLLMClient(
        base_url="http://localhost:8000/v1",
        api_key=None,  # Optional for OmniRoute
        model="omniroute/auto",
        client=mock_client,
    )

    resp = await client.generate(
        system_prompt="You are a RAG system assistant.",
        user_prompt="Explain local AI gateway.",
        num_predict=512,
        temperature=0.1,
    )

    assert resp.answer == "OmniRoute routed response"
    assert resp.model_name == "omniroute/auto"
    assert resp.provider == "omniroute"
    assert resp.finish_reason == "stop"
    assert resp.token_usage is not None
    assert resp.token_usage.prompt_tokens == 15
    assert resp.token_usage.completion_tokens == 30
    assert resp.token_usage.total_tokens == 45

    mock_client.post.assert_called_once()
    call_args, call_kwargs = mock_client.post.call_args
    assert call_args[0] == "http://localhost:8000/v1/chat/completions"

    json_payload = call_kwargs["json"]
    assert json_payload["model"] == "omniroute/auto"
    assert json_payload["max_tokens"] == 512
    assert json_payload["temperature"] == 0.1
    assert len(json_payload["messages"]) == 2
    assert json_payload["messages"][0] == {"role": "system", "content": "You are a RAG system assistant."}
    assert json_payload["messages"][1] == {"role": "user", "content": "Explain local AI gateway."}


@pytest.mark.asyncio
async def test_omniroute_with_optional_api_key():
    """Verify OmniRoute client passes authorization header if API key is provided."""
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Keyed answer"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    client = OmniRouteLLMClient(
        base_url="http://localhost:8000/v1",
        api_key="omni-secret-key",
        model="omniroute/qwen",
        client=mock_client,
    )

    resp = await client.generate(system_prompt="", user_prompt="Test query")

    assert resp.answer == "Keyed answer"
    call_args, call_kwargs = mock_client.post.call_args
    headers = call_kwargs["headers"]
    assert headers["Authorization"] == "Bearer omni-secret-key"


def test_factory_omniroute_provider_selection():
    """Verify get_llm_client instantiates OmniRouteLLMClient correctly."""
    with patch("app.llm.factory.get_settings") as mock_settings, patch("app.llm.openai_client.get_settings", mock_settings):
        mock_settings.return_value = MagicMock(
            LLM_PROVIDER="omniroute",
            OMNIROUTE_MODEL="omniroute/auto",
            OMNIROUTE_BASE_URL="http://localhost:8000/v1",
            OMNIROUTE_API_KEY=None,
            LLM_TEMPERATURE=0.1,
            LLM_TIMEOUT_SECONDS=120.0,
            LLM_MAX_RETRIES=3,
        )

        client = get_llm_client(provider="omniroute")
        assert isinstance(client, OmniRouteLLMClient)
        assert client.model == "omniroute/auto"
        assert client.base_url == "http://localhost:8000/v1"
        assert client.provider == "omniroute"


def test_factory_omniroute_model_autodetect():
    """Verify model prefix 'omniroute/' auto-selects omniroute provider."""
    with patch("app.llm.factory.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            LLM_PROVIDER="ollama",
            OMNIROUTE_MODEL="omniroute/auto",
            OMNIROUTE_BASE_URL="http://localhost:8000/v1",
            OMNIROUTE_API_KEY=None,
            LLM_TEMPERATURE=0.1,
            LLM_TIMEOUT_SECONDS=120.0,
            LLM_MAX_RETRIES=3,
        )

        client = get_llm_client(model="omniroute/claude-3-5-sonnet")
        assert isinstance(client, OmniRouteLLMClient)
        assert client.model == "omniroute/claude-3-5-sonnet"
        assert client.provider == "omniroute"


def test_factory_autofast_model_selection():
    """Verify model 'auto/fast' with provider 'omniroute' correctly instantiates OmniRouteLLMClient."""
    with patch("app.llm.factory.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            LLM_PROVIDER="ollama",
            OMNIROUTE_MODEL="omniroute/auto",
            OMNIROUTE_BASE_URL="http://localhost:20128/v1",
            OMNIROUTE_API_KEY=None,
            LLM_TEMPERATURE=0.1,
            LLM_TIMEOUT_SECONDS=120.0,
            LLM_MAX_RETRIES=3,
        )

        client = get_llm_client(provider="omniroute", model="auto/fast")
        assert isinstance(client, OmniRouteLLMClient)
        assert client.model == "auto/fast"
        assert client.provider == "omniroute"

