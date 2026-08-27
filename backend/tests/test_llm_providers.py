"""Unit and integration tests for LLM providers (Ollama, OpenRouter, NVIDIA) and factory."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from app.core.config import Settings, get_settings
from app.llm.client import LLMAPIError, LLMClientError, LLMModelError, LLMTimeoutError
from app.llm.factory import get_llm_client
from app.llm.ollama_client import OllamaLLMClient
from app.llm.openai_client import (
    NvidiaLLMClient,
    OpenAICompatibleLLMClient,
    OpenRouterLLMClient,
)
from app.llm.response import LLMResponse, TokenUsage
from app.rag.service import RAGService


# --- 1. Unit Tests for OpenAICompatibleLLMClient -----------------------------

@pytest.mark.asyncio
async def test_openai_compatible_headers_and_payload_construction():
    """Verify authorization headers, custom provider headers, and request payload structure."""
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Sample answer"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    client = OpenRouterLLMClient(
        api_key="test-openrouter-key",
        model="meta-llama/llama-3.3-70b-instruct",
        client=mock_client,
        prompt_price_per_1m=0.15,
        completion_price_per_1m=0.60,
    )

    resp = await client.generate(
        system_prompt="You are a helpful RAG assistant.",
        user_prompt="Explain SipraLocalRAG",
        num_predict=256,
        temperature=0.2,
    )

    assert resp.answer == "Sample answer"
    assert resp.model_name == "meta-llama/llama-3.3-70b-instruct"
    assert resp.provider == "openrouter"
    assert resp.finish_reason == "stop"
    assert resp.token_usage is not None
    assert resp.token_usage.prompt_tokens == 10
    assert resp.token_usage.completion_tokens == 20
    assert resp.token_usage.total_tokens == 30

    assert resp.generation_time_ms is not None
    assert resp.tokens_per_second is not None
    assert resp.cost_usd == pytest.approx(0.0000135, abs=1e-5)


    mock_client.post.assert_called_once()
    call_args, call_kwargs = mock_client.post.call_args
    assert call_args[0] == "https://openrouter.ai/api/v1/chat/completions"

    headers = call_kwargs["headers"]
    assert headers["Authorization"] == "Bearer test-openrouter-key"
    assert headers["HTTP-Referer"] == "https://sipra.io"
    assert headers["X-Title"] == "Sipra Local RAG"

    json_payload = call_kwargs["json"]
    assert json_payload["model"] == "meta-llama/llama-3.3-70b-instruct"
    assert json_payload["max_tokens"] == 256
    assert json_payload["temperature"] == 0.2
    assert len(json_payload["messages"]) == 2
    assert json_payload["messages"][0] == {"role": "system", "content": "You are a helpful RAG assistant."}
    assert json_payload["messages"][1] == {"role": "user", "content": "Explain SipraLocalRAG"}


@pytest.mark.asyncio
async def test_nvidia_client_configuration_and_override():
    """Verify NVIDIA client configuration and per-request model override."""
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Nemotron answer"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
    }

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post.return_value = mock_response

    client = NvidiaLLMClient(
        api_key="nvapi-test-key",
        model="nvidia/nemotron-4-340b-instruct",
        client=mock_client,
    )

    resp = await client.generate(
        system_prompt="",
        user_prompt="What is Nemotron?",
        model="meta/llama-3.3-70b-instruct",
    )

    assert resp.answer == "Nemotron answer"
    assert resp.model_name == "meta/llama-3.3-70b-instruct"
    assert resp.provider == "nvidia"
    assert resp.cost_usd is None  # Cost unavailable if pricing not set

    call_args, call_kwargs = mock_client.post.call_args
    headers = call_kwargs["headers"]
    assert headers["Authorization"] == "Bearer nvapi-test-key"
    assert call_kwargs["json"]["model"] == "meta/llama-3.3-70b-instruct"


@pytest.mark.asyncio
async def test_missing_api_key_handling():
    """Verify clear error raising when API key is missing."""
    client = OpenRouterLLMClient(api_key="")
    with pytest.raises(LLMClientError) as exc_info:
        await client.generate(system_prompt="", user_prompt="Hello")

    assert "OPENROUTER_API_KEY" in str(exc_info.value)



@pytest.mark.asyncio
async def test_provider_error_handling_and_status_codes():
    """Verify HTTP status code mappings to LLM client errors."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    # 1. 404 Model Error
    mock_client.post.return_value = MagicMock(status_code=404, text="Model not found")
    client_404 = OpenRouterLLMClient(api_key="key", client=mock_client, max_retries=0)
    with pytest.raises(LLMModelError):
        await client_404.generate(system_prompt="", user_prompt="Hi")

    # 2. 401 Auth Error
    mock_client.post.return_value = MagicMock(status_code=401, text="Unauthorized key")
    client_401 = OpenRouterLLMClient(api_key="key", client=mock_client, max_retries=0)
    with pytest.raises(LLMAPIError):
        await client_401.generate(system_prompt="", user_prompt="Hi")

    # 3. Timeout Error
    mock_client.post.side_effect = httpx.TimeoutException("Connection timed out")
    client_timeout = OpenRouterLLMClient(api_key="key", client=mock_client, max_retries=0)
    with pytest.raises(LLMTimeoutError):
        await client_timeout.generate(system_prompt="", user_prompt="Hi")


@pytest.mark.asyncio
async def test_openai_compatible_streaming():
    """Verify SSE line parsing and streaming tokens."""
    sse_lines = [
        "data: {\"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}\n\n",
        "data: {\"choices\": [{\"delta\": {\"content\": \" world!\"}}]}\n\n",
        "data: [DONE]\n\n",
    ]

    async def mock_aiter_lines():
        for line in sse_lines:
            yield line

    mock_stream_ctx = MagicMock()
    mock_stream_ctx.status_code = 200
    mock_stream_ctx.aiter_lines = mock_aiter_lines

    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.stream.return_value.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
    mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=None)

    client = OpenRouterLLMClient(api_key="key", client=mock_client)
    tokens = []
    async for token in client.generate_stream(system_prompt="", user_prompt="Hi"):
        tokens.append(token)

    assert "".join(tokens) == "Hello world!"


# --- 2. Unit Tests for Provider Factory (get_llm_client) ---------------------

def test_factory_missing_api_key_errors():
    """Verify exact error messages when required API keys are missing for openrouter and nvidia providers."""
    with patch("app.llm.factory.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            LLM_PROVIDER="openrouter",
            OPENROUTER_API_KEY="",
            NVIDIA_API_KEY="",
            OPENROUTER_MODEL="google/gemma-4-31b-it:free",
            NVIDIA_MODEL="nvidia/nemotron-4-340b-instruct",
        )

        # 1. Missing OpenRouter Key
        with pytest.raises(LLMClientError) as exc1:
            get_llm_client(provider="openrouter")
        assert "OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter" in str(exc1.value)

        # 2. Missing NVIDIA Key
        with pytest.raises(LLMClientError) as exc2:
            get_llm_client(provider="nvidia")
        assert "NVIDIA_API_KEY is required when LLM_PROVIDER=nvidia" in str(exc2.value)


def test_factory_model_and_url_selection():
    """Verify correct default model and base URL configuration for all providers."""
    from app.llm.openai_client import OmniRouteLLMClient
    with patch("app.llm.factory.get_settings") as mock_settings, patch("app.llm.openai_client.get_settings", mock_settings):
        mock_settings.return_value = MagicMock(
            LLM_PROVIDER="ollama",
            OLLAMA_MODEL="qwen3:8b",
            OPENROUTER_MODEL="google/gemma-4-31b-it:free",
            OPENROUTER_API_KEY="sk-or-v1-testkey",
            OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
            NVIDIA_MODEL="nvidia/nemotron-4-340b-instruct",
            NVIDIA_API_KEY="nvapi-testkey",
            NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1",
            OMNIROUTE_MODEL="omniroute/auto",
            OMNIROUTE_BASE_URL="http://localhost:8000/v1",
            OMNIROUTE_API_KEY=None,
            LLM_TEMPERATURE=0.1,
            LLM_TIMEOUT_SECONDS=120.0,
            LLM_MAX_RETRIES=3,
        )

        # 1. Baseline Ollama
        client_ollama = get_llm_client(provider="ollama")
        assert isinstance(client_ollama, OllamaLLMClient)

        # 2. OpenRouter with Gemma 4 31B Free
        client_or = get_llm_client(provider="openrouter")
        assert isinstance(client_or, OpenRouterLLMClient)
        assert client_or.model == "google/gemma-4-31b-it:free"
        assert client_or.base_url == "https://openrouter.ai/api/v1"

        # 3. NVIDIA NIM with Nemotron
        client_nv = get_llm_client(provider="nvidia")
        assert isinstance(client_nv, NvidiaLLMClient)
        assert client_nv.model == "nvidia/nemotron-4-340b-instruct"
        assert client_nv.base_url == "https://integrate.api.nvidia.com/v1"

        # 4. OmniRoute Gateway
        client_omni = get_llm_client(provider="omniroute")
        assert isinstance(client_omni, OmniRouteLLMClient)
        assert client_omni.model == "omniroute/auto"
        assert client_omni.base_url == "http://localhost:8000/v1"


# --- 3. Integration Tests for RAGService Multi-Provider Integration ---------

@pytest.mark.asyncio
async def test_rag_service_provider_override_integration():
    """Verify RAGService instantiates and routes requests to specified provider."""
    mock_session = AsyncMock()
    mock_openrouter_client = AsyncMock(spec=OpenRouterLLMClient)
    mock_openrouter_client.model = "google/gemma-4-31b-it:free"
    mock_openrouter_client.generate.return_value = LLMResponse(
        answer="Grounded answer from OpenRouter Gemma 4",
        model_name="google/gemma-4-31b-it:free",
        token_usage=TokenUsage(100, 50, 150),
        provider="openrouter",
    )

    with patch("app.llm.factory.get_llm_client", return_value=mock_openrouter_client) as mock_factory:
        rag = RAGService(session=mock_session, provider="openrouter", model="google/gemma-4-31b-it:free")
        mock_factory.assert_called_with(provider="openrouter", model="google/gemma-4-31b-it:free")
        assert rag.llm_client == mock_openrouter_client

