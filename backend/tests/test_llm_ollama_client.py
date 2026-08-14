"""Unit tests for `app.llm.ollama_client`."""
import json

import httpx
import pytest

from app.llm.client import (
    LLMAPIError,
    LLMClientError,
    LLMModelError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.ollama_client import OllamaLLMClient


def _chat_body(
    *,
    content: str = "Generated answer.",
    model: str = "test-chat-model",
    done_reason: str = "stop",
    prompt_eval_count: int = 12,
    eval_count: int = 8,
) -> dict:
    return {
        "model": model,
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }


def _make_transport(
    *,
    status_code: int = 200,
    body: dict | None = None,
    fail_times: int = 0,
    capture: dict | None = None,
) -> httpx.MockTransport:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        payload = json.loads(request.content)
        if capture is not None:
            capture["payload"] = payload
            capture["attempts"] = capture.get("attempts", 0) + 1
        if fail_times > attempts:
            attempts += 1
            return httpx.Response(503, text="temporary failure")
        assert payload["model"] == "test-chat-model"
        assert payload["stream"] is False
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        return httpx.Response(status_code, json=body or _chat_body())

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_generate_returns_answer_and_metadata() -> None:
    transport = _make_transport()
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaLLMClient(
            base_url="http://ollama.test",
            model="test-chat-model",
            temperature=0.5,
            timeout=5.0,
            max_retries=0,
            client=http_client,
        )
        response = await client.generate("System rules.", "User question?")

    assert response.answer == "Generated answer."
    assert response.model_name == "test-chat-model"
    assert response.finish_reason == "stop"
    assert response.token_usage is not None
    assert response.token_usage.prompt_tokens == 12
    assert response.token_usage.completion_tokens == 8
    assert response.token_usage.total_tokens == 20
    await client.close()


@pytest.mark.asyncio
async def test_generate_retries_on_transient_failure() -> None:
    transport = _make_transport(fail_times=1)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaLLMClient(
            base_url="http://ollama.test",
            model="test-chat-model",
            max_retries=2,
            retry_backoff=0.01,
            client=http_client,
        )
        response = await client.generate("System.", "Question?")
        assert response.answer == "Generated answer."
        await client.close()


@pytest.mark.asyncio
async def test_generate_raises_after_max_retries() -> None:
    transport = _make_transport(fail_times=5)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaLLMClient(
            base_url="http://ollama.test",
            model="test-chat-model",
            max_retries=1,
            retry_backoff=0.01,
            client=http_client,
        )
        with pytest.raises(LLMAPIError):
            await client.generate("System.", "Question?")
        await client.close()


@pytest.mark.asyncio
async def test_invalid_model_returns_model_error_without_retry() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model 'bad-model' not found")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaLLMClient(
            base_url="http://ollama.test",
            model="bad-model",
            max_retries=3,
            client=http_client,
        )
        with pytest.raises(LLMModelError, match="not found"):
            await client.generate("System.", "Question?")
        await client.close()


@pytest.mark.asyncio
async def test_timeout_maps_to_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaLLMClient(
            base_url="http://ollama.test",
            model="test-chat-model",
            max_retries=0,
            client=http_client,
        )
        with pytest.raises(LLMTimeoutError):
            await client.generate("System.", "Question?")
        await client.close()


@pytest.mark.asyncio
async def test_rejects_empty_user_prompt() -> None:
    client = OllamaLLMClient(base_url="http://ollama.test", model="test-chat-model")
    with pytest.raises(LLMClientError, match="user_prompt"):
        await client.generate("System.", "   ")


@pytest.mark.asyncio
async def test_rejects_empty_system_prompt() -> None:
    client = OllamaLLMClient(base_url="http://ollama.test", model="test-chat-model")
    with pytest.raises(LLMClientError, match="system_prompt"):
        await client.generate("   ", "Question?")


@pytest.mark.asyncio
async def test_cpu_fallback_sends_num_gpu_zero() -> None:
    capture: dict = {}
    transport = _make_transport(capture=capture)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaLLMClient(
            base_url="http://ollama.test",
            model="test-chat-model",
            use_gpu=False,
            num_thread=8,
            max_retries=0,
            client=http_client,
        )
        await client.generate("System.", "Question?")
        await client.close()

    options = capture["payload"]["options"]
    assert options["num_gpu"] == 0
    assert options["num_thread"] == 8


@pytest.mark.asyncio
async def test_gpu_enabled_can_limit_num_gpu() -> None:
    capture: dict = {}
    transport = _make_transport(capture=capture)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaLLMClient(
            base_url="http://ollama.test",
            model="test-chat-model",
            use_gpu=True,
            num_gpu=12,
            max_retries=0,
            client=http_client,
        )
        await client.generate("System.", "Question?")
        await client.close()

    assert capture["payload"]["options"]["num_gpu"] == 12


@pytest.mark.asyncio
async def test_oom_maps_to_unavailable_without_retry() -> None:
    capture: dict = {"attempts": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        capture["attempts"] = capture.get("attempts", 0) + 1
        return httpx.Response(
            500,
            text=(
                "llama-server reported out-of-memory during startup: "
                "failed to allocate CUDA_Host buffer"
            ),
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaLLMClient(
            base_url="http://ollama.test",
            model="test-chat-model",
            max_retries=3,
            retry_backoff=0.01,
            client=http_client,
        )
        with pytest.raises(LLMUnavailableError) as exc_info:
            await client.generate("System.", "Question?")
        await client.close()

    assert capture["attempts"] == 1
    body = exc_info.value.to_response_body()
    assert body["error"] == "LLM unavailable"
    assert body["reason"] == "Ollama failed to load the configured model"
    assert "out-of-memory" in body["details"].lower() or "oom" in body["details"].lower()


@pytest.mark.asyncio
async def test_supports_vision_returns_true_for_vision_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/show"
        return httpx.Response(200, json={
            "details": {
                "families": ["llama", "clip"]
            }
        })

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaLLMClient(
            base_url="http://ollama.test",
            model="vision-model",
            client=http_client,
        )
        assert await client.supports_vision() is True
        await client.close()


@pytest.mark.asyncio
async def test_supports_vision_returns_false_for_non_vision_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/show"
        return httpx.Response(200, json={
            "details": {
                "families": ["llama"]
            }
        })

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaLLMClient(
            base_url="http://ollama.test",
            model="text-model",
            client=http_client,
        )
        assert await client.supports_vision() is False
        await client.close()


@pytest.mark.asyncio
async def test_generate_with_images_encodes_base64() -> None:
    capture: dict = {}
    
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        capture["payload"] = payload
        return httpx.Response(200, json=_chat_body(model="test-chat-model"))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaLLMClient(
            base_url="http://ollama.test",
            model="test-chat-model",
            client=http_client,
        )
        # 1x1 transparent GIF
        gif_bytes = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
        await client.generate("System.", "Question?", images=[gif_bytes])
        await client.close()

    payload = capture["payload"]
    user_msg = payload["messages"][1]
    assert user_msg["role"] == "user"
    assert "images" in user_msg
    assert isinstance(user_msg["images"], list)
    assert len(user_msg["images"]) == 1
    # Check that it's base64 encoded string
    assert isinstance(user_msg["images"][0], str)
    assert len(user_msg["images"][0]) > 10
