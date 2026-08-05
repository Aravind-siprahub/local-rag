"""Unit tests for `app.embeddings.client`."""
import json

import httpx
import pytest

from app.embeddings.client import (
    EmbeddingAPIError,
    EmbeddingClientError,
    EmbeddingTimeoutError,
    OllamaEmbeddingClient,
)

_VECTOR = [0.1] * 768


def _make_transport(
    *,
    status_code: int = 200,
    body: dict | None = None,
    fail_times: int = 0,
) -> httpx.MockTransport:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if fail_times > attempts:
            attempts += 1
            return httpx.Response(503, text="temporary failure")
        payload = json.loads(request.content)
        assert payload["model"] == "test-model"
        assert payload["prompt"] == "hello"
        return httpx.Response(status_code, json=body or {"embedding": _VECTOR})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_embed_returns_vector() -> None:
    transport = _make_transport()
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaEmbeddingClient(
            base_url="http://ollama.test",
            model="test-model",
            dimensions=768,
            timeout=5.0,
            max_retries=0,
            client=http_client,
        )
        vector = await client.embed("hello")
        assert len(vector) == 768
        await client.close()


@pytest.mark.asyncio
async def test_embed_retries_on_failure() -> None:
    transport = _make_transport(fail_times=1)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaEmbeddingClient(
            base_url="http://ollama.test",
            model="test-model",
            dimensions=768,
            max_retries=2,
            retry_backoff=0.01,
            client=http_client,
        )
        vector = await client.embed("hello")
        assert len(vector) == 768
        await client.close()


@pytest.mark.asyncio
async def test_embed_raises_after_max_retries() -> None:
    transport = _make_transport(fail_times=5)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaEmbeddingClient(
            base_url="http://ollama.test",
            model="test-model",
            dimensions=768,
            max_retries=1,
            retry_backoff=0.01,
            client=http_client,
        )
        with pytest.raises(EmbeddingAPIError):
            await client.embed("hello")
        await client.close()


@pytest.mark.asyncio
async def test_embed_rejects_wrong_dimensions() -> None:
    transport = _make_transport(body={"embedding": [0.1, 0.2]})
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaEmbeddingClient(
            base_url="http://ollama.test",
            model="test-model",
            dimensions=768,
            client=http_client,
        )
        with pytest.raises(EmbeddingAPIError, match="768"):
            await client.embed("hello")
        await client.close()


@pytest.mark.asyncio
async def test_embed_rejects_empty_text() -> None:
    client = OllamaEmbeddingClient(base_url="http://ollama.test", model="test-model", dimensions=768)
    with pytest.raises(EmbeddingClientError, match="empty"):
        await client.embed("   ")


@pytest.mark.asyncio
async def test_timeout_maps_to_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OllamaEmbeddingClient(
            base_url="http://ollama.test",
            model="test-model",
            dimensions=768,
            max_retries=0,
            client=http_client,
        )
        with pytest.raises(EmbeddingTimeoutError):
            await client.embed("hello")
        await client.close()
