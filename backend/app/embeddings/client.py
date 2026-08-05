"""Embedding provider interface and Ollama HTTP client."""
from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingClientError(Exception):
    """Base class for embedding client failures."""


class EmbeddingTimeoutError(EmbeddingClientError):
    """The embedding request timed out."""


class EmbeddingAPIError(EmbeddingClientError):
    """The embedding provider returned an error response."""


@runtime_checkable
class EmbeddingClient(Protocol):
    """Minimal contract for an async embedding provider."""

    async def embed(self, text: str) -> list[float]: ...

    async def close(self) -> None: ...


class OllamaEmbeddingClient:
    """Async Ollama `/api/embeddings` client with retry and timeout support."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        retry_backoff: float = 0.5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_host).rstrip("/")
        self.model = model or settings.EMBEDDING_MODEL
        self.dimensions = dimensions if dimensions is not None else settings.EMBEDDING_DIMENSIONS
        self.timeout = timeout if timeout is not None else settings.EMBEDDING_TIMEOUT_SECONDS
        self.max_retries = max_retries if max_retries is not None else settings.EMBEDDING_MAX_RETRIES
        self.retry_backoff = retry_backoff
        self._client = client
        self._owns_client = client is None

    async def embed(self, text: str) -> list[float]:
        """Generate one embedding vector for a single text input."""
        if not text or not text.strip():
            raise EmbeddingClientError("Cannot embed empty text.")

        payload = {"model": self.model, "prompt": text, "keep_alive": 0}
        response_data = await self._request_with_retry("/api/embeddings", payload)
        embedding = response_data.get("embedding")
        if not isinstance(embedding, list):
            raise EmbeddingAPIError("Ollama response missing 'embedding' list.")

        if len(embedding) != self.dimensions:
            raise EmbeddingAPIError(
                f"Ollama returned {len(embedding)} dimensions, expected {self.dimensions}."
            )

        logger.info(
            "Embedding OK: vector_length=%d configured_dimensions=%d model=%s",
            len(embedding),
            self.dimensions,
            self.model,
        )

        return [float(value) for value in embedding]

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request_with_retry(self, path: str, payload: dict) -> dict:
        last_error: Exception | None = None
        url = f"{self.base_url}{path}"

        for attempt in range(self.max_retries + 1):
            try:
                client = await self._get_client()
                response = await client.post(url, json=payload)
                if response.status_code >= 400:
                    raise EmbeddingAPIError(
                        f"Ollama returned HTTP {response.status_code}: {response.text}"
                    )
                return response.json()
            except httpx.TimeoutException as exc:
                last_error = EmbeddingTimeoutError(f"Ollama request timed out after {self.timeout}s.")
                logger.warning("Ollama embedding timeout (attempt %d/%d)", attempt + 1, self.max_retries + 1)
            except httpx.HTTPError as exc:
                last_error = EmbeddingAPIError(f"Ollama HTTP error: {exc}")
                logger.warning(
                    "Ollama embedding HTTP error (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
            except EmbeddingAPIError as exc:
                last_error = exc
                logger.warning(
                    "Ollama embedding API error (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )

            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_backoff * (2 ** attempt))

        assert last_error is not None
        raise last_error

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
