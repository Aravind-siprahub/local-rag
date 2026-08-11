"""Ollama chat API client."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.config import get_settings
from app.llm.client import (
    LLMAPIError,
    LLMClientError,
    LLMModelError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.response import LLMResponse, TokenUsage
from app.llm.sanitize import ThinkingStreamFilter, supports_think_parameter, sanitize_response

logger = logging.getLogger(__name__)

_OOM_MARKERS = (
    "out of memory",
    "out-of-memory",
    "cuda_host",
    "cuda host",
    "unable to allocate",
    "failed to allocate",
    "ggml_gallocr_alloc_graph",
    "insufficient memory",
    "oom",
)


_OLLAMA_CONCURRENCY_SEMAPHORE: asyncio.Semaphore | None = None


def _get_concurrency_semaphore() -> asyncio.Semaphore:
    global _OLLAMA_CONCURRENCY_SEMAPHORE
    if _OLLAMA_CONCURRENCY_SEMAPHORE is None:
        settings = get_settings()
        limit = max(1, getattr(settings, "OLLAMA_MAX_CONCURRENCY", 4))
        _OLLAMA_CONCURRENCY_SEMAPHORE = asyncio.Semaphore(limit)
    return _OLLAMA_CONCURRENCY_SEMAPHORE


class OllamaLLMClient:
    """Async Ollama `/api/chat` client with retry, timeout, bounded concurrency, and keep-alive support."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        retry_backoff: float = 0.5,
        use_gpu: bool | None = None,
        num_gpu: int | None = None,
        num_thread: int | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_host).rstrip("/")
        self.model = model or settings.ollama_chat_model
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.timeout = timeout if timeout is not None else settings.LLM_TIMEOUT_SECONDS
        self.max_retries = max_retries if max_retries is not None else settings.LLM_MAX_RETRIES
        self.retry_backoff = retry_backoff
        self.use_gpu = settings.OLLAMA_USE_GPU if use_gpu is None else use_gpu
        self.num_gpu = num_gpu if num_gpu is not None else settings.OLLAMA_NUM_GPU
        self.num_thread = num_thread if num_thread is not None else settings.OLLAMA_NUM_THREAD
        self.num_ctx = settings.OLLAMA_NUM_CTX
        self.num_predict = settings.OLLAMA_NUM_PREDICT
        self.keep_alive = getattr(settings, "OLLAMA_KEEP_ALIVE", "30m")
        self._client = client
        self._owns_client = client is None

    def _build_options(self, *, num_predict: int | None = None) -> dict[str, Any]:
        settings = get_settings()
        predict = (
            num_predict
            if num_predict is not None
            else getattr(settings, "OLLAMA_NUM_PREDICT", 512)
        )
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": predict,
        }
        if not self.use_gpu:
            options["num_gpu"] = 0
        elif self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu
        if self.num_thread is not None:
            options["num_thread"] = self.num_thread
        return options

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        num_predict: int | None = None,
    ) -> LLMResponse:
        """Generate a completion from system and user prompts."""
        if not user_prompt or not user_prompt.strip():
            raise LLMClientError("user_prompt must not be empty.")
        if not system_prompt or not system_prompt.strip():
            raise LLMClientError("system_prompt must not be empty.")

        options = self._build_options(num_predict=num_predict)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            "stream": stream,
            "keep_alive": ka,
            "options": options,
        }
        # qwen3 / thinking models often put the entire answer in `message.thinking`
        # and leave `content` empty on long RAG prompts. Force non-thinking output.
        from app.llm.sanitize import supports_think_parameter

        if supports_think_parameter(self.model):
            payload["think"] = False


        logger.info(
            "Ollama LLM request starting: model=%s execution=%s num_gpu=%s "
            "num_thread=%s num_predict=%s timeout_seconds=%.1f max_retries=%d stream=%s keep_alive=%s think=%s",
            self.model,
            "GPU enabled" if self.use_gpu else "CPU fallback",
            options.get("num_gpu", "default"),
            options.get("num_thread", "default"),
            options.get("num_predict"),
            self.timeout,
            self.max_retries,
            payload["stream"],
            self.keep_alive,
            payload.get("think", "omitted"),
        )
        started_at = datetime.now(timezone.utc)
        start_mono = time.monotonic()

        sem = _get_concurrency_semaphore()
        async with sem:
            response_data = await self._request_with_retry("/api/chat", payload)

        latency_ms = int((time.monotonic() - start_mono) * 1000)
        logger.info(
            "Ollama LLM request finished: model=%s latency_ms=%d started_at=%s",
            self.model,
            latency_ms,
            started_at.isoformat(),
        )
        return _parse_chat_response(response_data, fallback_model=self.model)


    async def generate_stream(
        self, system_prompt: str, user_prompt: str
    ):
        """Yield token deltas as they stream from Ollama `/api/chat`."""
        if not user_prompt or not user_prompt.strip():
            raise LLMClientError("user_prompt must not be empty.")
        if not system_prompt or not system_prompt.strip():
            raise LLMClientError("system_prompt must not be empty.")

        payload = self._build_payload(system_prompt, user_prompt, stream=True)

        url = f"{self.base_url}/api/chat"
        client = await self._get_client()
        import json

        stream_filter = ThinkingStreamFilter()

        async with client.stream("POST", url, json=payload) as response:
            if response.status_code >= 400:
                error_text = await response.aread()
                raise LLMAPIError(f"Ollama stream returned HTTP {response.status_code}: {error_text.decode('utf-8')}")

            async for line in response.aiter_lines():
                if not line or not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        continue
                    msg = data.get("message")
                    if not isinstance(msg, dict):
                        continue
                    # Never stream the separate thinking field — answer tokens only.
                    raw_token = msg.get("content")
                    token = raw_token if isinstance(raw_token, str) else ""
                    if not token:
                        continue
                    safe = stream_filter.feed(token)
                    if safe:
                        yield safe
                except (json.JSONDecodeError, TypeError, ValueError):
                    logger.debug("Skipping malformed Ollama stream line")
                    continue
                except Exception:
                    logger.warning("Unexpected error parsing Ollama stream chunk", exc_info=True)
                    continue

            tail = stream_filter.flush()
            if tail:
                yield tail

    async def warmup(self) -> bool:
        """Pre-load the model into Ollama memory to ensure sub-5s local response times."""
        try:
            client = await self._get_client()
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "keep_alive": "10m",
                "options": {"num_predict": 1},
            }
            res = await client.post(url, json=payload)
            if res.status_code == 200:
                logger.info("Ollama model %s warmed up successfully", self.model)
                return True
            logger.warning("Ollama warmup returned status %d: %s", res.status_code, res.text)
            return False
        except Exception as exc:
            logger.warning("Ollama warmup request failed: %s", exc)
            return False

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _request_with_retry(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        url = f"{self.base_url}{path}"

        for attempt in range(self.max_retries + 1):
            attempt_no = attempt + 1
            request_start = time.monotonic()
            logger.debug(
                "Ollama LLM HTTP POST %s attempt=%d/%d model=%s timeout_seconds=%.1f",
                url,
                attempt_no,
                self.max_retries + 1,
                payload.get("model"),
                self.timeout,
            )
            try:
                client = await self._get_client()
                response = await client.post(url, json=payload)
                request_ms = int((time.monotonic() - request_start) * 1000)
                logger.info(
                    "Ollama LLM HTTP response: status=%d attempt=%d/%d latency_ms=%d",
                    response.status_code,
                    attempt_no,
                    self.max_retries + 1,
                    request_ms,
                )
                if response.status_code == 404:
                    raise LLMModelError(
                        f"Ollama model {payload.get('model')!r} not found: {response.text}"
                    )
                if response.status_code >= 400:
                    error_text = response.text
                    unavailable = _unavailable_error_from_response(
                        status_code=response.status_code,
                        error_text=error_text,
                        model=str(payload.get("model")),
                    )
                    if unavailable is not None:
                        raise unavailable
                    if _looks_like_model_error(response.status_code, error_text):
                        raise LLMModelError(
                            f"Ollama model error for {payload.get('model')!r}: {error_text}"
                        )
                    raise LLMAPIError(f"Ollama returned HTTP {response.status_code}: {error_text}")
                return response.json()
            except httpx.TimeoutException:
                request_ms = int((time.monotonic() - request_start) * 1000)
                last_error = LLMTimeoutError(f"Ollama request timed out after {self.timeout}s.")
                logger.warning(
                    "Ollama LLM timeout during generation/read (attempt %d/%d, elapsed_ms=%d, timeout_seconds=%.1f)",
                    attempt_no,
                    self.max_retries + 1,
                    request_ms,
                    self.timeout,
                )
            except httpx.HTTPError as exc:
                last_error = LLMAPIError(f"Ollama HTTP error: {exc}")
                logger.warning(
                    "Ollama LLM HTTP error (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
            except (LLMModelError, LLMUnavailableError):
                # Model missing / OOM will not recover on retry.
                raise
            except LLMAPIError as exc:
                last_error = exc
                logger.warning(
                    "Ollama LLM API error (attempt %d/%d): %s",
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
            # Single timeout applies to connect + read while Ollama generates.
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        return self._client


def _parse_chat_response(data: dict[str, Any], fallback_model: str) -> LLMResponse:
    if not isinstance(data, dict):
        raise LLMAPIError("Ollama response is not a JSON object.")

    message = data.get("message")
    if not isinstance(message, dict):
        raise LLMAPIError("Ollama response missing 'message' object.")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        # Some thinking models still return the answer under alternate keys.
        for alt_key in ("thinking", "reasoning"):
            alt = message.get(alt_key)
            if isinstance(alt, str) and alt.strip():
                content = alt
                break
    if not isinstance(content, str) or not content.strip():
        raise LLMAPIError("Ollama response missing assistant 'content'.")

    model_name = data.get("model") if isinstance(data.get("model"), str) else fallback_model
    finish_reason = data.get("done_reason") if isinstance(data.get("done_reason"), str) else None

    prompt_tokens = _optional_int(data.get("prompt_eval_count"))
    completion_tokens = _optional_int(data.get("eval_count"))
    total_tokens = None
    if prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    token_usage = None
    if prompt_tokens is not None or completion_tokens is not None:
        token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    return LLMResponse(
        answer=content.strip(),
        model_name=model_name,
        token_usage=token_usage,
        finish_reason=finish_reason,
    )


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _looks_like_model_error(status_code: int, error_text: str) -> bool:
    lowered = error_text.lower()
    if status_code == 400 and "model" in lowered:
        return True
    return any(marker in lowered for marker in ("not found", "does not exist", "unknown model"))


def _looks_like_oom(error_text: str) -> bool:
    lowered = error_text.lower()
    return any(marker in lowered for marker in _OOM_MARKERS)


def _unavailable_error_from_response(
    *,
    status_code: int,
    error_text: str,
    model: str,
) -> LLMUnavailableError | None:
    """Map Ollama load/resource failures to a structured unavailable error."""
    if status_code < 500 and not _looks_like_oom(error_text):
        return None

    if _looks_like_oom(error_text):
        return LLMUnavailableError(
            reason="Ollama failed to load the configured model",
            details=f"OOM error: {error_text.strip()}",
        )

    if status_code >= 500 and any(
        marker in error_text.lower()
        for marker in ("failed to load", "unable to load", "llama-server", "model failed")
    ):
        return LLMUnavailableError(
            reason="Ollama failed to load the configured model",
            details=error_text.strip()[:500] or f"HTTP {status_code} for model {model}",
        )

    return None
