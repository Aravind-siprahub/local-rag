"""OpenAI-compatible LLM provider client (OpenRouter, NVIDIA API, etc.)."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any, AsyncGenerator

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

logger = logging.getLogger(__name__)


class OpenAICompatibleLLMClient:
    """Reusable async client for OpenAI-compatible `/chat/completions` REST endpoints."""

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str | None = None,
        model: str,
        temperature: float | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        prompt_price_per_1m: float | None = None,
        completion_price_per_1m: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = get_settings()
        self.provider = provider.lower()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.timeout = timeout if timeout is not None else settings.LLM_TIMEOUT_SECONDS
        self.max_retries = max_retries if max_retries is not None else settings.LLM_MAX_RETRIES
        self.prompt_price_per_1m = prompt_price_per_1m
        self.completion_price_per_1m = completion_price_per_1m
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._owns_client = True
        elif getattr(self._client, "is_closed", False) is True:
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._owns_client = True
        return self._client


    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://sipra.io"
            headers["X-Title"] = "Sipra Local RAG"

        return headers

    def _validate_credentials(self) -> None:
        if self.provider == "omniroute":
            # OmniRoute runs as a local gateway — API key is optional.
            return
        if not self.api_key or not self.api_key.strip():
            if self.provider == "openrouter":
                raise LLMClientError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")
            elif self.provider in ("nvidia", "nemotron"):
                raise LLMClientError("NVIDIA_API_KEY is required when LLM_PROVIDER=nvidia")
            else:
                raise LLMClientError(f"{self.provider.upper()}_API_KEY is required when LLM_PROVIDER={self.provider}")


    @staticmethod
    def _format_messages(
        system_prompt: str,
        user_prompt: str,
        images: list[bytes] | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        if images:
            content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt.strip()}]
            for img in images:
                mime = "image/jpeg"
                if img.startswith(b"\x89PNG"):
                    mime = "image/png"
                elif img.startswith(b"GIF8"):
                    mime = "image/gif"
                elif img.startswith(b"RIFF") and b"WEBP" in img[:12]:
                    mime = "image/webp"
                b64 = base64.b64encode(img).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_prompt.strip()})
        return messages

    async def supports_vision(self, model: str | None = None) -> bool:
        """Check whether the target model or provider supports multimodal/vision input."""
        target = (model or self.model or "").lower()
        if target.startswith("omniroute/"):
            target = target[len("omniroute/") :]
        _VISION_PATTERNS = (
            "vision", "-vl", ":vl", "4o", "flash", "omni", "llava", "multimodal",
            "local-rag-vision", "nemotron-3-nano-omni", "qwen-vl", "qwen2-vl", "qwen2.5-vl", "qwen3-vl"
        )
        return any(p in target for p in _VISION_PATTERNS)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        num_predict: int | None = None,
        response_format: str | None = None,
        temperature: float | None = None,
        images: list[bytes] | None = None,
        model: str | None = None,
        request_id: str | None = None,
    ) -> LLMResponse:
        """Execute a non-streaming completion request against OpenAI-compatible API."""
        self._validate_credentials()
        target_model = model or self.model
        if self.provider == "nvidia" and target_model in ("nvidia/nemotron-4-340b-instruct", "nemotron-4-340b-instruct"):
            target_model = "nvidia/nemotron-3.5-lightning-30b-a3b"
        elif self.provider == "omniroute" and target_model and target_model.startswith("omniroute/"):
            target_model = target_model[len("omniroute/") :]
        elif self.provider == "nvidia" and target_model and (target_model.startswith("nvidia/meta/") or target_model.startswith("nvidia/microsoft/")):
            target_model = target_model[len("nvidia/") :]
        temp = self.temperature if temperature is None else temperature
        headers = self._build_headers()
        req_id = request_id or f"req-{time.time_ns()}"
        headers["X-Request-ID"] = req_id

        messages = self._format_messages(system_prompt, user_prompt, images)

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temp,
            "stream": False,
        }
        if num_predict is not None:
            payload["max_tokens"] = num_predict

        client = await self._get_client()
        endpoint_url = f"{self.base_url}/chat/completions"
        start_mono = time.monotonic()

        logger.info(
            "stage=provider_request_started request_id=%s provider=%s model=%s base_url=%s endpoint=%s stream=false",
            req_id, self.provider, target_model, self.base_url, endpoint_url
        )

        last_error: Exception | None = None
        for attempt in range(1 + self.max_retries):
            try:
                response = await client.post(endpoint_url, json=payload, headers=headers)
                generation_time_ms = max((time.monotonic() - start_mono) * 1000.0, 0.1)

                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise LLMAPIError(f"{self.provider} returned an empty choices list.")

                    first_choice = choices[0]
                    message = first_choice.get("message", {})
                    answer = message.get("content", "") or ""
                    finish_reason = first_choice.get("finish_reason")

                    raw_usage = data.get("usage", {})
                    token_usage = None
                    tokens_per_sec = None
                    cost_usd = None
                    prompt_tokens = 0
                    completion_tokens = 0

                    if raw_usage:
                        prompt_tokens = raw_usage.get("prompt_tokens") or 0
                        completion_tokens = raw_usage.get("completion_tokens") or 0
                        total_tokens = raw_usage.get("total_tokens") or (prompt_tokens + completion_tokens)
                        token_usage = TokenUsage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                        )

                        if completion_tokens and generation_time_ms > 0:
                            tokens_per_sec = round(completion_tokens / (generation_time_ms / 1000.0), 2)

                        if prompt_tokens is not None and completion_tokens is not None:
                            if self.prompt_price_per_1m is not None and self.completion_price_per_1m is not None:
                                cost_usd = round(
                                    (prompt_tokens * self.prompt_price_per_1m / 1e6)
                                    + (completion_tokens * self.completion_price_per_1m / 1e6),
                                    6,
                                )

                    logger.info(
                        "stage=provider_response_received request_id=%s provider=%s model=%s status=200 prompt_tokens=%s completion_tokens=%s duration_ms=%.2f",
                        req_id, self.provider, target_model, prompt_tokens, completion_tokens, generation_time_ms
                    )

                    return LLMResponse(
                        answer=answer,
                        model_name=target_model,
                        token_usage=token_usage,
                        finish_reason=finish_reason,
                        ttft_ms=None,
                        generation_time_ms=round(generation_time_ms, 2),
                        tokens_per_second=tokens_per_sec,
                        cost_usd=cost_usd,
                        provider=self.provider,
                    )

                elif response.status_code in (401, 403):
                    raise LLMAPIError(
                        f"{self.provider.upper()} Authentication error ({response.status_code}): {response.text}"
                    )
                elif response.status_code == 404:
                    raise LLMModelError(
                        f"{self.provider.upper()} Model not found ({target_model}): {response.text}"
                    )
                elif response.status_code == 429:
                    last_error = LLMAPIError(f"{self.provider.upper()} Rate limit exceeded: {response.text}")
                else:
                    last_error = LLMAPIError(
                        f"{self.provider.upper()} HTTP {response.status_code}: {response.text}"
                    )

            except httpx.TimeoutException as exc:
                last_error = LLMTimeoutError(
                    f"{self.provider.upper()} request timed out after {self.timeout}s: {exc}"
                )
            except httpx.RequestError as exc:
                last_error = LLMClientError(f"{self.provider.upper()} Network connection error: {exc}")

            if attempt < self.max_retries:
                await asyncio.sleep(0.5 * (2**attempt))

        if last_error:
            raise last_error
        raise LLMClientError(f"{self.provider.upper()} request failed after {self.max_retries} retries.")

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        num_predict: int | None = None,
        temperature: float | None = None,
        images: list[bytes] | None = None,
        model: str | None = None,
        request_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream completion tokens from OpenAI-compatible SSE endpoint."""
        self._validate_credentials()
        target_model = model or self.model
        if self.provider == "nvidia" and target_model in ("nvidia/nemotron-4-340b-instruct", "nemotron-4-340b-instruct"):
            target_model = "nvidia/nemotron-3.5-lightning-30b-a3b"
        elif self.provider == "omniroute" and target_model and target_model.startswith("omniroute/"):
            target_model = target_model[len("omniroute/") :]
        elif self.provider == "nvidia" and target_model and (target_model.startswith("nvidia/meta/") or target_model.startswith("nvidia/microsoft/")):
            target_model = target_model[len("nvidia/") :]
        temp = self.temperature if temperature is None else temperature
        headers = self._build_headers()
        req_id = request_id or f"req-{time.time_ns()}"
        headers["X-Request-ID"] = req_id

        messages = self._format_messages(system_prompt, user_prompt, images)

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temp,
            "stream": True,
        }
        if num_predict is not None:
            payload["max_tokens"] = num_predict

        client = await self._get_client()
        endpoint_url = f"{self.base_url}/chat/completions"
        start_mono = time.monotonic()

        logger.info(
            "stage=provider_request_started request_id=%s provider=%s model=%s base_url=%s endpoint=%s stream=true",
            req_id, self.provider, target_model, self.base_url, endpoint_url
        )

        async with client.stream("POST", endpoint_url, json=payload, headers=headers) as response:
            if response.status_code != 200:
                body = await response.aread()
                logger.error(
                    "stage=provider_response_failed request_id=%s provider=%s model=%s status=%d body=%s",
                    req_id, self.provider, target_model, response.status_code, body.decode("utf-8", errors="ignore")[:200]
                )
                raise LLMAPIError(f"{self.provider.upper()} streaming error {response.status_code}: {body.decode('utf-8', errors='ignore')}")

            logger.info(
                "stage=provider_response_received request_id=%s provider=%s model=%s status=200 stream=true",
                req_id, self.provider, target_model
            )
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk_json = json.loads(data_str)
                    choices = chunk_json.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                except Exception:
                    continue

    async def close(self) -> None:
        if self._owns_client and self._client and not self._client.is_closed:
            await self._client.aclose()


class OpenRouterLLMClient(OpenAICompatibleLLMClient):
    """OpenRouter API LLM Client implementation."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        prompt_price_per_1m: float | None = None,
        completion_price_per_1m: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = get_settings()
        super().__init__(
            provider="openrouter",
            base_url=base_url or settings.OPENROUTER_BASE_URL,
            api_key=api_key if api_key is not None else settings.OPENROUTER_API_KEY,
            model=model or settings.OPENROUTER_MODEL,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            prompt_price_per_1m=prompt_price_per_1m,
            completion_price_per_1m=completion_price_per_1m,
            client=client,
        )


class NvidiaLLMClient(OpenAICompatibleLLMClient):
    """NVIDIA API / NVIDIA Build LLM Client implementation."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        prompt_price_per_1m: float | None = None,
        completion_price_per_1m: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = get_settings()
        super().__init__(
            provider="nvidia",
            base_url=base_url or settings.NVIDIA_BASE_URL,
            api_key=api_key if api_key is not None else settings.NVIDIA_API_KEY,
            model=model or settings.NVIDIA_MODEL,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            prompt_price_per_1m=prompt_price_per_1m,
            completion_price_per_1m=completion_price_per_1m,
            client=client,
        )


class OmniRouteLLMClient(OpenAICompatibleLLMClient):
    """OmniRoute Local AI Gateway LLM Client implementation."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        prompt_price_per_1m: float | None = None,
        completion_price_per_1m: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = get_settings()
        target_model = model or settings.OMNIROUTE_MODEL
        if target_model and target_model.startswith("omniroute/"):
            target_model = target_model[len("omniroute/") :]
        super().__init__(
            provider="omniroute",
            base_url=base_url or settings.OMNIROUTE_BASE_URL,
            api_key=api_key if api_key is not None else settings.OMNIROUTE_API_KEY,
            model=target_model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries if max_retries is not None else 1,
            prompt_price_per_1m=prompt_price_per_1m,
            completion_price_per_1m=completion_price_per_1m,
            client=client,
        )
