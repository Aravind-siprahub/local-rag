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


_GLOBAL_OLLAMA_CLIENT = None

def get_global_ollama_client() -> "OllamaLLMClient":
    """Return a global singleton of OllamaLLMClient to reuse HTTP connections."""
    global _GLOBAL_OLLAMA_CLIENT
    if _GLOBAL_OLLAMA_CLIENT is None:
        _GLOBAL_OLLAMA_CLIENT = OllamaLLMClient()
    return _GLOBAL_OLLAMA_CLIENT


def _parse_user_prompt(user_prompt: str) -> tuple[list[dict[str, str]], str]:
    """Extract chat history messages from the user_prompt prefix.

    Returns ``(history_messages, remaining)`` where *remaining* is the full
    context + question block that should be sent as the user message unchanged.
    We deliberately do NOT try to re-split the context from the question here
    because the branch conditions previously used did not match our template
    format (USER_PROMPT_WITH_CONTEXT), causing document context to be silently
    dropped every time.
    """
    user_prompt = user_prompt.strip()
    history_messages: list[dict[str, str]] = []

    remaining = user_prompt

    # Extract the optional chat-history prefix added by format_user_prompt().
    if "Recent Conversation:" in remaining and "---------------------------------" in remaining:
        parts = remaining.split("Recent Conversation:\n", 1)
        if len(parts) == 2:
            after_header = parts[1]
            if "---------------------------------" in after_header:
                history_part, remaining = after_header.split("---------------------------------", 1)
                remaining = remaining.strip()

                for line in history_part.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("User:"):
                        history_messages.append({"role": "user", "content": line[5:].strip()})
                    elif line.startswith("Assistant:"):
                        history_messages.append({"role": "assistant", "content": line[10:].strip()})
                    elif line.startswith("[Prior Conversation Summary:") and line.endswith("]"):
                        history_messages.append({"role": "system", "content": line[1:-1].strip()})

    return history_messages, remaining or user_prompt


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
        default_predict = getattr(settings, "OLLAMA_NUM_PREDICT", 128)
        predict = num_predict if num_predict is not None else default_predict
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": predict,
            "top_p": getattr(settings, "OLLAMA_TOP_P", 0.9),
            "top_k": getattr(settings, "OLLAMA_TOP_K", 40),
            "think": False,
        }
        if not self.use_gpu:
            options["num_gpu"] = 0
        elif self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu
        else:
            options["num_gpu"] = getattr(settings, "OLLAMA_NUM_GPU", 99) or 99

        if self.num_thread is not None:
            options["num_thread"] = self.num_thread
        elif settings.OLLAMA_NUM_THREAD is not None:
            options["num_thread"] = settings.OLLAMA_NUM_THREAD
        else:
            import os
            cpu_cnt = os.cpu_count() or 4
            options["num_thread"] = max(1, cpu_cnt // 2 if cpu_cnt > 4 else cpu_cnt)

        return options

    async def supports_vision(self, model: str | None = None) -> bool:
        """Query the model info from Ollama to see if it supports vision/multimodal input."""
        target_model = model or self.model
        model_lower = target_model.lower()

        # Fast path: trust the model name before hitting the network.
        # qwen3-vl, qwen2-vl, llava*, bakllava, moondream, minicpm-v, mllama, etc.
        _VISION_NAME_PATTERNS = ("vision", "-vl", ":vl", "llava", "mllama", "bakllava", "minicpm", "moondream")
        if any(p in model_lower for p in _VISION_NAME_PATTERNS):
            logger.info("[VISION] supports_vision=True (name match) model=%s", target_model)
            return True

        # Slow path: ask Ollama /api/show for explicit capability metadata.
        try:
            client = await self._get_client()
            url = f"{self.base_url}/api/show"
            payload = {"name": target_model}
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                # 1. Explicit capabilities array (modern Ollama)
                capabilities = data.get("capabilities", [])
                if "vision" in capabilities:
                    logger.info("[VISION] supports_vision=True (capabilities) model=%s", target_model)
                    return True
                # 2. Model family — includes qwen alongside llava/mllama/clip
                details = data.get("details", {})
                families = details.get("families", []) or ([details.get("family")] if details.get("family") else [])
                _VISION_FAMILIES = {"mllama", "llava", "clip", "minicpm", "moondream", "qwen-vl", "qwen2-vl", "qwen3-vl"}
                if any(f in _VISION_FAMILIES for f in families):
                    logger.info("[VISION] supports_vision=True (family=%s) model=%s", families, target_model)
                    return True
                # 3. model_info projector key (older Ollama builds)
                model_info = data.get("model_info", {})
                if any("projector" in k for k in model_info.keys()):
                    logger.info("[VISION] supports_vision=True (projector key) model=%s", target_model)
                    return True
            logger.info("[VISION] supports_vision=False (Ollama show returned no vision signal) model=%s", target_model)
            return False
        except Exception as exc:
            logger.warning("[VISION] /api/show failed for model=%s: %s — defaulting to False", target_model, exc)
            return False

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
        """Generate a completion from system and user prompts."""
        if not user_prompt or not user_prompt.strip():
            raise LLMClientError("user_prompt must not be empty.")
        if not system_prompt or not system_prompt.strip():
            raise LLMClientError("system_prompt must not be empty.")

        req_id = request_id or f"req-{time.time_ns()}"
        target_model = model or self.model
        payload = self._build_payload(
            system_prompt,
            user_prompt,
            stream=False,
            num_predict=num_predict,
            response_format=response_format,
            temperature=temperature,
            images=images,
            model=model,
        )

        logger.info(
            "stage=provider_request_started request_id=%s provider=ollama model=%s endpoint=%s/api/chat stream=false execution=%s",
            req_id, target_model, self.base_url, "GPU" if self.use_gpu else "CPU"
        )
        started_at = datetime.now(timezone.utc)
        start_mono = time.monotonic()

        sem = _get_concurrency_semaphore()
        async with sem:
            response_data = await self._request_with_retry("/api/chat", payload)

        latency_ms = int((time.monotonic() - start_mono) * 1000)
        prompt_tokens = response_data.get("prompt_eval_count") or 0
        completion_tokens = response_data.get("eval_count") or 0
        logger.info(
            "stage=provider_response_received request_id=%s provider=ollama model=%s status=200 prompt_tokens=%s completion_tokens=%s duration_ms=%d",
            req_id, target_model, prompt_tokens, completion_tokens, latency_ms
        )
        return _parse_chat_response(response_data, fallback_model=model or self.model)

    def _build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        stream: bool = False,
        num_predict: int | None = None,
        response_format: str | None = None,
        temperature: float | None = None,
        images: list[bytes] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Build a standard Ollama `/api/chat` request body."""
        import base64

        options = self._build_options(num_predict=num_predict)
        if temperature is not None:
            options["temperature"] = temperature

        history_messages, remaining = _parse_user_prompt(user_prompt)

        messages: list[dict[str, Any]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})

        messages.extend(history_messages)
        
        last_message: dict[str, Any] = {"role": "user", "content": remaining or user_prompt.strip()}
        if images:
            b64_images: list[str] = []
            for img in images:
                if isinstance(img, bytes):
                    b64_images.append(base64.b64encode(img).decode("utf-8"))
                elif isinstance(img, str):
                    clean_str = img.split(",", 1)[-1] if "," in img else img
                    b64_images.append(clean_str.strip())
            if b64_images:
                last_message["images"] = b64_images
                logger.info(
                    "[OLLAMA VISION PAYLOAD] attached %d image(s) to user message (b64 len sample=%d)",
                    len(b64_images),
                    len(b64_images[0]),
                )
        messages.append(last_message)

        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": options,
        }
        if response_format is not None:
            payload["format"] = response_format
        target_model = model or self.model
        if supports_think_parameter(target_model):
            payload["think"] = False
        return payload

    async def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        num_predict: int | None = None,
        images: list[bytes] | None = None,
        model: str | None = None,
        request_id: str | None = None,
    ):
        """Yield token deltas as they stream from Ollama `/api/chat`."""
        if not user_prompt or not user_prompt.strip():
            raise LLMClientError("user_prompt must not be empty.")
        if not system_prompt or not system_prompt.strip():
            raise LLMClientError("system_prompt must not be empty.")

        req_id = request_id or f"req-{time.time_ns()}"
        target_model = model or self.model
        payload = self._build_payload(system_prompt, user_prompt, stream=True, num_predict=num_predict, images=images, model=model)

        url = f"{self.base_url}/api/chat"
        client = await self._get_client()
        import json

        stream_filter = ThinkingStreamFilter()

        logger.info(
            "stage=provider_request_started request_id=%s provider=ollama model=%s endpoint=%s stream=true",
            req_id, target_model, url
        )

        async with client.stream("POST", url, json=payload) as response:
            if response.status_code >= 400:
                error_text = await response.aread()
                logger.error(
                    "stage=provider_response_failed request_id=%s provider=ollama model=%s status=%d body=%s",
                    req_id, target_model, response.status_code, error_text.decode("utf-8", errors="ignore")[:200]
                )
                raise LLMAPIError(f"Ollama stream returned HTTP {response.status_code}: {error_text.decode('utf-8')}")

            logger.info(
                "stage=provider_response_received request_id=%s provider=ollama model=%s status=200 stream=true",
                req_id, target_model
            )

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
            limits = httpx.Limits(max_keepalive_connections=20, max_connections=100, keepalive_expiry=600.0)
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout), limits=limits)
        return self._client


import re

def _normalize_final_answer(answer: str) -> str:
    """Safely remove reasoning wrappers and 'Final answer:' prefixes without destroying facts."""
    if not answer or not isinstance(answer, str):
        return ""
    
    cleaned = answer.strip()
    
    # 1. Strip structural thinking blocks via existing sanitizer
    from app.llm.sanitize import sanitize_response
    cleaned = sanitize_response(cleaned)
    
    # 2. Extract everything after literal "Final answer:" or "Therefore, the answer is:"
    # This handles Qwen3:4b models that leak meta-commentary into the content chunk.
    final_answer_match = re.search(r"(?i)(?:final\s*answer|therefore,? the answer is)[:\s]*\n*(.*)", cleaned, flags=re.DOTALL)
    if final_answer_match:
        extracted = final_answer_match.group(1).strip()
        if extracted:
            cleaned = extracted
            
    return cleaned.strip()

def _parse_chat_response(data: dict[str, Any], fallback_model: str) -> LLMResponse:
    if not isinstance(data, dict):
        raise LLMAPIError("Ollama response is not a JSON object.")

    message = data.get("message")
    if not isinstance(message, dict):
        raise LLMAPIError("Ollama response missing 'message' object.")

    content = message.get("content")
    thinking = message.get("thinking")

    from app.llm.sanitize import is_reasoning_model

    if isinstance(content, str):
        content = _normalize_final_answer(content)

    model_name = data.get("model") if isinstance(data.get("model"), str) else fallback_model

    if not isinstance(content, str) or not content.strip():
        if is_reasoning_model(model_name) and (not isinstance(thinking, str) or not thinking.strip()):
            content = ""
        else:
            content = "Information not found in document excerpts."

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
