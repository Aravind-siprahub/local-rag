"""LLM provider interface and shared errors."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.llm.response import LLMResponse


class LLMClientError(Exception):
    """Base class for LLM client failures."""


class LLMTimeoutError(LLMClientError):
    """The LLM request timed out."""


class LLMAPIError(LLMClientError):
    """The LLM provider returned an error response."""


class LLMModelError(LLMAPIError):
    """The requested model is missing or invalid."""


class LLMUnavailableError(LLMClientError):
    """The LLM provider cannot load or run the configured model (e.g. OOM)."""

    def __init__(
        self,
        *,
        reason: str = "Ollama failed to load the configured model",
        details: str,
        error: str = "LLM unavailable",
    ) -> None:
        self.error = error
        self.reason = reason
        self.details = details
        super().__init__(f"{error}: {reason} ({details})")

    def to_response_body(self) -> dict[str, str]:
        return {
            "error": self.error,
            "reason": self.reason,
            "details": self.details,
        }


@runtime_checkable
class LLMClient(Protocol):
    """Minimal contract for an async text-generation provider."""

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
    ) -> LLMResponse: ...

    async def close(self) -> None: ...
