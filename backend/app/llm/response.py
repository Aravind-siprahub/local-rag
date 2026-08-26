"""LLM generation response types."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenUsage:
    """Token counts when the provider reports them."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    """Normalized result from an LLM generation call."""

    answer: str
    model_name: str
    token_usage: TokenUsage | None = None
    finish_reason: str | None = None
    ttft_ms: float | None = None
    generation_time_ms: float | None = None
    tokens_per_second: float | None = None
    cost_usd: float | None = None
    provider: str | None = None
