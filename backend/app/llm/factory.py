"""LLM Provider Factory for instantiating LLM clients (Ollama, OpenRouter, NVIDIA, OmniRoute)."""
from __future__ import annotations

import logging

from app.core.config import get_settings
from app.llm.client import LLMClient, LLMClientError
from app.llm.ollama_client import OllamaLLMClient, get_global_ollama_client
from app.llm.openai_client import NvidiaLLMClient, OmniRouteLLMClient, OpenRouterLLMClient

logger = logging.getLogger(__name__)


def get_llm_client(
    provider: str | None = None,
    model: str | None = None,
    request_id: str | None = None,
    route: str | None = None,
) -> LLMClient:
    """Return an LLMClient instance based on provider and model parameters.

    If `provider` is None, defaults to `get_settings().LLM_PROVIDER` (default "ollama").
    Supported providers:
      - "ollama": Local Ollama instance (baseline Qwen3 8B)
      - "openrouter": OpenRouter API
      - "nvidia": NVIDIA API / NVIDIA Build
      - "omniroute": OmniRoute Local AI Gateway
    """
    settings = get_settings()

    logger.info(
        "stage=runtime_llm_config request_id=%s LLM_PROVIDER=%s OMNIROUTE_MODEL=%s OPENROUTER_MODEL=%s NVIDIA_MODEL=%s OLLAMA_MODEL=%s OLLAMA_USE_GPU=%s",
        request_id or "untracked",
        getattr(settings, "LLM_PROVIDER", "None"),
        getattr(settings, "OMNIROUTE_MODEL", "None"),
        getattr(settings, "OPENROUTER_MODEL", "None"),
        getattr(settings, "NVIDIA_MODEL", "None"),
        getattr(settings, "OLLAMA_MODEL", "None"),
        getattr(settings, "OLLAMA_USE_GPU", "None"),
    )

    effective_provider = (provider or "").strip().lower()
    if not effective_provider and model:
        m_lower = model.strip().lower()
        if (
            m_lower.startswith("omniroute/")
            or m_lower.startswith("omni/")
            or m_lower.startswith("auto/")
            or m_lower == "auto/fast"
            or m_lower.startswith("combo/")
            or m_lower.startswith("local-rag")
            or m_lower == (getattr(settings, "OMNIROUTE_MODEL", "") or "").strip().lower()
            or m_lower == (getattr(settings, "OMNIROUTE_VISION_MODEL", "") or "").strip().lower()
        ):
            effective_provider = "omniroute"
        elif m_lower.startswith("nvidia/") or "nemotron" in m_lower or "meta/llama-3.2-" in m_lower:
            effective_provider = "nvidia"
        elif "openrouter" in m_lower or "google/" in m_lower or "meta-llama/" in m_lower or ("/" in m_lower and ":" in m_lower):
            effective_provider = "openrouter"

    if not effective_provider:
        effective_provider = (settings.LLM_PROVIDER or "ollama").strip().lower()

    client_instance: LLMClient
    if effective_provider == "ollama":
        if model:
            client_instance = OllamaLLMClient(model=model)
        else:
            client_instance = get_global_ollama_client()

    elif effective_provider == "openrouter":
        if not settings.OPENROUTER_API_KEY or not settings.OPENROUTER_API_KEY.strip():
            raise LLMClientError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")
        target_model = model or settings.OPENROUTER_MODEL
        client_instance = OpenRouterLLMClient(model=target_model)

    elif effective_provider in ("nvidia", "nemotron"):
        if not settings.NVIDIA_API_KEY or not settings.NVIDIA_API_KEY.strip():
            raise LLMClientError("NVIDIA_API_KEY is required when LLM_PROVIDER=nvidia")
        target_model = model or settings.NVIDIA_MODEL
        client_instance = NvidiaLLMClient(model=target_model)

    elif effective_provider == "omniroute":
        target_model = model or settings.OMNIROUTE_MODEL
        client_instance = OmniRouteLLMClient(model=target_model)

    else:
        raise LLMClientError(
            f"Unsupported LLM provider '{effective_provider}'. Supported providers: 'ollama', 'openrouter', 'nvidia', 'omniroute'."
        )

    logger.info(
        "stage=provider_selected request_id=%s route=%s provider=%s model=%s base_url=%s",
        request_id or "untracked",
        route or "UNSPECIFIED",
        effective_provider,
        getattr(client_instance, "model", model or "default"),
        getattr(client_instance, "base_url", "local"),
    )
    return client_instance
