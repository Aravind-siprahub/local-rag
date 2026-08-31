"""Configurable Model Router for agentic task assignment and model selection."""
from __future__ import annotations

import logging
from enum import Enum
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class TaskRole(str, Enum):
    CLASSIFY = "CLASSIFY"
    QUERY_REWRITE = "QUERY_REWRITE"
    RAG_REASONING = "RAG_REASONING"
    COMPLEX_REASONING = "COMPLEX_REASONING"
    VISION = "VISION"
    FINAL_ANSWER = "FINAL_ANSWER"
    EMBEDDING = "EMBEDDING"


class ModelRouter:
    """Configurable model selection engine based on environment settings."""

    @staticmethod
    def get_model(role: TaskRole) -> str:
        """Return the configured model name for a specific task role."""
        settings = get_settings()

        if role == TaskRole.CLASSIFY:
            model = getattr(settings, "MODEL_ROUTER_CLASSIFY", None) or getattr(settings, "OLLAMA_MODEL", None) or getattr(settings, "CHAT_MODEL", "qwen3:8b")
        elif role == TaskRole.QUERY_REWRITE:
            model = getattr(settings, "MODEL_QUERY_REWRITE", None) or getattr(settings, "OLLAMA_MODEL", None) or getattr(settings, "CHAT_MODEL", "qwen3:8b")
        elif role == TaskRole.RAG_REASONING:
            model = getattr(settings, "MODEL_RAG_REASONING", None) or getattr(settings, "OLLAMA_MODEL", None) or getattr(settings, "CHAT_MODEL", "qwen3:8b")
        elif role == TaskRole.COMPLEX_REASONING:
            model = getattr(settings, "MODEL_COMPLEX_REASONING", None) or getattr(settings, "OLLAMA_MODEL", None) or getattr(settings, "CHAT_MODEL", "qwen3:8b")
        elif role == TaskRole.VISION:
            model = getattr(settings, "OLLAMA_VISION_MODEL", "qwen3-vl:4b")
        elif role == TaskRole.FINAL_ANSWER:
            model = getattr(settings, "MODEL_FINAL_ANSWER", None) or getattr(settings, "OLLAMA_MODEL", None) or getattr(settings, "CHAT_MODEL", "qwen3:8b")
        elif role == TaskRole.EMBEDDING:
            model = getattr(settings, "EMBEDDING_MODEL", "nomic-embed-text")
        else:
            model = getattr(settings, "CHAT_MODEL", "qwen3:8b")

        logger.debug("[MODEL ROUTER] role=%s selected_model=%s", role.value, model)
        return model
