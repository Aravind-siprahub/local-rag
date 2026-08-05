"""Application configuration.

Single source of truth for all runtime configuration, loaded from `.env` in
development and real environment variables in production. This is the only
module allowed to read `os.environ` / `.env` directly — everything else
imports `get_settings()`.
"""
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse, urlunparse

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PLACEHOLDER_MARKERS = ("<PASSWORD>", "xxxxxxxxx", "<PROJECT_REF>")


class Settings(BaseSettings):
    """Strongly-typed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- App metadata -----------------------------------------------------
    APP_NAME: str = "Local RAG API"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"  # development | staging | production
    DEBUG: bool = False

    # --- Database -----------------------------------------------------------
    DATABASE_URL: str

    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800
    DB_ECHO: bool = False

    # --- Local file storage (document uploads) ------------------------------
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE_MB: int = 25

    # --- Document text processing (parse / chunk) -----------------------------
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # --- Embeddings (Ollama) --------------------------------------------------
    # Prefer OLLAMA_HOST when set; OLLAMA_BASE_URL kept for backward compatibility.
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_HOST: str | None = None
    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_DIMENSIONS: int = 768
    EMBEDDING_TIMEOUT_SECONDS: float = 30.0
    EMBEDDING_MAX_RETRIES: int = 3

    # --- Chat LLM (Ollama) ----------------------------------------------------
    # Prefer OLLAMA_MODEL when set; CHAT_MODEL remains the documented default.
    CHAT_MODEL: str = "qwen3:8b"
    OLLAMA_MODEL: str | None = None
    # When False, requests send options.num_gpu=0 (CPU). When True, Ollama may
    # use GPU; OLLAMA_NUM_GPU optionally limits offloaded layers (None = all).
    OLLAMA_USE_GPU: bool = True
    OLLAMA_NUM_GPU: int | None = None
    OLLAMA_NUM_THREAD: int | None = None
    OLLAMA_NUM_CTX: int = 2048
    # qwen3 with thinking enabled can exceed 120s on CPU; 300s is a safe default.
    LLM_TIMEOUT_SECONDS: float = 300.0
    LLM_MAX_RETRIES: int = 3
    LLM_TEMPERATURE: float = 0.7

    # --- Vector retrieval -----------------------------------------------------
    TOP_K: int = 10
    SIMILARITY_THRESHOLD: float = 0.0

    # --- Prompt building ------------------------------------------------------
    MAX_CONTEXT_CHARS: int = 8000
    SYSTEM_PROMPT: str = (
        "You are a helpful assistant that answers questions using only the "
        "provided document excerpts. Reference chunk numbers when citing "
        "sources. If the excerpts do not contain enough information, say so "
        "clearly instead of guessing."
    )

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def ollama_host(self) -> str:
        """Effective Ollama base URL (`OLLAMA_HOST` overrides `OLLAMA_BASE_URL`)."""
        return (self.OLLAMA_HOST or self.OLLAMA_BASE_URL).rstrip("/")

    @property
    def ollama_chat_model(self) -> str:
        """Effective chat model (`OLLAMA_MODEL` overrides `CHAT_MODEL`)."""
        return self.OLLAMA_MODEL or self.CHAT_MODEL

    @property
    def ollama_execution_mode(self) -> str:
        """Human-readable execution mode for startup logs."""
        return "GPU enabled" if self.OLLAMA_USE_GPU else "CPU fallback"

    def build_ollama_runtime_options(self, *, temperature: float | None = None) -> dict[str, Any]:
        """Ollama `/api/chat` `options` for temperature, GPU offload, and threads.

        - `OLLAMA_USE_GPU=false` → `num_gpu=0` (force CPU; avoids CUDA OOM on small VRAM).
        - `OLLAMA_USE_GPU=true` + `OLLAMA_NUM_GPU` set → pass that layer count.
        - `OLLAMA_USE_GPU=true` + `OLLAMA_NUM_GPU` unset → omit `num_gpu` (Ollama default).
        """
        options: dict[str, Any] = {
            "temperature": self.LLM_TEMPERATURE if temperature is None else temperature,
            "num_ctx": self.OLLAMA_NUM_CTX,
        }
        if not self.OLLAMA_USE_GPU:
            options["num_gpu"] = 0
        elif self.OLLAMA_NUM_GPU is not None:
            options["num_gpu"] = self.OLLAMA_NUM_GPU
        if self.OLLAMA_NUM_THREAD is not None:
            options["num_thread"] = self.OLLAMA_NUM_THREAD
        return options

    @field_validator("LLM_TEMPERATURE")
    @classmethod
    def validate_llm_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 2.0:
            raise ValueError("LLM_TEMPERATURE must be between 0.0 and 2.0.")
        return value

    @field_validator("LLM_MAX_RETRIES")
    @classmethod
    def validate_llm_max_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("LLM_MAX_RETRIES must be non-negative.")
        return value

    @field_validator("OLLAMA_NUM_GPU")
    @classmethod
    def validate_ollama_num_gpu(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("OLLAMA_NUM_GPU must be >= 0.")
        return value

    @field_validator("OLLAMA_NUM_THREAD")
    @classmethod
    def validate_ollama_num_thread(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("OLLAMA_NUM_THREAD must be greater than 0 when set.")
        return value

    @field_validator("MAX_CONTEXT_CHARS")
    @classmethod
    def validate_max_context_chars(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("MAX_CONTEXT_CHARS must be greater than 0.")
        return value

    @field_validator("SIMILARITY_THRESHOLD")
    @classmethod
    def validate_similarity_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("SIMILARITY_THRESHOLD must be between 0.0 and 1.0.")
        return value

    @field_validator("TOP_K")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("TOP_K must be greater than 0.")
        return value

    @field_validator("EMBEDDING_DIMENSIONS")
    @classmethod
    def validate_embedding_dimensions(cls, value: int) -> int:
        # Must match app.models.embedding.EMBEDDING_DIM / VECTOR(n) column.
        if value != 768:
            raise ValueError(
                "EMBEDDING_DIMENSIONS must be 768 — the database embeddings table "
                "has a fixed VECTOR(768) column (nomic-embed-text)."
            )
        return value

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError(
                "DATABASE_URL is not set. Copy .env.example to .env and fill in "
                "your Supabase connection string."
            )

        if not value.startswith(("postgresql://", "postgresql+psycopg://")):
            raise ValueError(
                "DATABASE_URL must be a PostgreSQL URL starting with "
                "'postgresql://' or 'postgresql+psycopg://' "
                f"(got: {value.split('://', 1)[0]!r})"
            )

        if any(marker in value for marker in _PLACEHOLDER_MARKERS):
            raise ValueError(
                "DATABASE_URL still contains placeholder text from "
                ".env.example — replace it with your real connection string."
            )

        return value

    @property
    def masked_database_url(self) -> str:
        """DATABASE_URL with credentials redacted for safe logging."""
        parsed = urlparse(self.DATABASE_URL)
        if not parsed.hostname:
            return "<invalid DATABASE_URL>"
        user = parsed.username or ""
        netloc = f"{user}:****@{parsed.hostname}"
        if parsed.port:
            netloc += f":{parsed.port}"
        return urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))

    @property
    def async_database_url(self) -> str:
        """DATABASE_URL normalized to the async psycopg3 driver.

        `postgresql+psycopg://` is dialect-neutral in SQLAlchemy 2.x — the
        same URL works for both `create_engine` (sync) and
        `create_async_engine` (async); psycopg3 ships native asyncio support,
        so no separate async-only driver package (e.g. asyncpg) is needed.
        """
        if self.DATABASE_URL.startswith("postgresql://"):
            return self.DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
        return self.DATABASE_URL

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached settings singleton."""
    return Settings()
