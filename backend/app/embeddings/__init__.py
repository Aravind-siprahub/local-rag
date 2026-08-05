"""Embedding generation pipeline — Ollama client, generator, and worker.

Independent of chat, prompt building, and vector search retrieval.
Import submodules directly to avoid pulling the full ORM stack when only
the HTTP client is needed.
"""

__all__ = [
    "client",
    "generator",
    "worker",
]
