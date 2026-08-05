"""LLM text generation clients.

Independent of retrieval, prompting orchestration, and chat APIs. Import
submodules directly when only the interface or response types are needed.
"""

__all__ = [
    "client",
    "ollama_client",
    "response",
]
