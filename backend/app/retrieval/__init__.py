"""Vector similarity retrieval over pgvector embeddings.

Independent of chat generation and prompt building. Import submodules
directly to avoid pulling unnecessary dependencies when only ranking logic
is needed.
"""

__all__ = [
    "ranking",
    "retriever",
    "search",
]
