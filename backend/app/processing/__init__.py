"""Document text processing pipeline — parse, clean, and chunk uploaded files.

Independent of embeddings, Ollama, and vector search. Import submodules
directly (e.g. `from app.processing.parser import parse_file`) to avoid
pulling the full ORM stack when only pure functions are needed.
"""

__all__ = [
    "cleaner",
    "chunker",
    "parser",
    "processor",
]
