"""Agent Router tools package."""
from __future__ import annotations

from app.tools.calculator import CalculatorError, CalculatorResult, calculate
from app.tools.web_search import (
    DuckDuckGoWebSearchProvider,
    StubWebSearchProvider,
    WebSearchError,
    WebSearchHit,
    WebSearchProvider,
    WebSearchResult,
    get_web_search_provider,
)

__all__ = [
    "CalculatorError",
    "CalculatorResult",
    "calculate",
    "DuckDuckGoWebSearchProvider",
    "StubWebSearchProvider",
    "WebSearchError",
    "WebSearchHit",
    "WebSearchProvider",
    "WebSearchResult",
    "get_web_search_provider",
]
