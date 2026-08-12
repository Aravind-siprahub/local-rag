"""Web search package."""
from app.web_search.ddg import SearchResult, format_web_results_as_context, web_search

__all__ = ["web_search", "format_web_results_as_context", "SearchResult"]
