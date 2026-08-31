"""Unit tests for SearXNG Web Search Provider, WebSearchService, and SSRF security validations."""
import pytest
import httpx
from unittest.mock import AsyncMock, patch

from app.tools.web_search import SearXNGWebSearchProvider, WebSearchError, WebSearchHit, WebSearchResult
from app.services.web_search_service import WebSearchService, format_web_context
from app.tools.page_fetcher import is_safe_url
from app.rag.intent_router import Route, classify


@pytest.mark.asyncio
async def test_searxng_provider_success():
    """Test SearXNGWebSearchProvider parses JSON response correctly."""
    mock_payload = {
        "results": [
            {
                "title": "Python Official Site",
                "url": "https://www.python.org/",
                "content": "Python is a high-level programming language.",
                "engine": "google",
                "publishedDate": "2026-01-01",
            },
            {
                "title": "Python Releases",
                "url": "https://www.python.org/downloads/",
                "content": "Download latest Python release.",
                "engine": "bing",
            },
        ]
    }

    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_payload
    mock_client.get.return_value = mock_response

    provider = SearXNGWebSearchProvider(base_url="http://localhost:8080", client=mock_client)
    res = await provider.search("python release", max_results=2)

    assert isinstance(res, WebSearchResult)
    assert res.provider == "searxng"
    assert len(res.hits) == 2
    assert res.hits[0].title == "Python Official Site"
    assert res.hits[0].url == "https://www.python.org/"
    assert res.hits[0].source == "python.org"
    assert res.hits[0].published_at == "2026-01-01"


@pytest.mark.asyncio
async def test_searxng_provider_empty_query():
    """Test empty search query raises WebSearchError."""
    provider = SearXNGWebSearchProvider(base_url="http://localhost:8080")
    with pytest.raises(WebSearchError, match="Search query must not be empty"):
        await provider.search("")


@pytest.mark.asyncio
async def test_searxng_provider_fallback_to_ddg_on_zero_results():
    """Test that when SearXNG yields 0 hits, it falls back to DuckDuckGo."""
    mock_searxng_response = AsyncMock()
    mock_searxng_response.status_code = 200
    mock_searxng_response.json.return_value = {"results": []}

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_searxng_response

    provider = SearXNGWebSearchProvider(base_url="http://localhost:8080", client=mock_client)

    with patch("app.tools.web_search.DuckDuckGoWebSearchProvider.search") as mock_ddg_search:
        mock_ddg_search.return_value = WebSearchResult(
            query="test",
            hits=[WebSearchHit(title="DDG Fallback", url="https://ddg.com", snippet="fallback snippet")],
            provider="duckduckgo",
        )
        res = await provider.search("test")
        assert res.provider == "duckduckgo"
        assert len(res.hits) == 1
        assert res.hits[0].title == "DDG Fallback"


def test_ssrf_protection_blocked_ips():
    """Verify SSRF protection blocks forbidden hosts and IP ranges."""
    assert is_safe_url("http://127.0.0.1/admin")[0] is False
    assert is_safe_url("http://localhost/keys")[0] is False
    assert is_safe_url("http://169.254.169.254/latest/meta-data/")[0] is False
    assert is_safe_url("http://10.0.0.1/internal")[0] is False
    assert is_safe_url("http://192.168.1.1/router")[0] is False
    assert is_safe_url("ftp://example.com/file")[0] is False


def test_ssrf_protection_allowed_urls():
    """Verify public URLs are allowed."""
    assert is_safe_url("https://www.python.org/")[0] is True
    assert is_safe_url("https://github.com/diegosouzapw/OmniRoute")[0] is True


@pytest.mark.asyncio
async def test_web_search_service_deduplication():
    """Test WebSearchService deduplicates URLs and formats context properly."""
    mock_provider = AsyncMock()
    mock_provider.search.return_value = WebSearchResult(
        query="python",
        provider="searxng",
        hits=[
            WebSearchHit(title="Title 1", url="https://python.org/doc", snippet="Snippet 1"),
            WebSearchHit(title="Title 1 Dup", url="https://python.org/doc/", snippet="Snippet Dup"),
            WebSearchHit(title="Title 2", url="https://python.org/download", snippet="Snippet 2"),
        ],
    )

    mock_fetcher = AsyncMock()
    mock_fetcher.fetch_and_extract.return_value = "Detailed extracted page content"

    service = WebSearchService(provider=mock_provider, page_fetcher=mock_fetcher)
    result = await service.search_web("python", max_results=5, fetch_pages=True)

    assert len(result.hits) == 2  # Deduplicated from 3 to 2
    assert result.hits[0].url == "https://python.org/doc"
    assert result.hits[1].url == "https://python.org/download"

    formatted = format_web_context(result.hits)
    assert "Result [1] Title 1 [python.org]" in formatted
    assert "Result [2] Title 2 [python.org]" in formatted


def test_intent_router_classification():
    """Verify intent router classifies local, web, and hybrid queries properly."""
    assert classify("What does our HR policy say about leave?") in (Route.DOCUMENT_QA, Route.DOCUMENT_DETAIL, Route.DOCUMENT_SUMMARY)
    assert classify("What is the latest Python release?") == Route.WEB
    assert classify("What are today's AI news headlines?") == Route.WEB
    assert classify("Compare our internal policy with current external regulations.") == Route.HYBRID
