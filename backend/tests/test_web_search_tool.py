"""Comprehensive test suite for Real-Time Web Search Tool, SSRF Protection, and Page Content Extraction."""
from __future__ import annotations

import uuid
import pytest
from unittest.mock import AsyncMock, patch

from app.agent.tools.base import ToolInput
from app.agent.tools.web_tool import WebSearchTool
from app.tools.page_fetcher import PageFetcher, extract_readable_content, is_safe_url
from app.tools.web_search import (
    DuckDuckGoWebSearchProvider,
    StubWebSearchProvider,
    WebSearchError,
    WebSearchHit,
    WebSearchResult,
)
from app.llm.response import LLMResponse
from app.agent.planner import Planner
from app.agent.state import AgentState, AgentStatus
from app.agent.orchestrator import AgentOrchestrator


@pytest.mark.asyncio
async def test_successful_web_search() -> None:
    """Requirement 11, Test 1: Successful web search returning structured results."""
    stub_provider = StubWebSearchProvider()
    tool = WebSearchTool(web_search=stub_provider)

    tool_input = ToolInput(query="Python 3.12 features", parameters={"max_results": 3, "fetch_pages": False})
    output = await tool.execute(tool_input)

    assert output.success is True
    assert output.data["count"] == 1
    assert output.data["provider"] == "stub"
    hit = output.data["hits"][0]
    assert hit["title"] == "Stub web result"
    assert hit["url"] == "https://example.com/stub"
    assert "Stub search result" in hit["snippet"]
    assert hit["source"] == "example.com"
    assert len(output.data["evidence"]) == 1
    assert output.data["evidence"][0]["url"] == "https://example.com/stub"


@pytest.mark.asyncio
async def test_empty_search_results() -> None:
    """Requirement 11, Test 2: Search yielding empty results handled gracefully."""
    mock_provider = AsyncMock()
    mock_provider.search.return_value = WebSearchResult(query="empty query", hits=[], provider="mock")
    tool = WebSearchTool(web_search=mock_provider)

    tool_input = ToolInput(query="empty query", parameters={"fetch_pages": False})
    output = await tool.execute(tool_input)

    assert output.success is True
    assert output.data["count"] == 0
    assert output.data["hits"] == []
    assert output.data["evidence"] == []


@pytest.mark.asyncio
async def test_network_failure_handling() -> None:
    """Requirement 11, Test 3: Network failure handling without backend crash."""
    mock_provider = AsyncMock()
    mock_provider.search.side_effect = WebSearchError("Network failure connecting to search backend")
    tool = WebSearchTool(web_search=mock_provider)

    tool_input = ToolInput(query="test query")
    output = await tool.execute(tool_input)

    assert output.success is False
    assert output.error is not None and "Network failure" in output.error
    assert output.data["hits"] == []


@pytest.mark.asyncio
async def test_search_timeout_handling() -> None:
    """Requirement 11, Test 4: Timeout exception handled gracefully."""
    mock_provider = AsyncMock()
    mock_provider.search.side_effect = WebSearchError("Web search timed out. Please try again.")
    tool = WebSearchTool(web_search=mock_provider)

    tool_input = ToolInput(query="timeout query")
    output = await tool.execute(tool_input)

    assert output.success is False
    assert output.error is not None and "timed out" in output.error


@pytest.mark.asyncio
async def test_ssrf_and_private_ip_protection() -> None:
    """Requirement 11, Test 5: SSRF protection blocking private IPs, localhost, and metadata endpoints."""
    forbidden_urls = [
        "http://localhost:8000/api",
        "http://127.0.0.1/admin",
        "http://10.0.0.1/secret",
        "http://172.16.0.5/internal",
        "http://192.168.1.1/router",
        "http://169.254.169.254/latest/meta-data/",  # AWS metadata endpoint
        "http://[::1]/local",
    ]

    for url in forbidden_urls:
        is_safe, reason = is_safe_url(url)
        assert is_safe is False, f"URL {url} should be blocked for SSRF, but passed. Reason: {reason}"

    # Test valid external public URL format
    valid_url = "https://example.com/page"
    is_safe, _ = is_safe_url(valid_url)
    assert is_safe is True


def test_html_page_content_extraction() -> None:
    """Requirement 11, Test 6: HTML cleaner strips boilerplate (scripts, nav, styles)."""
    raw_html = """
    <html>
        <head>
            <style>body { color: red; }</style>
            <script>console.log("ad script");</script>
        </head>
        <body>
            <header><nav><a href="#">Home</a></nav></header>
            <main>
                <h1>Main Headline</h1>
                <p>This is the primary readable article content for testing.</p>
            </main>
            <footer>Copyright 2026</footer>
        </body>
    </html>
    """
    extracted = extract_readable_content(raw_html)

    assert "Main Headline" in extracted
    assert "primary readable article content" in extracted
    assert "console.log" not in extracted
    assert "color: red" not in extracted
    assert "Copyright 2026" not in extracted


@pytest.mark.asyncio
async def test_page_fetch_failure_tolerance() -> None:
    """Requirement 11, Test 7: Failure to fetch 1 webpage does not crash web search tool."""
    mock_provider = AsyncMock()
    mock_provider.search.return_value = WebSearchResult(
        query="python news",
        provider="mock",
        hits=[
            WebSearchHit(title="Good Page", url="https://example.com/good", snippet="Good snippet"),
            WebSearchHit(title="Broken Page", url="https://example.com/broken", snippet="Broken snippet"),
        ],
    )

    mock_fetcher = AsyncMock()
    # Good page succeeds, broken page returns None
    mock_fetcher.fetch_and_extract.side_effect = [
        "Fetched clean article text from good page.",
        None,
    ]

    tool = WebSearchTool(web_search=mock_provider, page_fetcher=mock_fetcher)
    output = await tool.execute(ToolInput(query="python news", parameters={"fetch_pages": True}))

    assert output.success is True
    assert output.data["count"] == 2
    hits = output.data["hits"]
    assert hits[0]["content"] == "Fetched clean article text from good page."
    assert hits[1]["content"] is None or hits[1]["content"] == "Broken snippet"


@pytest.mark.asyncio
async def test_duplicate_url_deduplication() -> None:
    """Requirement 11, Test 8: Duplicate URLs are deduplicated by provider."""
    provider = DuckDuckGoWebSearchProvider()
    
    with patch.object(provider, "_get_client", new_callable=AsyncMock) as mock_get_client:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json = lambda: {
            "Heading": "Python Org",
            "AbstractText": "Python Official Site",
            "AbstractURL": "https://www.python.org",
            "RelatedTopics": [
                {"Text": "Python Docs", "FirstURL": "https://www.python.org"},
                {"Text": "Python Download", "FirstURL": "https://www.python.org/downloads"},
            ],
        }
        mock_client_inst = AsyncMock()
        mock_client_inst.get.return_value = mock_response
        mock_get_client.return_value = mock_client_inst

        result = await provider.search("python", max_results=5)
        urls = [h.url for h in result.hits]
        assert len(urls) == len(set(urls))
        assert "https://www.python.org" in urls
        assert "https://www.python.org/downloads" in urls


@pytest.mark.asyncio
async def test_source_url_preservation() -> None:
    """Requirement 11, Test 9: Source URLs preserved in hits and evidence."""
    mock_provider = AsyncMock()
    mock_provider.search.return_value = WebSearchResult(
        query="latest news",
        provider="mock",
        hits=[
            WebSearchHit(title="Tech Article", url="https://techcrunch.com/article1", snippet="Snippet text", source="techcrunch.com")
        ],
    )
    tool = WebSearchTool(web_search=mock_provider, page_fetcher=AsyncMock())

    output = await tool.execute(ToolInput(query="latest news", parameters={"fetch_pages": False}))
    assert output.success is True
    hit = output.data["hits"][0]
    evidence = output.data["evidence"][0]

    assert hit["url"] == "https://techcrunch.com/article1"
    assert hit["source"] == "techcrunch.com"
    assert evidence["url"] == "https://techcrunch.com/article1"


@pytest.mark.asyncio
async def test_agent_planner_web_search_trigger() -> None:
    """Requirement 11, Test 10: Planner triggers web_search tool for live queries."""
    planner = Planner()
    state = AgentState(user_query="What is the latest OpenAI news today?")

    plan = await planner.create_plan(state)
    assert len(plan) >= 1
    assert plan[0].target_tool == "web_search"


@pytest.mark.asyncio
async def test_agent_orchestrator_web_search_execution(db_session) -> None:
    """Requirement 11, Test 11: Agent orchestrator executes web_search and incorporates evidence."""
    from app.agent.state import VerificationOutcome

    mock_web = AsyncMock()
    mock_web.search.return_value = WebSearchResult(
        query="latest news",
        provider="mock",
        hits=[
            WebSearchHit(
                title="Live News Report",
                url="https://news.example.com/live",
                snippet="OpenAI announced new features today.",
                source="news.example.com",
            )
        ],
    )

    orchestrator = AgentOrchestrator(db_session, web_search=mock_web)
    
    with patch.object(orchestrator.llm_client, "generate", new_callable=AsyncMock) as mock_generate, \
         patch.object(orchestrator.verifier, "verify_final_answer") as mock_verify:

        mock_generate.return_value = LLMResponse(
            answer="Based on recent reports from news.example.com, OpenAI announced new features today.",
            model_name="qwen3:8b",
        )
        mock_verify.return_value = VerificationOutcome(is_valid=True)

        state = await orchestrator.run(query="What is the latest OpenAI news today?")
        
        assert state.status == AgentStatus.COMPLETED
        assert state.final_answer is not None
        assert "OpenAI" in state.final_answer
        assert mock_web.search.called
