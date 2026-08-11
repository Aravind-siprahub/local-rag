"""Unit tests for calculator and web-search tools."""
from __future__ import annotations

import pytest

from app.tools.calculator import CalculatorError, calculate
from app.tools.web_search import (
    StubWebSearchProvider,
    WebSearchError,
    WebSearchHit,
    WebSearchResult,
    get_web_search_provider,
)


class TestCalculator:
    def test_percent_of(self) -> None:
        result = calculate("What is 18% of 45000?")
        assert result.value == 8100.0
        assert "8100" in result.display

    def test_simple_addition(self) -> None:
        result = calculate("What is 10 + 5?")
        assert result.value == 15.0

    def test_rejects_unsafe_input(self) -> None:
        with pytest.raises(CalculatorError):
            calculate("__import__('os').system('id')")


class TestWebSearchProviders:
    @pytest.mark.asyncio
    async def test_stub_provider_no_network(self) -> None:
        provider = StubWebSearchProvider()
        result = await provider.search("When is Good Friday in 2026?")
        assert isinstance(result, WebSearchResult)
        assert result.query
        assert len(result.hits) >= 1
        assert isinstance(result.hits[0], WebSearchHit)
        assert result.hits[0].title
        assert result.hits[0].snippet

    def test_factory_returns_stub_when_configured(self) -> None:
        provider = get_web_search_provider(provider_name="stub")
        assert provider.__class__.__name__ == "StubWebSearchProvider"

    def test_factory_defaults_to_duckduckgo(self) -> None:
        provider = get_web_search_provider(provider_name="duckduckgo")
        assert provider.__class__.__name__ == "DuckDuckGoWebSearchProvider"

    def test_settings_default_provider_is_duckduckgo(self) -> None:
        from app.core.config import Settings

        assert Settings.model_fields["WEB_SEARCH_PROVIDER"].default == "duckduckgo"
