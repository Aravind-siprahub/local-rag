"""Web search tool providers for Agent Router v1."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)


class WebSearchError(Exception):
    """Raised when web search fails in a controlled way."""


@dataclass(frozen=True)
class WebSearchHit:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class WebSearchResult:
    query: str
    hits: list[WebSearchHit] = field(default_factory=list)
    provider: str = "unknown"

    def concise_answer(self) -> str:
        """Format hits into a short answer string for the chat response."""
        if not self.hits:
            return (
                "I could not find reliable web results for that question right now. "
                "Please try again shortly."
            )
        lines: list[str] = []
        for idx, hit in enumerate(self.hits[:5], start=1):
            snippet = hit.snippet.strip() or hit.title
            url_part = f" ({hit.url})" if hit.url else ""
            lines.append(f"{idx}. {hit.title}: {snippet}{url_part}")
        return "Here is what I found:\n" + "\n".join(lines)


@runtime_checkable
class WebSearchProvider(Protocol):
    """Vendor-agnostic web search contract."""

    async def search(self, query: str) -> WebSearchResult: ...


class StubWebSearchProvider:
    """Deterministic provider for tests and offline use — no network calls."""

    async def search(self, query: str) -> WebSearchResult:
        q = (query or "").strip() or "empty"
        logger.info("[WEB SEARCH] provider=stub query_len=%d", len(q))
        return WebSearchResult(
            query=q,
            provider="stub",
            hits=[
                WebSearchHit(
                    title="Stub web result",
                    url="https://example.com/stub",
                    snippet=(
                        f"Stub search result for: {q}. "
                        "Configure WEB_SEARCH_PROVIDER=duckduckgo for live results."
                    ),
                )
            ],
        )


class DuckDuckGoWebSearchProvider:
    """DuckDuckGo Instant Answer API — no API key required."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 8.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self._client = client
        self._owns_client = client is None

    async def search(self, query: str) -> WebSearchResult:
        q = (query or "").strip()
        if not q:
            raise WebSearchError("Search query must not be empty.")

        logger.info("[WEB SEARCH] provider=duckduckgo query_len=%d", len(q))
        params = {
            "q": q,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
        try:
            client = await self._get_client()
            response = await client.get("https://api.duckduckgo.com/", params=params)
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as exc:
            logger.warning("[WEB SEARCH] duckduckgo timeout")
            raise WebSearchError("Web search timed out. Please try again.") from exc
        except httpx.HTTPError as exc:
            logger.warning("[WEB SEARCH] duckduckgo http error")
            raise WebSearchError("Web search is temporarily unavailable.") from exc
        except Exception as exc:
            logger.exception("[WEB SEARCH] duckduckgo unexpected error")
            raise WebSearchError("Web search failed.") from exc

        hits = _hits_from_duckduckgo(payload)
        return WebSearchResult(query=q, hits=hits, provider="duckduckgo")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds),
                headers={"User-Agent": "local-rag-agent-router/1.0"},
            )
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None


def _hits_from_duckduckgo(payload: dict) -> list[WebSearchHit]:
    hits: list[WebSearchHit] = []

    abstract = (payload.get("AbstractText") or "").strip()
    abstract_url = (payload.get("AbstractURL") or "").strip()
    heading = (payload.get("Heading") or payload.get("AbstractSource") or "DuckDuckGo").strip()
    if abstract:
        hits.append(
            WebSearchHit(
                title=heading or "Result",
                url=abstract_url,
                snippet=abstract,
            )
        )

    answer = (payload.get("Answer") or "").strip()
    if answer and answer != abstract:
        hits.append(
            WebSearchHit(
                title=heading or "Answer",
                url=abstract_url,
                snippet=answer,
            )
        )

    for topic in payload.get("RelatedTopics") or []:
        if len(hits) >= 5:
            break
        if not isinstance(topic, dict):
            continue
        if "Topics" in topic:
            for nested in topic.get("Topics") or []:
                if len(hits) >= 5:
                    break
                hit = _hit_from_related(nested)
                if hit:
                    hits.append(hit)
            continue
        hit = _hit_from_related(topic)
        if hit:
            hits.append(hit)

    return hits


def _hit_from_related(topic: dict) -> WebSearchHit | None:
    if not isinstance(topic, dict):
        return None
    text = (topic.get("Text") or "").strip()
    url = (topic.get("FirstURL") or "").strip()
    if not text:
        return None
    title = text.split(" - ", 1)[0][:120]
    return WebSearchHit(title=title, url=url, snippet=text)


def get_web_search_provider(
    *,
    provider_name: str | None = None,
    timeout_seconds: float | None = None,
) -> WebSearchProvider:
    """Factory: ``duckduckgo`` (default) or ``stub``."""
    from app.core.config import get_settings

    settings = get_settings()
    name = (provider_name or getattr(settings, "WEB_SEARCH_PROVIDER", None) or "duckduckgo")
    name = str(name).strip().lower()
    timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else float(getattr(settings, "WEB_SEARCH_TIMEOUT_SECONDS", 8.0))
    )

    if name == "stub":
        logger.info("[WEB SEARCH] using provider=stub")
        return StubWebSearchProvider()
    if name == "duckduckgo":
        logger.info("[WEB SEARCH] using provider=duckduckgo")
        return DuckDuckGoWebSearchProvider(timeout_seconds=timeout)

    logger.warning("[WEB SEARCH] unknown provider=%s; falling back to duckduckgo", name)
    return DuckDuckGoWebSearchProvider(timeout_seconds=timeout)
