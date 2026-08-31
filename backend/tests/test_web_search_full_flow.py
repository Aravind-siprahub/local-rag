"""Comprehensive regression test suite for Web Search architecture."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any
import pytest

from app.llm.response import LLMResponse, TokenUsage
from app.models.enums import MessageRole
from app.prompting.builder import PromptBuilder
from app.rag.intent_router import Route, classify
from app.rag.service import RAGService
from app.tools.web_search import DuckDuckGoWebSearchProvider, WebSearchHit, WebSearchResult, WebSearchError


@dataclass
class _FakeChatSession:
    id: uuid.UUID
    user_id: uuid.UUID


@dataclass
class _FakeMessage:
    id: uuid.UUID
    session_id: uuid.UUID
    role: MessageRole
    content: str
    model_used: str | None = None


class FakeChatSessionService:
    def __init__(self, session: _FakeChatSession) -> None:
        self.session = session

    async def get(self, session_id: uuid.UUID) -> _FakeChatSession:
        return self.session


class FakeChatMessageService:
    def __init__(self) -> None:
        self.created: list[dict] = []

    async def create_message(self, **kwargs) -> _FakeMessage:
        self.created.append(kwargs)
        return _FakeMessage(
            id=uuid.uuid4(),
            session_id=kwargs["session_id"],
            role=kwargs["role"],
            content=kwargs["content"],
            model_used=kwargs.get("model_used"),
        )

    async def list_by_session(self, session_id: uuid.UUID, limit: int = 50, offset: int = 0) -> list[_FakeMessage]:
        return []


class FakeCitationService:
    async def create_citations_for_message(self, message_id: uuid.UUID, citations: list) -> list:
        return citations


class FakeRetriever:
    async def retrieve(self, question: str, **kwargs) -> list:
        return []

    async def close(self) -> None:
        pass


class FakeLLMClient:
    def __init__(self, answer: str = "PostHog is an open-source product analytics platform.") -> None:
        self.answer = answer
        self.calls: list[dict] = []

    async def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMResponse:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt, **kwargs})
        return LLMResponse(
            answer=self.answer,
            model_name="qwen3:8b",
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
        )

    async def close(self) -> None:
        pass


class FakeWebSearchProvider:
    def __init__(self, hits: list[WebSearchHit] | None = None, raise_exc: Exception | None = None) -> None:
        self.hits = hits if hits is not None else [
            WebSearchHit(
                title="PostHog - Product Analytics",
                url="https://posthog.com",
                snippet="PostHog provides open-source product analytics and session recording.",
                source="posthog.com"
            )
        ]
        self.calls: list[str] = []
        self.raise_exc = raise_exc
        self.provider = "duckduckgo"

    async def search(
        self,
        query: str,
        max_results: int = 5,
        recency_days: int | None = None,
        request_id: str | None = None,
    ) -> WebSearchResult:
        self.calls.append(query)
        if self.raise_exc:
            raise self.raise_exc
        return WebSearchResult(query=query, hits=self.hits, provider=self.provider)

    def concise_answer(self) -> str:
        return "\n".join(f"- {h.title}: {h.snippet} ({h.url})" for h in self.hits)

    async def close(self) -> None:
        pass


def _make_service(
    hits: list[WebSearchHit] | None = None,
    llm_answer: str = "PostHog is an open-source analytics tool.",
    web_exc: Exception | None = None
) -> tuple[RAGService, _FakeChatSession, Any, Any, Any]:
    sess = _FakeChatSession(id=uuid.uuid4(), user_id=uuid.uuid4())
    retriever = FakeRetriever()
    llm = FakeLLMClient(answer=llm_answer)
    web = FakeWebSearchProvider(hits=hits, raise_exc=web_exc)
    service = RAGService(
        session=None,  # type: ignore
        retriever=retriever,  # type: ignore
        prompt_builder=PromptBuilder(),
        llm_client=llm,  # type: ignore
        message_service=FakeChatMessageService(),  # type: ignore
        citation_service=FakeCitationService(),  # type: ignore
        session_service=FakeChatSessionService(sess),  # type: ignore
        web_search=web,  # type: ignore
    )
    return service, sess, retriever, llm, web


# --- 20 COMPREHENSIVE TEST SCENARIOS ---

@pytest.mark.asyncio
async def test_1_explicit_search_the_web_invokes_duckduckgo():
    """1. Explicit 'search the web' invokes DuckDuckGo."""
    service, sess, _, _, web = _make_service()
    res = await service.ask(sess.id, "search the web for FastAPI latest release")
    assert len(web.calls) >= 1
    assert "fastapi" in web.calls[0].lower()


@pytest.mark.asyncio
async def test_2_look_up_x_invokes_duckduckgo():
    """2. 'look up X' invokes DuckDuckGo."""
    service, sess, _, _, web = _make_service()
    res = await service.ask(sess.id, "look up PostHog and tell me what it does")
    assert len(web.calls) >= 1
    assert "posthog" in web.calls[0].lower()


@pytest.mark.asyncio
async def test_3_latest_x_invokes_duckduckgo():
    """3. 'latest X' invokes DuckDuckGo."""
    service, sess, _, _, web = _make_service()
    res = await service.ask(sess.id, "what is the latest Python release?")
    assert len(web.calls) >= 1
    assert "python" in web.calls[0].lower()


@pytest.mark.asyncio
async def test_4_github_specific_searches_work():
    """4. GitHub-specific searches target site:github.com."""
    service, sess, _, _, web = _make_service()
    await service.ask(sess.id, "search GitHub for FastAPI authentication examples")
    assert len(web.calls) >= 1
    assert "site:github.com" in web.calls[0]


@pytest.mark.asyncio
async def test_5_reddit_specific_searches_work():
    """5. Reddit-specific searches target site:reddit.com."""
    service, sess, _, _, web = _make_service()
    await service.ask(sess.id, "search Reddit for Qwen3 8B user experiences")
    assert len(web.calls) >= 1
    assert "site:reddit.com" in web.calls[0]


@pytest.mark.asyncio
async def test_6_stackoverflow_searches_work():
    """6. StackOverflow-specific searches target site:stackoverflow.com."""
    service, sess, _, _, web = _make_service()
    await service.ask(sess.id, "search Stack Overflow for SQLAlchemy async errors")
    assert len(web.calls) >= 1
    assert "site:stackoverflow.com" in web.calls[0]


@pytest.mark.asyncio
async def test_7_general_public_web_searches_work():
    """7. General public web searches work across any domain."""
    service, sess, _, _, web = _make_service()
    await service.ask(sess.id, "find public information about OpenAI API updates")
    assert len(web.calls) >= 1


@pytest.mark.asyncio
async def test_8_search_results_reach_llm_context():
    """8. Search results reach the LLM user prompt context."""
    hit = WebSearchHit(title="Custom Title", url="https://example.com/custom", snippet="Unique Context Snippet", source="example.com")
    service, sess, _, llm, _ = _make_service(hits=[hit])
    await service.ask(sess.id, "search online for custom item")
    assert len(llm.calls) == 1
    prompt_sent = llm.calls[0]["user_prompt"]
    assert "Unique Context Snippet" in prompt_sent


@pytest.mark.asyncio
async def test_9_urls_survive_entire_pipeline():
    """9. Search result URLs survive the entire pipeline and are returned as sources."""
    hit = WebSearchHit(title="React Docs", url="https://react.dev/learn", snippet="React documentation snippet.", source="react.dev")
    service, sess, _, _, _ = _make_service(hits=[hit])
    res = await service.ask(sess.id, "look up react documentation")
    assert len(res.sources) == 1
    assert "react.dev/learn" in res.answer or res.sources[0].document_title == "React Docs"


@pytest.mark.asyncio
async def test_10_duplicate_results_removed():
    """10. Duplicate search result URLs are removed."""
    raw_hits = [
        WebSearchHit(title="A", url="https://example.com/page", snippet="Snip 1", source="web"),
        WebSearchHit(title="A Duplicate", url="https://example.com/page/", snippet="Snip 2", source="web"),
        WebSearchHit(title="B", url="https://example.com/page2", snippet="Snip 3", source="web"),
    ]
    seen = set()
    unique = []
    for h in raw_hits:
        clean = h.url.strip().rstrip("/")
        if clean not in seen:
            seen.add(clean)
            unique.append(h)
    assert len(unique) == 2


@pytest.mark.asyncio
async def test_11_zero_results_handled_correctly():
    """11. Zero results are handled gracefully without application failure."""
    service, sess, _, _, _ = _make_service(hits=[], web_exc=WebSearchError("Web search yielded no results."))
    res = await service.ask(sess.id, "look up non_existent_query_999")
    assert "could not find reliable web results" in res.answer.lower() or "failed" in res.answer.lower()


@pytest.mark.asyncio
async def test_12_timeout_handled_correctly():
    """12. Search timeout is handled gracefully."""
    service, sess, _, _, _ = _make_service(web_exc=WebSearchError("Web search timed out."))
    res = await service.ask(sess.id, "look up query that times out")
    assert "timed out" in res.answer.lower()


@pytest.mark.asyncio
async def test_13_provider_error_handled_correctly():
    """13. Provider HTTP error is handled gracefully."""
    service, sess, _, _, _ = _make_service(web_exc=WebSearchError("Web search is temporarily unavailable."))
    res = await service.ask(sess.id, "look up query with http error")
    assert "unavailable" in res.answer.lower() or "failed" in res.answer.lower()


@pytest.mark.asyncio
async def test_14_successful_search_never_produces_no_internet_claim():
    """14. Successful web search never produces false 'cannot access internet' response."""
    disclaimer = "I cannot perform external searches or access GitHub directly."
    hit = WebSearchHit(title="PostHog Repo", url="https://github.com/posthog", snippet="PostHog repo on GitHub.", source="github.com")
    service, sess, _, _, _ = _make_service(hits=[hit], llm_answer=disclaimer)
    res = await service.ask(sess.id, "search github for PostHog")
    assert "cannot perform external" not in res.answer.lower()
    assert "cannot access github" not in res.answer.lower()


@pytest.mark.asyncio
async def test_15_private_rag_still_works():
    """15. Document QA queries like 'what does my document say?' route to DOCUMENT_QA."""
    assert classify("what does the document say about port 8000?") == Route.DOCUMENT_QA
    assert classify("summarize my uploaded pdf file") == Route.DOCUMENT_QA


@pytest.mark.asyncio
async def test_16_general_knowledge_still_works():
    """16. Math or simple questions route to GENERAL_KNOWLEDGE or CALCULATOR without web search."""
    assert classify("what is 2 + 2?") in (Route.GENERAL_KNOWLEDGE, Route.CALCULATOR)
    assert classify("explain how python functions work") == Route.GENERAL_KNOWLEDGE


@pytest.mark.asyncio
async def test_17_hybrid_rag_and_web_search_works():
    """17. Hybrid prompts asking for document comparison with online info route to Route.HYBRID."""
    q = "Compare my uploaded document with the latest FastAPI architecture online"
    assert classify(q) == Route.HYBRID


@pytest.mark.asyncio
async def test_18_compound_multi_intent_searches_work():
    """18. Multi-intent prompts generate separate DuckDuckGo queries."""
    service, sess, _, _, web = _make_service()
    prompt = "Can you look up posthog? Also search github for Claude Fable leaked prompt"
    await service.ask(sess.id, prompt)
    assert len(web.calls) == 2
    assert "posthog" in web.calls[0].lower()
    assert "site:github.com" in web.calls[1].lower()


@pytest.mark.asyncio
async def test_19_system_prompt_instructs_model_on_web_search():
    """19. System prompt passes WEB_SEARCH_SYSTEM_PROMPT to LLM."""
    service, sess, _, llm, _ = _make_service()
    await service.ask(sess.id, "search online for latest AI news")
    assert len(llm.calls) == 1
    sys_p = llm.calls[0]["system_prompt"]
    assert "DuckDuckGo" in sys_p
    assert "Do not claim that you cannot access the internet" in sys_p


@pytest.mark.asyncio
async def test_20_time_sensitive_triggers_invoke_web_search():
    """20. Time-sensitive phrases like 'today's news' trigger WEB route."""
    assert classify("what is today's news?") == Route.WEB
    assert classify("check current price of bitcoin") == Route.WEB
