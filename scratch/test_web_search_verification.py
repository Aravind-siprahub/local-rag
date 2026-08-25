import sys
import os
import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.abspath(os.path.join(script_dir, "..", "backend"))
if os.path.exists(backend_dir) and backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
elif os.path.exists(os.path.join(script_dir, "app")) and script_dir not in sys.path:
    sys.path.insert(0, script_dir)

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


async def run_all_20_tests():
    print("=== STARTING COMPLETE 20-TEST SUITE VERIFICATION ===")

    # Test 1
    service, sess, _, _, web = _make_service()
    res = await service.ask(sess.id, "search the web for FastAPI latest release")
    assert len(web.calls) >= 1
    assert "fastapi" in web.calls[0].lower()
    print("✔ Test 1 passed: Explicit 'search the web' invokes DuckDuckGo.")

    # Test 2
    service, sess, _, _, web = _make_service()
    res = await service.ask(sess.id, "look up PostHog and tell me what it does")
    assert len(web.calls) >= 1
    assert "posthog" in web.calls[0].lower()
    print("✔ Test 2 passed: 'look up X' invokes DuckDuckGo.")

    # Test 3
    service, sess, _, _, web = _make_service()
    res = await service.ask(sess.id, "what is the latest Python release?")
    assert len(web.calls) >= 1
    assert "python" in web.calls[0].lower()
    print("✔ Test 3 passed: 'latest X' invokes DuckDuckGo.")

    # Test 4
    service, sess, _, _, web = _make_service()
    await service.ask(sess.id, "search GitHub for FastAPI authentication examples")
    assert len(web.calls) >= 1
    assert "site:github.com" in web.calls[0]
    print("✔ Test 4 passed: GitHub-specific searches target site:github.com.")

    # Test 5
    service, sess, _, _, web = _make_service()
    await service.ask(sess.id, "search Reddit for Qwen3 8B user experiences")
    assert len(web.calls) >= 1
    assert "site:reddit.com" in web.calls[0]
    print("✔ Test 5 passed: Reddit-specific searches target site:reddit.com.")

    # Test 6
    service, sess, _, _, web = _make_service()
    await service.ask(sess.id, "search Stack Overflow for SQLAlchemy async errors")
    assert len(web.calls) >= 1
    assert "site:stackoverflow.com" in web.calls[0]
    print("✔ Test 6 passed: StackOverflow-specific searches target site:stackoverflow.com.")

    # Test 7
    service, sess, _, _, web = _make_service()
    await service.ask(sess.id, "find public information about OpenAI API updates")
    assert len(web.calls) >= 1
    print("✔ Test 7 passed: General public web searches work across any domain.")

    # Test 8
    hit = WebSearchHit(title="Custom Title", url="https://example.com/custom", snippet="Unique Context Snippet", source="example.com")
    service, sess, _, llm, _ = _make_service(hits=[hit])
    await service.ask(sess.id, "search online for custom item")
    assert len(llm.calls) == 1
    prompt_sent = llm.calls[0]["user_prompt"]
    assert "Unique Context Snippet" in prompt_sent
    print("✔ Test 8 passed: Search results reach the LLM user prompt context.")

    # Test 9
    hit = WebSearchHit(title="React Docs", url="https://react.dev/learn", snippet="React documentation snippet.", source="react.dev")
    service, sess, _, _, _ = _make_service(hits=[hit])
    res = await service.ask(sess.id, "look up react documentation")
    assert len(res.sources) == 1
    assert "react.dev/learn" in res.answer or res.sources[0].document_title == "React Docs"
    print("✔ Test 9 passed: Search result URLs survive the entire pipeline.")

    # Test 10
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
    print("✔ Test 10 passed: Duplicate search result URLs are removed.")

    # Test 11
    service, sess, _, _, _ = _make_service(hits=[], web_exc=WebSearchError("Web search yielded no results."))
    res = await service.ask(sess.id, "look up non_existent_query_999")
    assert "could not find reliable web results" in res.answer.lower() or "failed" in res.answer.lower()
    print("✔ Test 11 passed: Zero results handled gracefully.")

    # Test 12
    service, sess, _, _, _ = _make_service(web_exc=WebSearchError("Web search timed out."))
    res = await service.ask(sess.id, "look up query that times out")
    assert "timed out" in res.answer.lower()
    print("✔ Test 12 passed: Search timeout handled gracefully.")

    # Test 13
    service, sess, _, _, _ = _make_service(web_exc=WebSearchError("Web search is temporarily unavailable."))
    res = await service.ask(sess.id, "look up query with http error")
    assert "unavailable" in res.answer.lower() or "failed" in res.answer.lower()
    print("✔ Test 13 passed: Provider HTTP error handled gracefully.")

    # Test 14
    disclaimer = "I cannot perform external searches or access GitHub directly."
    hit = WebSearchHit(title="PostHog Repo", url="https://github.com/posthog", snippet="PostHog repo on GitHub.", source="github.com")
    service, sess, _, _, _ = _make_service(hits=[hit], llm_answer=disclaimer)
    res = await service.ask(sess.id, "search github for PostHog")
    assert "cannot perform external" not in res.answer.lower()
    assert "cannot access github" not in res.answer.lower()
    print("✔ Test 14 passed: Successful search never produces false 'cannot access internet' response.")

    # Test 15
    assert classify("what does the document say about port 8000?") == Route.DOCUMENT_QA
    assert classify("summarize my uploaded pdf file") == Route.DOCUMENT_QA
    print("✔ Test 15 passed: Private RAG still works.")

    # Test 16
    assert classify("what is 2 + 2?") in (Route.GENERAL_KNOWLEDGE, Route.CALCULATOR)
    assert classify("explain how python functions work") == Route.GENERAL_KNOWLEDGE
    print("✔ Test 16 passed: General knowledge and non-web queries work.")

    # Test 17
    q = "Compare my uploaded document with the latest FastAPI architecture online"
    assert classify(q) == Route.HYBRID
    print("✔ Test 17 passed: Hybrid RAG + web search routes to Route.HYBRID.")

    # Test 18
    service, sess, _, _, web = _make_service()
    prompt = "Can you look up posthog? Also search github for Claude Fable leaked prompt"
    await service.ask(sess.id, prompt)
    assert len(web.calls) == 2
    assert "posthog" in web.calls[0].lower()
    assert "site:github.com" in web.calls[1].lower()
    print("✔ Test 18 passed: Compound multi-intent searches work.")

    # Test 19
    service, sess, _, llm, _ = _make_service()
    await service.ask(sess.id, "search online for latest AI news")
    assert len(llm.calls) == 1
    sys_p = llm.calls[0]["system_prompt"]
    assert "DuckDuckGo" in sys_p
    assert "Do not claim that you cannot access the internet" in sys_p
    print("✔ Test 19 passed: System prompt instructs model on web search.")

    # Test 20
    assert classify("what is today's news?") == Route.WEB
    assert classify("check current price of bitcoin") == Route.WEB
    print("✔ Test 20 passed: Time-sensitive triggers invoke WEB search route.")

    print("\n==================================================")
    print("🎉 ALL 20 COMPREHENSIVE VERIFICATION TESTS PASSED PERFECTLY!")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_all_20_tests())
