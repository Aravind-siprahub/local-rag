"""Live ASGI verification of Agent Router v1 via POST /api/chat (no network server)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import httpx
from httpx import ASGITransport

from app.api.dependencies import get_rag_service
from app.llm.response import LLMResponse, TokenUsage
from app.main import app
from app.models.enums import MessageRole
from app.prompting.builder import PromptBuilder
from app.rag.service import RAGService
from app.retrieval.ranking import RankedResult
from app.tools.web_search import WebSearchHit, WebSearchResult


@dataclass
class _Sess:
    id: uuid.UUID
    user_id: uuid.UUID


@dataclass
class _Msg:
    id: uuid.UUID
    session_id: uuid.UUID
    role: MessageRole
    content: str
    model_used: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None
    generation_time_ms: int | None = None


class _Messages:
    def __init__(self) -> None:
        self._msgs: list[_Msg] = []

    async def create_message(self, **kwargs) -> _Msg:
        msg = _Msg(
            id=uuid.uuid4(),
            session_id=kwargs["session_id"],
            role=kwargs["role"],
            content=kwargs["content"],
            model_used=kwargs.get("model_used"),
            prompt_tokens=kwargs.get("prompt_tokens"),
            completion_tokens=kwargs.get("completion_tokens"),
            latency_ms=kwargs.get("latency_ms"),
            generation_time_ms=kwargs.get("generation_time_ms"),
        )
        self._msgs.append(msg)
        return msg

    async def list_by_session(self, session_id: uuid.UUID, limit: int = 50) -> list[_Msg]:
        return [m for m in self._msgs if m.session_id == session_id][-limit:]


class _Sessions:
    def __init__(self, session: _Sess) -> None:
        self.session = session

    async def get(self, session_id: uuid.UUID) -> _Sess:
        return self.session


class _Citations:
    async def create_citations_for_message(self, message_id: uuid.UUID, citations: list) -> list:
        return citations


class TrackingRetriever:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def retrieve(self, question: str, **kwargs) -> list[RankedResult]:
        self.calls.append(question)
        return [
            RankedResult(
                chunk_id=uuid.uuid4(),
                chunk_text="Nginx is used as a reverse proxy in Deployment_Guide.docx.",
                document_id=uuid.uuid4(),
                document_version_id=uuid.uuid4(),
                document_title="Deployment_Guide.docx",
                similarity_score=0.91,
                rank=1,
                section_title="Nginx",
                page_number=3,
            )
        ]

    async def close(self) -> None:
        return None


class TrackingLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.model = "test-chat-model"

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        num_predict: int | None = None,
    ) -> LLMResponse:
        self.calls.append({"num_predict": num_predict, "user_prompt": user_prompt})
        return LLMResponse(
            answer="Nginx is configured as a reverse proxy.",
            model_name="test-chat-model",
            token_usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
            finish_reason="stop",
        )

    async def close(self) -> None:
        return None


class TrackingWebSearch:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def search(self, query: str) -> WebSearchResult:
        self.calls.append(query)
        return WebSearchResult(
            query=query,
            provider="fake",
            hits=[
                WebSearchHit(
                    title="Good Friday 2026",
                    url="https://example.com/good-friday-2026",
                    snippet="Good Friday in 2026 falls on Friday, 3 April 2026.",
                )
            ],
        )


def main() -> None:
    session = _Sess(id=uuid.uuid4(), user_id=uuid.uuid4())
    retriever = TrackingRetriever()
    llm = TrackingLLM()
    web = TrackingWebSearch()
    service = RAGService(
        session=None,
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        llm_client=llm,
        message_service=_Messages(),
        citation_service=_Citations(),
        session_service=_Sessions(session),
        web_search=web,
    )
    app.dependency_overrides[get_rag_service] = lambda: service

    transport = ASGITransport(app=app, raise_app_exceptions=True)
    with httpx.Client(transport=transport, base_url="http://test") as client:
        cases = [
            ("When is Good Friday in 2026?", "WEB"),
            ("What is 18% of 45000?", "CALCULATOR"),
            ("What is Python?", "DIRECT"),
            ("What does Deployment_Guide.docx say about Nginx?", "RAG"),
        ]
        print("=== ASGI POST /api/chat Agent Router verification ===")
        for question, expected in cases:
            before_ret, before_web, before_llm = len(retriever.calls), len(web.calls), len(llm.calls)
            response = client.post(
                "/api/chat",
                json={"session_id": str(session.id), "question": question},
            )
            body = response.json()
            print(f"\n[{expected}] status={response.status_code} model={body.get('model')}")
            print(f"  answer={(body.get('answer') or '')[:160]}")
            print(
                f"  delta retriever={len(retriever.calls)-before_ret} "
                f"web={len(web.calls)-before_web} llm={len(llm.calls)-before_llm}"
            )
            assert response.status_code == 200, body
            if expected == "WEB":
                assert len(web.calls) > before_web and len(retriever.calls) == before_ret
            elif expected == "CALCULATOR":
                assert "8100" in body["answer"] and len(retriever.calls) == before_ret
            elif expected == "DIRECT":
                assert len(retriever.calls) == before_ret and llm.calls[-1]["num_predict"] == 128
            elif expected == "RAG":
                assert len(retriever.calls) > before_ret and len(web.calls) == before_web

    app.dependency_overrides.clear()
    print("\nPASS: WEB bypassed RAG; RAG still calls retriever; CALCULATOR/DIRECT skip retrieval.")


if __name__ == "__main__":
    main()
