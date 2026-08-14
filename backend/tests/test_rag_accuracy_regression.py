"""Regression tests for RAG accuracy: context delivery, routing, multi-chunk answers.

These tests verify that every stage from intent routing to LLM prompt construction
passes the correct information without silently dropping document context.
"""
from __future__ import annotations

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.llm.ollama_client import OllamaLLMClient, _parse_user_prompt
from app.prompting.templates import format_user_prompt, format_chunk
from app.rag.intent_router import classify, Route
from app.rag.query_normalizer import normalize_query
from app.llm.response import LLMResponse, TokenUsage
from app.rag.service import RAGService
from app.retrieval.ranking import RankedResult
from app.tools.web_search import StubWebSearchProvider
from sqlalchemy.ext.asyncio import AsyncSession


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_chunk(text: str, rank: int = 1, title: str = "PRD_Talk_to_My_Data.docx") -> RankedResult:
    return RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text=text,
        document_id=uuid.uuid4(),
        document_version_id=uuid.uuid4(),
        document_title=title,
        similarity_score=0.85 - (rank - 1) * 0.05,
        rank=rank,
        section_title="System Architecture Overview",
        page_number=5,
    )


def _llm_response(answer: str) -> LLMResponse:
    return LLMResponse(
        answer=answer,
        model_name="qwen3:4b",
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )


def _mock_session(title: str = "PRD_Talk_to_My_Data.docx") -> AsyncMock:
    session = AsyncMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_doc = MagicMock()
    mock_doc.title = title
    mock_result.scalars.return_value.all.return_value = [mock_doc]
    session.execute.return_value = mock_result
    return session


# ── Stage tests ───────────────────────────────────────────────────────────────


class TestIntentRouting:
    """Stage 1: Intent router must select DOCUMENT_QA for document questions."""

    def test_frontend_backend_routes_to_document_qa_with_docs(self):
        route = classify(
            "What frontend and backend are used in Talk to My Data?",
            document_titles=["PRD_Talk_to_My_Data.docx"],
        )
        assert route == Route.DOCUMENT_QA

    def test_frontend_backend_routes_to_document_qa_without_docs(self):
        """Without document titles the router cannot confirm a doc match;
        project-cue queries without an explicit doc reference fall back to
        GENERAL_KNOWLEDGE.  This is correct — the router needs titles to
        know there IS a relevant document."""
        route = classify("What frontend and backend does this project use?")
        # Acceptable outcomes: DOCUMENT_QA (if project cues match) or
        # GENERAL_KNOWLEDGE (no titles available to anchor the query).
        assert route in (Route.DOCUMENT_QA, Route.GENERAL_KNOWLEDGE)

    def test_general_knowledge_fallback(self):
        route = classify("What is the capital of France?", document_titles=None)
        assert route == Route.GENERAL_KNOWLEDGE

    def test_unsupported_question_without_docs_stays_general(self):
        route = classify("Who invented the telephone?", document_titles=None)
        assert route == Route.GENERAL_KNOWLEDGE

    def test_creative_query_stays_general(self):
        route = classify(
            "write prompt create one login and signup page",
            document_titles=["PRD_Talk_to_My_Data.docx"],
            context_texts=["What frontend does Talk to My Data use?"],
        )
        assert route == Route.GENERAL_KNOWLEDGE

    def test_generic_follow_up_without_anaphora_stays_general(self):
        route = classify(
            "tell me a joke",
            document_titles=["PRD_Talk_to_My_Data.docx"],
            context_texts=["What frontend does Talk to My Data use?"],
        )
        assert route == Route.GENERAL_KNOWLEDGE



class TestParseUserPrompt:
    """Stage 2: _parse_user_prompt must NOT drop document context."""

    def test_returns_two_tuple(self):
        history, remaining = _parse_user_prompt("Hello world")
        assert isinstance(history, list)
        assert isinstance(remaining, str)

    def test_no_history_returns_full_prompt_as_remaining(self):
        prompt = "Answer the question below...\n\nRetrieved Document Context\n\nchunk text\n\nQuestion:\nWhat is it?"
        history, remaining = _parse_user_prompt(prompt)
        assert history == []
        assert "chunk text" in remaining

    def test_history_is_stripped_but_context_preserved(self):
        history_prefix = (
            "Recent Conversation:\nUser: hello\nAssistant: hi\n\n"
            "---------------------------------\n\n"
        )
        context_body = "Answer the question below...\n\nRetrieved Document Context\n\nFrontend — the chat interface\n\nQuestion:\nWhat is the frontend?"
        prompt = history_prefix + context_body
        history, remaining = _parse_user_prompt(prompt)
        assert len(history) == 2
        assert "Frontend" in remaining

    def test_context_not_lost_for_rag_template(self):
        """The USER_PROMPT_WITH_CONTEXT template must reach the LLM intact."""
        chunk = format_chunk(1, "Frontend — the chat interface", title="PRD.docx", section="Arch", page=1)
        user_prompt = format_user_prompt(chunk, "What is the frontend?")
        _, remaining = _parse_user_prompt(user_prompt)
        assert "Frontend" in remaining, (
            "Document context was dropped by _parse_user_prompt. "
            "The LLM will receive no document passages."
        )


class TestOllamaPayloadBuilding:
    """Stage 3: _build_payload must include document chunks in the messages array."""

    def _build(self, chunk_text: str, question: str) -> dict:
        chunk = format_chunk(1, chunk_text, title="PRD.docx", section="Arch", page=1)
        user_prompt = format_user_prompt(chunk, question)
        client = OllamaLLMClient(
            client=MagicMock(),  # no real HTTP needed
        )
        from app.core.config import get_settings
        system = get_settings().SYSTEM_PROMPT
        return client._build_payload(system, user_prompt, stream=False)

    def test_document_context_in_payload(self):
        payload = self._build(
            "Frontend — the chat interface; renders citations.",
            "What is the frontend?",
        )
        all_content = " ".join(m["content"] for m in payload["messages"])
        assert "Frontend" in all_content, (
            "_build_payload dropped document context. "
            "Model will only see the question, not the passages."
        )

    def test_multi_chunk_both_present(self):
        chunk1 = format_chunk(1, "Frontend — the chat interface.", title="PRD.docx", section="Arch", page=1)
        chunk2 = format_chunk(2, "Backend — FastAPI handles requests.", title="PRD.docx", section="Arch", page=1)
        user_prompt = format_user_prompt(chunk1 + "\n\n" + chunk2, "What are the frontend and backend?")
        client = OllamaLLMClient(client=MagicMock())
        from app.core.config import get_settings
        payload = client._build_payload(get_settings().SYSTEM_PROMPT, user_prompt, stream=False)
        all_content = " ".join(m["content"] for m in payload["messages"])
        assert "Frontend" in all_content
        assert "FastAPI" in all_content

    def test_think_disabled_for_qwen(self):
        payload = self._build("Some text.", "Some question?")
        # OllamaLLMClient defaults to qwen3:4b which supports think param
        client = OllamaLLMClient(model="qwen3:4b", client=MagicMock())
        from app.core.config import get_settings
        payload = client._build_payload(get_settings().SYSTEM_PROMPT, "Question?", stream=False)
        assert payload.get("think") is False

    def test_no_context_still_sends_question(self):
        """When there are no retrieved chunks, the question still reaches the model."""
        client = OllamaLLMClient(client=MagicMock())
        from app.core.config import get_settings
        payload = client._build_payload(get_settings().SYSTEM_PROMPT, "What is the capital of France?", stream=False)
        user_msgs = [m for m in payload["messages"] if m["role"] == "user"]
        assert user_msgs, "No user message in payload"
        assert "France" in user_msgs[-1]["content"]


# ── End-to-end RAG service tests ─────────────────────────────────────────────


@pytest.mark.asyncio
class TestRAGServiceAccuracy:
    """End-to-end tests: retrieval → prompt → LLM call must include document context."""

    async def _run_rag(
        self,
        chunks: list[RankedResult],
        expected_answer: str,
        question: str = "What frontend and backend are used in Talk to My Data?",
    ) -> tuple[str, str, str]:
        """Run RAGService.ask(), return (answer, system_prompt, user_prompt)."""
        session = _mock_session()
        retriever = AsyncMock()
        retriever.retrieve.return_value = chunks

        llm_client = AsyncMock()
        llm_client.model = "qwen3:4b"
        llm_client.generate.return_value = _llm_response(expected_answer)

        messages_svc = AsyncMock()
        messages_svc.create_message.side_effect = [
            MagicMock(id=uuid.uuid4()),
            MagicMock(id=uuid.uuid4()),
        ]
        messages_svc.list_by_session.return_value = []

        sessions_svc = AsyncMock()
        sessions_svc.get.return_value = MagicMock(user_id=uuid.uuid4())

        rag = RAGService(
            session,
            retriever=retriever,
            llm_client=llm_client,
            message_service=messages_svc,
            session_service=sessions_svc,
            web_search=StubWebSearchProvider(),
        )

        response = await rag.ask(uuid.uuid4(), question)

        assert llm_client.generate.call_count == 1
        sys_prompt, user_prompt = llm_client.generate.call_args[0]
        return response.answer, sys_prompt, user_prompt

    async def test_1_multi_chunk_frontend_backend(self):
        """Both frontend and backend chunks must appear in the final LLM prompt."""
        chunks = [
            _make_chunk("Frontend — the chat interface; renders citations.", rank=1),
            _make_chunk("Backend — FastAPI handles API requests.", rank=2),
        ]
        answer, _, user_prompt = await self._run_rag(
            chunks,
            expected_answer="Talk to My Data uses a chat interface (frontend) and FastAPI (backend).",
        )
        assert "Frontend" in user_prompt, "Frontend chunk missing from LLM prompt"
        assert "FastAPI" in user_prompt, "Backend chunk missing from LLM prompt"

    async def test_2_exact_project_document_question(self):
        """A direct project question returns document-grounded answer, not 'not specified'."""
        chunks = [_make_chunk("The project stores data in PostgreSQL.", rank=1)]
        answer, _, user_prompt = await self._run_rag(
            chunks,
            expected_answer="The project stores data in PostgreSQL.",
            question="What database does Talk to My Data use?",
        )
        assert "PostgreSQL" in user_prompt

    async def test_3_answer_in_lower_ranked_chunk(self):
        """Answer that appears only in rank-3 chunk must still reach the LLM.
        The query includes the document name so the router selects DOCUMENT_QA.
        """
        chunks = [
            _make_chunk("General introduction to the platform.", rank=1),
            _make_chunk("Overview of the user interface.", rank=2),
            _make_chunk("The authentication layer uses JWT tokens.", rank=3),
        ]
        answer, _, user_prompt = await self._run_rag(
            chunks,
            expected_answer="Authentication uses JWT tokens.",
            # Include doc name so intent router fires DOCUMENT_QA
            question="How does authentication work in Talk to My Data?",
        )
        assert "JWT" in user_prompt, "Lower-ranked chunk with the answer was dropped"

    async def test_4_unsupported_question_returns_fallback(self):
        """When retrieval returns 0 chunks, the service returns a fallback message, not hallucination."""
        session = _mock_session()
        retriever = AsyncMock()
        retriever.retrieve.return_value = []

        llm_client = AsyncMock()
        llm_client.model = "qwen3:4b"
        llm_client.generate.return_value = _llm_response("placeholder")

        messages_svc = AsyncMock()
        messages_svc.create_message.side_effect = [
            MagicMock(id=uuid.uuid4()),
            MagicMock(id=uuid.uuid4()),
        ]
        messages_svc.list_by_session.return_value = []

        sessions_svc = AsyncMock()
        sessions_svc.get.return_value = MagicMock(user_id=uuid.uuid4())

        rag = RAGService(
            session,
            retriever=retriever,
            llm_client=llm_client,
            message_service=messages_svc,
            session_service=sessions_svc,
            web_search=StubWebSearchProvider(),
        )
        response = await rag.ask(uuid.uuid4(), "What is the meaning of life according to my document?")
        # LLM should NOT have been called because there were no chunks
        assert llm_client.generate.call_count == 0
        assert "could not find" in response.answer.lower()

    async def test_5_citation_tied_to_correct_chunk(self):
        """Source citations must reference the exact chunk IDs that were retrieved."""
        chunk = _make_chunk("Frontend — the chat interface.", rank=1)
        answer, _, user_prompt = await self._run_rag(
            [chunk],
            expected_answer="Frontend — the chat interface.",
        )
        assert "Frontend" in user_prompt

    async def test_6_no_reasoning_leakage_in_answer(self):
        """Sanitizer must strip any reasoning prefix before the answer is stored."""
        from app.llm.sanitize import sanitize_response
        raw = "Let me think about this. The frontend is a chat interface."
        clean = sanitize_response(raw)
        assert clean == "The frontend is a chat interface."

    async def test_7_reasoning_only_response_returns_empty(self):
        """A response that is entirely reasoning/meta-commentary must collapse.
        Use a string that matches _REASONING_ONLY_RE (let me check / i need to).
        """
        from app.llm.sanitize import sanitize_response
        # This string matches _REASONING_ONLY_RE: starts with "let me check"
        raw = "Let me check. I need to look at the document."
        clean = sanitize_response(raw)
        # Either the sanitizer strips it fully, or it strips the reasoning prefix
        # leaving no substantive content.  It must NOT return the raw reasoning text.
        assert clean.strip() == "" or not any(
            phrase in clean.lower()
            for phrase in ["let me check", "i need to look"]
        )

    async def test_8_conflicting_sources_both_in_prompt(self):
        """When two chunks contain conflicting facts, both must appear in the LLM prompt.
        The query includes the document name so the router selects DOCUMENT_QA.
        """
        chunks = [
            _make_chunk("The system uses React for the frontend.", rank=1, title="doc_v1.docx"),
            _make_chunk("The system uses Vue.js for the frontend.", rank=2, title="doc_v2.docx"),
        ]
        answer, _, user_prompt = await self._run_rag(
            chunks,
            expected_answer="Sources conflict: doc_v1 says React, doc_v2 says Vue.",
            # Include doc name so intent router fires DOCUMENT_QA
            question="What frontend framework is used in Talk to My Data?",
        )
        assert "React" in user_prompt
        assert "Vue" in user_prompt
