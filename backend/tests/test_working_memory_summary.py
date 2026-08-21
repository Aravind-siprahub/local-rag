"""Automated regression test suite for Working Memory Summary Layer."""
from __future__ import annotations

import uuid
from app.rag.memory_summarizer import summarize_session_history
from app.prompting.templates import format_user_prompt
from app.prompting.builder import PromptBuilder
from app.retrieval.ranking import RankedResult


def test_1_summarize_session_history():
    """Verify summarize_session_history compresses multi-turn messages into concise summary paragraph."""
    history = [
        {"role": "user", "content": "What frontend and backend are used in Talk to My Data?"},
        {"role": "assistant", "content": "Frontend: React with Vite. Backend: FastAPI."},
        {"role": "user", "content": "What ports does SipraOne use?"},
        {"role": "assistant", "content": "Frontend port: 8001. Backend port: 5000."},
    ]

    summary = summarize_session_history(history)
    assert "Talk to My Data" in summary
    assert "SipraOne" in summary
    assert len(summary.split()) <= 150


def test_2_format_user_prompt_with_working_memory():
    """Verify format_user_prompt injects <working_memory_summary> block when present."""
    context = "Frontend uses React. Backend uses FastAPI."
    question = "Can you summarize the architecture?"
    memory_summary = "Active Project: Talk to My Data. Discussed Topics: Tech Stack & Architecture."

    prompt_text = format_user_prompt(context, question, working_memory_summary=memory_summary)
    assert "<working_memory_summary>" in prompt_text
    assert memory_summary in prompt_text
    assert question in prompt_text


def test_3_prompt_builder_integration():
    """Verify PromptBuilder passes working_memory_summary through to final Prompt."""
    builder = PromptBuilder()
    chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="Frontend: React with Vite. Backend: FastAPI.",
        document_id=uuid.uuid4(),
        similarity_score=0.85,
        rank=1,
        document_title="PRD_Talk_to_My_Data.docx",
    )
    memory_summary = "Active Project: Talk to My Data."

    prompt = builder.build("What is the tech stack?", [chunk], working_memory_summary=memory_summary)
    assert "<working_memory_summary>" in prompt.user_prompt
    assert "Talk to My Data" in prompt.user_prompt


def test_4_token_savings_over_raw_history():
    """Verify working memory summary uses significantly fewer characters than raw history dump."""
    raw_history = [
        {"role": "user", "content": "Can you explain the detailed frontend and backend architecture of Talk to My Data including all packages and dependencies?"},
        {"role": "assistant", "content": "Talk to My Data uses a React frontend with Vite, TypeScript, and TailwindCSS for fast UI rendering. The backend is built with FastAPI in Python, providing asynchronous REST endpoints and pgvector similarity search."},
        {"role": "user", "content": "What about SipraOne?"},
        {"role": "assistant", "content": "SipraOne frontend runs on port 8001 using React + Vite, while the backend API runs on Node.js / Express on port 5000 managed by PM2."},
    ]

    summary = summarize_session_history(raw_history)
    raw_char_count = sum(len(m["content"]) for m in raw_history)
    summary_char_count = len(summary)

    assert summary_char_count < raw_char_count * 0.5
