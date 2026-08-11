"""Unit tests for deterministic intent routing (Agent Router v1)."""
from __future__ import annotations

from app.rag.intent_router import Route, classify


class TestIntentRouter:
    def test_good_friday_routes_to_web(self) -> None:
        assert classify("When is Good Friday in 2026?") == Route.WEB

    def test_deployment_guide_routes_to_rag(self) -> None:
        assert (
            classify("What does Deployment_Guide.docx say about Nginx?")
            == Route.RAG
        )

    def test_percent_of_routes_to_calculator(self) -> None:
        assert classify("What is 18% of 45000?") == Route.CALCULATOR

    def test_what_is_python_routes_to_direct(self) -> None:
        assert classify("What is Python?") == Route.DIRECT

    def test_document_cue_beats_web_cue(self) -> None:
        assert (
            classify("According to my documents, what is the latest weather?")
            == Route.RAG
        )

    def test_arithmetic_beats_document_cue(self) -> None:
        assert classify("What is 10 + 5 according to my documents?") == Route.CALCULATOR
