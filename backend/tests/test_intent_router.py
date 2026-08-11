"""Unit tests for deterministic intent routing (Agent Router v1)."""
from __future__ import annotations

from app.rag.intent_router import Route, classify


class TestIntentRouter:
    def test_good_friday_routes_to_web(self) -> None:
        assert classify("weather today in London") == Route.WEB

    def test_deployment_guide_routes_to_rag(self) -> None:
        assert (
            classify("What does Deployment_Guide.docx say about Nginx?")
            == Route.DOCUMENT_QA
        )

    def test_percent_of_routes_to_calculator(self) -> None:
        assert classify("What is 18% of 45000?") == Route.CALCULATOR

    def test_what_is_python_routes_to_general_knowledge(self) -> None:
        assert classify("What is Python?") == Route.GENERAL_KNOWLEDGE

    def test_earth_question_routes_to_general_knowledge(self) -> None:
        assert classify("earth is 2 planet or 3 planet") == Route.GENERAL_KNOWLEDGE
        assert classify("earth which planet") == Route.GENERAL_KNOWLEDGE

    def test_greetings_route_to_generic_chat(self) -> None:
        assert classify("hello") == Route.GENERIC_CHAT
        assert classify("good morning") == Route.GENERIC_CHAT

    def test_document_list_routes_correctly(self) -> None:
        assert classify("what documents do I have?") == Route.DOCUMENT_LIST
        assert classify("what doc u have") == Route.DOCUMENT_LIST

    def test_document_metadata_routes_correctly(self) -> None:
        assert classify("when was PRD_Talk_to_My_Data.docx uploaded?") == Route.DOCUMENT_METADATA
        assert classify("when this file upload") == Route.DOCUMENT_METADATA

    def test_document_cue_beats_web_cue(self) -> None:
        assert (
            classify("According to my document, what is the policy?")
            == Route.DOCUMENT_QA
        )

    def test_arithmetic_beats_document_cue(self) -> None:
        assert classify("What is 10 + 5 according to my documents?") == Route.CALCULATOR

