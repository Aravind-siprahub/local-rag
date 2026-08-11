"""Unit tests for intent routing regression: project docs must be DOCUMENT_QA."""
from __future__ import annotations

from app.rag.intent_router import Route, classify

DOCS = [
    "Deployment_Guide.docx",
    "PRD_Talk_to_My_Data.docx",
    "Technology_Stack_Summary.docx",
    "AIRIS_Staging_Deployment_Guide_4.docx",
]


def test_talk_to_my_data_tech_stack_is_document_qa():
    route = classify(
        "what tech stack were using for talk to my data",
        document_titles=DOCS,
    )
    assert route == Route.DOCUMENT_QA


def test_frontend_backend_talk_to_my_data_is_document_qa():
    route = classify(
        "tell frontend and backend what using for talk to my data",
        document_titles=DOCS,
    )
    assert route == Route.DOCUMENT_QA


def test_airis_tech_stack_is_document_qa():
    route = classify(
        "AIRIS what tech stack were using tell",
        document_titles=DOCS,
    )
    assert route == Route.DOCUMENT_QA


def test_what_is_airis_stays_general_without_project_cues():
    route = classify("what is AIRIS?", document_titles=DOCS)
    assert route == Route.GENERAL_KNOWLEDGE


def test_pm2_definition_stays_general():
    route = classify("what is PM2?", document_titles=DOCS)
    assert route == Route.GENERAL_KNOWLEDGE


def test_deployment_document_pm2_is_document_qa():
    route = classify(
        "what does the deployment document say about PM2?",
        document_titles=DOCS,
    )
    assert route == Route.DOCUMENT_QA


def test_leave_policy_is_document_qa():
    route = classify("leave policy what say", document_titles=DOCS)
    assert route == Route.DOCUMENT_QA


def test_earth_stays_general():
    route = classify("earth is 2 planet or 3 planet", document_titles=DOCS)
    assert route == Route.GENERAL_KNOWLEDGE


def test_anaphoric_tech_stack_uses_conversation_context():
    route = classify(
        "what tech stack were using?",
        document_titles=DOCS,
        context_texts=["What is Talk to My Data?"],
    )
    assert route == Route.DOCUMENT_QA


def test_without_document_titles_project_query_not_forced():
    # Without corpus awareness, do not invent DOCUMENT_QA from hard-coded brand names alone.
    route = classify("what tech stack were using for talk to my data")
    assert route == Route.GENERAL_KNOWLEDGE
