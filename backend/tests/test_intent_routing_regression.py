"""Regression test for intent routing of software release version queries vs document metadata."""
import pytest
from app.rag.intent_router import route_question, Route

def test_python_version_query_routing():
    query = "What is the current latest release version of Python in 2026?"
    res = route_question(query)
    # Must NOT route to DOCUMENT_METADATA
    assert res.route != Route.DOCUMENT_METADATA
    assert res.route in (Route.WEB, Route.GENERAL_KNOWLEDGE)

def test_document_metadata_version_routing():
    query = "What is the document version of this uploaded file?"
    res = route_question(query)
    assert res.route == Route.DOCUMENT_METADATA
