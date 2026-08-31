import pytest
import asyncio
from app.llm.sanitize import sanitize_response
from app.rag.query_normalizer import normalize_query

def test_query_normalization_preserves_entities():
    q = "Tell about working hours in Sipra Hub in HR framework"
    res = normalize_query(q)
    assert "working hours" in res["retrieval_query"].lower()
    assert "sipra" in res["retrieval_query"].lower()

def test_sanitize_preserves_sipra_hub_factual_response():
    question = "Tell me about the working hours in Sipra Hub according to the new HR framework."
    raw_llm_response = (
        "According to the HR framework document, working hours at Sipra Hub are recorded using timesheet logs "
        "across a 5-day work week, though specific shift hours (such as 9 AM to 5 PM) are not explicitly stated."
    )
    sanitized = sanitize_response(raw_llm_response, question=question)
    assert "working hours at sipra hub" in sanitized.lower()
    assert "The requested information is not found in the documents." not in sanitized

def test_sanitize_preserves_non_existent_query_fallback():
    question = "What is the official policy for astronaut space suits on the Mars colony?"
    raw_llm_response = "The requested information is not found in the documents."
    sanitized = sanitize_response(raw_llm_response, question=question)
    assert sanitized == "The requested information is not found in the documents."
