import pytest
import asyncio
from app.llm.sanitize import sanitize_response

def test_sanitize_preserves_factual_answers():
    question = "Tell about working hours in Sipra Hub in HR framework"
    raw_answer = "According to the HR framework document, working hours at Sipra Hub are logged via timesheets across 5 days, though specific shift times are not explicitly stated."
    cleaned = sanitize_response(raw_answer, question=question)
    assert "working hours at sipra hub" in cleaned.lower()
    assert "The requested information is not found in the documents." not in cleaned
