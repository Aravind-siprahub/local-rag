import pytest
from app.llm.sanitize import sanitize_response

def test_sanitize_does_not_override_valid_document_answers():
    """Verify sanitize_response does NOT override detailed document answers containing phrases like 'does not explicitly state'."""
    question = "tell about working hours in Sipra hub in HR framework"
    raw_llm_response = (
        "According to the HR framework document, working hours at Sipra Hub are tracked via 5-day timesheets. "
        "Although the document does not explicitly state fixed shift hours, daily tracking over 8 hours is recorded."
    )
    
    sanitized = sanitize_response(raw_llm_response, question=question)
    
    assert "working hours at sipra hub" in sanitized.lower()
    assert "The requested information is not found in the documents." not in sanitized
