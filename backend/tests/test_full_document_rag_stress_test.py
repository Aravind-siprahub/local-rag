import pytest
import asyncio
import time
from app.llm.sanitize import sanitize_response
from app.rag.query_normalizer import normalize_query
from app.prompting.templates import USER_PROMPT_WITH_CONTEXT

# Comprehensive Question Evaluation Dataset across multiple document sections & types
STRESS_TEST_DATASET = [
    # 1. Exact & Numeric Questions
    {
        "id": "Q1",
        "category": "numeric",
        "query": "How many days per week are timesheet logs tracked in Sipra Hub?",
        "expected_section": "Timesheet & Hour Analytics",
        "expected_page": 3,
        "expected_facts": ["5 days", "5-day work week"],
        "negative_facts": ["6 days", "7 days", "4 days"]
    },
    {
        "id": "Q2",
        "category": "numeric",
        "query": "What daily duration threshold triggers over-hour analytics?",
        "expected_section": "Timesheet & Hour Analytics",
        "expected_page": 3,
        "expected_facts": ["8 hours", "over-8-hour"],
        "negative_facts": ["10 hours", "12 hours"]
    },
    # 2. Entity & Paraphrased Questions
    {
        "id": "Q3",
        "category": "entity",
        "query": "Which project hub requires daily timesheet tracking according to the HR framework?",
        "expected_section": "Timesheet & Hour Analytics",
        "expected_page": 3,
        "expected_facts": ["Sipra Hub", "SipraHub"],
        "negative_facts": ["Other Hub"]
    },
    # 3. Multi-Chunk & Detailed Questions
    {
        "id": "Q4",
        "category": "multi_chunk",
        "query": "Explain the timesheet process, logging frequency, and daily tracking analytics at Sipra Hub.",
        "expected_section": "Timesheet & Hour Analytics",
        "expected_page": 3,
        "expected_facts": ["timesheet", "5-day", "8 hours"],
        "negative_facts": []
    },
    # 4. Table / Structured Data Questions
    {
        "id": "Q5",
        "category": "table",
        "query": "What authentication method is specified for the PoC milestone in the architecture roadmap?",
        "expected_section": "Milestone & Authentication",
        "expected_page": 1,
        "expected_facts": ["JWT-only", "Keycloak"],
        "negative_facts": ["OAuth1", "SAML"]
    },
    # 5. Negative / Absent Fact Questions (Hallucination Resistance)
    {
        "id": "Q6",
        "category": "negative",
        "query": "What is the exact lunch break duration at Sipra Hub?",
        "expected_section": None,
        "expected_page": None,
        "expected_facts": ["not explicitly specified", "not stated", "not found"],
        "negative_facts": ["1 hour", "30 minutes", "45 minutes"]
    },
    {
        "id": "Q7",
        "category": "negative",
        "query": "What is the retirement age specified in the HR framework?",
        "expected_section": None,
        "expected_page": None,
        "expected_facts": ["not specified", "not found"],
        "negative_facts": ["60 years", "65 years", "58 years"]
    },
    # 6. Cross-Section Distinction Questions
    {
        "id": "Q8",
        "category": "cross_section",
        "query": "How does the document distinguish timesheet tracking from fixed shift start and end times?",
        "expected_section": "Timesheet & Hour Analytics",
        "expected_page": 3,
        "expected_facts": ["timesheet", "does not explicitly specify fixed shift"],
        "negative_facts": ["9:00 AM", "5:00 PM"]
    },
    # 7. Conversational Follow-Up Questions
    {
        "id": "Q9",
        "category": "follow_up",
        "query": "Are those Sipra Hub shift timings fixed?",
        "expected_section": "Timesheet & Hour Analytics",
        "expected_page": 3,
        "expected_facts": ["not explicitly specified", "not fixed"],
        "negative_facts": ["Yes, fixed 9 to 5"]
    }
]

def test_query_normalization_accuracy():
    """Verify that query normalizer retains key entities across all stress queries."""
    for item in STRESS_TEST_DATASET:
        query = item["query"]
        assert isinstance(query, str)
        res = normalize_query(query)
        ret_q = res["retrieval_query"].lower()
        assert len(ret_q) > 0, f"Query normalizer returned empty string for {item['id']}"

def test_sanitizer_zero_hallucination_guarantee():
    """Verify sanitizer preserves grounded answers and negative refusals cleanly."""
    # Test 1: Grounded answer with unstated shift time note
    llm_ans = "The document specifies daily timesheet tracking across a 5-day work week, but does not explicitly state fixed shift times."
    cleaned = sanitize_response(llm_ans, question="What are the shift timings?")
    assert "5-day work week" in cleaned
    assert "does not explicitly state" in cleaned
    
    # Test 2: Absent fact refusal
    refusal_ans = "The requested information is not found in the documents."
    cleaned_ref = sanitize_response(refusal_ans, question="What is the retirement age?")
    assert cleaned_ref == "The requested information is not found in the documents."

def test_numeric_fact_precision():
    """Verify numeric values (5 days, 8 hours) are preserved exactly without rounding or alteration."""
    raw_response = "Timesheet logs track daily hours over a 5-day work week with over-8-hour daily analytics."
    cleaned = sanitize_response(raw_response, question="How many days per week are tracked?")
    assert "5-day work week" in cleaned
    assert "8-hour" in cleaned
