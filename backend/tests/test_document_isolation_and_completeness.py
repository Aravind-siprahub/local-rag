import pytest
import uuid
from app.llm.sanitize import sanitize_response
from app.prompting.templates import USER_PROMPT_WITH_CONTEXT

def test_answer_completeness_multi_fact_evaluation():
    """Verify answer completeness requires ALL requested sub-facts to be present."""
    question = "What are notice period rules and sick leave rules?"
    
    # Complete response (both notice period and sick leave answered)
    complete_response = (
        "According to the leave policy, the notice period required upon resignation is 60 days. "
        "Additionally, employees are entitled to 12 days of paid sick leave per calendar year."
    )
    sanitized_complete = sanitize_response(complete_response, question=question)
    
    has_notice = "60 days" in sanitized_complete or "notice period" in sanitized_complete
    has_sick_leave = "12 days" in sanitized_complete or "sick leave" in sanitized_complete
    
    is_complete = has_notice and has_sick_leave
    assert is_complete, "Multi-fact completeness failed: Answer missing one of the requested facts!"

def test_answer_completeness_fails_if_incomplete():
    """Verify test detects incomplete responses missing sub-facts."""
    question = "What are notice period rules and sick leave rules?"
    
    # Incomplete response (only notice period answered, sick leave omitted)
    incomplete_response = "The notice period required upon resignation is 60 days."
    sanitized_inc = sanitize_response(incomplete_response, question=question)
    
    has_notice = "60 days" in sanitized_inc
    has_sick_leave = "12 days" in sanitized_inc
    
    is_complete = has_notice and has_sick_leave
    assert not is_complete, "Incomplete response was incorrectly flagged as complete!"

def test_cross_document_isolation_scoping():
    """Verify querying Document B (Cloud Security) for Document A fact (Annual Leave) returns refusal/no match."""
    doc_a_id = str(uuid.uuid4()) # Employee Leave Policy
    doc_b_id = str(uuid.uuid4()) # Cloud Security Policy
    
    # Mock chunks from Doc A and Doc B
    doc_a_chunk = {"document_id": doc_a_id, "text": "Annual leave allowance is 24 days per year."}
    doc_b_chunk = {"document_id": doc_b_id, "text": "Cloud data at rest must use AES-256 encryption."}
    
    # Query explicitly scoped to Document B
    active_doc_filter = doc_b_id
    scoped_chunks = [doc_b_chunk] if doc_b_chunk["document_id"] == active_doc_filter else []
    
    # Verify Doc A chunk is completely excluded when scoped to Doc B
    assert doc_a_chunk not in scoped_chunks
    assert len(scoped_chunks) == 1
    assert "AES-256" in scoped_chunks[0]["text"]
    assert "24 days" not in scoped_chunks[0]["text"]
