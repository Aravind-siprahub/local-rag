import pytest
from app.rag.validator import topic_has_evidence, _is_substantive_answer, validate_and_reconcile_answer
from app.agent.planner import _is_followup_in_doc_context, Planner
from app.agent.state import AgentState
from app.rag.intent_router import classify, Route

def test_topic_has_evidence_synonyms():
    # Performance ratings synonym test: document mentions 'performance appraisal' and 'review'
    context = "The annual performance appraisal and review process evaluates employee contributions."
    assert topic_has_evidence("performance ratings", context) is True
    assert topic_has_evidence("performance outcomes", context) is True

def test_topic_has_evidence_partial_overlap():
    # Partial overlap (>= 50% substantive words)
    context = "We offer sick leave and personal time off to all full-time employees."
    assert topic_has_evidence("sick leave policy details", context) is True

def test_is_substantive_answer():
    # Meaningful answers should be marked substantive
    good_answer = "The performance evaluation cycle runs bi-annually with ratings from 1 to 5."
    assert _is_substantive_answer(good_answer) is True

    # Refusals should NOT be substantive
    refusal1 = "I couldn't find information about this in the document."
    assert _is_substantive_answer(refusal1) is False

    refusal2 = "The provided document does not specify the ratings."
    assert _is_substantive_answer(refusal2) is False

def test_validate_and_reconcile_preserves_substantive_answer():
    # If the LLM produces a substantive answer, validator does not overwrite it
    # even if keywords in question don't exactly match context
    question = "What about the performance ratings?"
    llm_answer = (
        "Based on the appraisal section, performance is rated across three tiers: "
        "Exceeds Expectations, Meets Expectations, and Needs Improvement."
    )
    # Context with slightly different wording
    context = "The employee appraisal section outlines tier benchmarks for employee evaluations."
    import uuid
    from app.retrieval.ranking import RankedResult

    chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text=context,
        document_id=uuid.uuid4(),
        similarity_score=0.9,
        rank=1,
        document_title="HR Policy",
    )
    reconciled = validate_and_reconcile_answer(
        question=question,
        answer=llm_answer,
        context_chunks=[chunk],
    )
    assert reconciled == llm_answer
    assert "does not specify" not in reconciled

def test_planner_followup_in_doc_context():
    # When state has recent document discussion, a short sub-question is recognized as doc follow-up
    state = AgentState(
        user_query="what about ratings?",
        conversation_context=[
            {"role": "user", "content": "Tell me about the HR framework document"},
            {"role": "assistant", "content": "The HR framework document outlines company policies, leave, and appraisal."},
        ],
    )
    assert _is_followup_in_doc_context("what about ratings?", state) is True

    # Unrelated conversational queries without doc context should return False
    state_empty = AgentState(user_query="hello", conversation_context=[])
    assert _is_followup_in_doc_context("hello", state_empty) is False

def test_intent_router_subquestion_cues():
    # "what about performance ratings" should classify as DOCUMENT_QA due to new cue words
    route = classify("what about performance ratings?")
    assert route in (Route.DOCUMENT_QA, Route.RAG)
