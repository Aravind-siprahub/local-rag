"""Regression tests for conversational RAG query normalization, heuristic ranking, and fulltext search."""
import uuid
import pytest
from app.rag.query_normalizer import normalize_query
from app.retrieval.ranking import _fallback_heuristic_rerank, _STOP_WORDS, RankedResult

def test_conversational_query_normalization():
    raw = 'Tell about working hours in Sipra hub'
    orig, norm, ret_q = normalize_query(raw)
    assert orig == raw
    assert ret_q == 'working hours in Sipra hub SipraHub' or 'working hours SipraHub Sipra hub' in ret_q or 'working hours' in ret_q
    assert 'tell about' not in ret_q.lower()

def test_heuristic_rerank_stop_words():
    assert "tell" in _STOP_WORDS
    assert "about" in _STOP_WORDS
    assert "explain" in _STOP_WORDS
    assert "give" in _STOP_WORDS

def test_heuristic_rerank_compound_matching():
    query = "Tell about working hours in Sipra hub"
    chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="The working hours in SipraHub are tracked through the timesheet feature.",
        document_id=uuid.uuid4(),
        similarity_score=0.35,
        rank=1,
        document_title="SipraHub_Timesheet_Policy.docx",
    )
    scored = _fallback_heuristic_rerank(query, [chunk])
    assert len(scored) == 1
    score, res = scored[0]
    # Score should be high because working, hours, sipra, hub, siprahub all match
    assert score > 0.40

def test_earth_query_normalization_regex():
    orig, norm, ret_q = normalize_query("Is Earth 2nd or 3rd planet?")
    assert "Earth" in norm or "planet" in norm
