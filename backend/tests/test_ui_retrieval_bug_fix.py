import pytest
from app.rag.intent_router import classify, Route

def test_intent_router_corpus_active_routing():
    """Verify that when document_titles are present in the user corpus, questions route to DOCUMENT_QA."""
    doc_titles = ["HR Framework & SipraHub Operations.pdf", "Architecture Roadmap.pdf"]
    
    # 1. Question without explicit "in document" phrase
    route1 = classify("What is the purpose of the HR & Compliance Framework?", document_titles=doc_titles)
    assert route1 in (Route.DOCUMENT_QA, Route.RAG, Route.HYBRID), f"Expected DOCUMENT_QA/RAG/HYBRID, got {route1}"
    
    # 2. Core values question
    route2 = classify("What are SipraHub's core values?", document_titles=doc_titles)
    assert route2 in (Route.DOCUMENT_QA, Route.RAG, Route.HYBRID), f"Expected DOCUMENT_QA/RAG/HYBRID, got {route2}"

    # 3. Casual leave question
    route3 = classify("What is the Casual Leave entitlement?", document_titles=doc_titles)
    assert route3 in (Route.DOCUMENT_QA, Route.RAG, Route.HYBRID), f"Expected DOCUMENT_QA/RAG/HYBRID, got {route3}"

    # 4. POSH policy complaint question
    route4 = classify("What does the POSH policy say about complaint filing?", document_titles=doc_titles)
    assert route4 in (Route.DOCUMENT_QA, Route.RAG, Route.HYBRID), f"Expected DOCUMENT_QA/RAG/HYBRID, got {route4}"

def test_intent_router_preserves_generic_chat_greetings():
    """Verify conversational greetings still route to GENERIC_CHAT."""
    doc_titles = ["HR Framework & SipraHub Operations.pdf"]
    
    route = classify("hello how are you", document_titles=doc_titles)
    assert route == Route.GENERIC_CHAT
