"""Regression tests for general knowledge and web search intent routing."""

from app.rag.intent_router import Route, classify

class TestRoutingRegression:
    def test_code_generation_routes_to_general_knowledge(self) -> None:
        """Ensure code generation prompts don't hit the RAG fallback."""
        route = classify("write prompt to create login and signup page")
        assert route == Route.GENERAL_KNOWLEDGE

    def test_general_knowledge_routes_correctly(self) -> None:
        """Ensure general questions don't hit the RAG fallback."""
        route = classify("what is Python?")
        assert route == Route.GENERAL_KNOWLEDGE

    def test_latest_information_routes_to_web(self) -> None:
        """Ensure 'latest' keywords trigger web search."""
        route = classify("what is the latest Python version?")
        assert route == Route.WEB
        
        route2 = classify("who won the match yesterday?")
        assert route2 == Route.WEB

    def test_document_qa_routes_correctly(self) -> None:
        """Ensure explicit document questions still route to RAG."""
        route = classify("what does my uploaded document say about login?")
        assert route == Route.DOCUMENT_QA
        
        route2 = classify("summarize my uploaded document")
        assert route2 == Route.DOCUMENT_QA
