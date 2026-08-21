import unittest
import sys
import os

# Add backend directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.rag.verifier import verify_answer, VerificationResult
from app.rag.query_understanding import QueryIntent, AttributeCategory, extract_query_intent
from app.llm.sanitize import sanitize_response
from app.retrieval.ranking import RankedResult
import uuid

class TestRAGFixes(unittest.TestCase):
    def test_intent_extraction(self):
        query = "What frontend and backend frameworks are used by talk to my data"
        intent = extract_query_intent(query)
        self.assertEqual(intent.entity, "Talk to My Data")
        self.assertEqual(intent.category, AttributeCategory.TECHNOLOGY)
        self.assertIn("frontend", intent.attributes)
        self.assertIn("backend", intent.attributes)

    def test_verify_answer_tech_stack(self):
        query = "What frontend and backend frameworks are used by talk to my data"
        intent = extract_query_intent(query)
        chunks = [
            RankedResult(
                chunk_id=uuid.uuid4(),
                chunk_text="Talk to My Data (SipraHub) is built with React for the frontend chat interface and FastAPI for the python backend service.",
                document_id=uuid.uuid4(),
                similarity_score=0.9,
                rank=1,
                document_title="architecture.pdf"
            )
        ]
        ans = "Frontend: React, Backend: FastAPI"
        res = verify_answer(ans, chunks, intent)
        self.assertTrue(res.is_valid, f"Verification failed: {res.reason}")

    def test_verify_answer_with_siprahub_mention(self):
        query = "What frontend and backend frameworks are used by talk to my data"
        intent = extract_query_intent(query)
        chunks = [
            RankedResult(
                chunk_id=uuid.uuid4(),
                chunk_text="Talk to My Data application in SipraHub uses React and FastAPI.",
                document_id=uuid.uuid4(),
                similarity_score=0.9,
                rank=1,
                document_title="architecture.pdf"
            )
        ]
        ans = "SipraHub's Talk to My Data uses React on the frontend and FastAPI on the backend."
        res = verify_answer(ans, chunks, intent)
        self.assertTrue(res.is_valid, f"Verification failed: {res.reason}")

    def test_sanitize_response_preserves_framework_answer(self):
        raw_llm = "Frontend: React\nBackend: FastAPI"
        clean = sanitize_response(raw_llm, question="What frontend and backend frameworks are used by talk to my data")
        self.assertIn("React", clean)
        self.assertIn("FastAPI", clean)

if __name__ == "__main__":
    unittest.main()
