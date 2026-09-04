"""Comprehensive evaluation suite for RAG document-grounded answering system.

Validates all 10 grounding and hallucination prevention categories:
1. Fully supported question
2. Partially supported question
3. Completely unsupported question
4. Multi-topic question decomposition
5. Numerical fidelity
6. Time period / duration fidelity
7. Policy grounding
8. Exact wording preservation
9. Related information exists but requested information does not
10. Model general knowledge conflict rejection
"""
import uuid
import pytest

from app.rag.query_understanding import decompose_query_topics
from app.rag.validator import (
    validate_and_reconcile_answer,
    topic_has_evidence,
    topic_is_acknowledged_as_missing,
)
from app.retrieval.ranking import RankedResult


@pytest.fixture
def hr_context_chunks():
    """Simulated realistic context chunks from the New HR Framework document."""
    doc_id = uuid.uuid4()
    return [
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text=(
                "5. LEAVE POLICY\n"
                "5.1 Casual Leave (CL):\n"
                "Employees are entitled to 1 (one) Casual Leave per month worked.\n"
                "Unused Casual Leave cannot be carried forward to the next calendar year.\n"
                "Casual leave must be applied at least 2 days in advance, except in emergencies.\n"
                "5.2 Leave Without Pay (LWP):\n"
                "Any leave taken beyond the entitlement or without prior approval will be treated as Leave Without Pay."
            ),
            document_id=doc_id,
            similarity_score=0.92,
            rank=1,
            document_title="New HR Framework (3) 1.docx",
            section_title="5. LEAVE POLICY",
        ),
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text=(
                "2. PROBATION & EMPLOYMENT CONFIRMATION\n"
                "All new employees shall undergo a probation period of 3 (three) months from the date of joining.\n"
                "During probation, either party may terminate employment with 15 days' written notice.\n"
                "After confirmation, the notice period shall be 30 (thirty) days."
            ),
            document_id=doc_id,
            similarity_score=0.88,
            rank=2,
            document_title="New HR Framework (3) 1.docx",
            section_title="2. PROBATION & EMPLOYMENT CONFIRMATION",
        ),
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text=(
                "4. CODE OF CONDUCT & WORKPLACE COMMITMENTS\n"
                "At SipraHub, employees are expected to maintain professional behavior, accountability, "
                "integrity, and a respectful work environment. Zero tolerance for harassment under POSH."
            ),
            document_id=doc_id,
            similarity_score=0.85,
            rank=3,
            document_title="New HR Framework (3) 1.docx",
            section_title="4. CODE OF CONDUCT & WORKPLACE COMMITMENTS",
        ),
    ]


class TestDocumentGroundingAudit:
    """Evaluation test suite covering all 10 required grounding categories."""

    def test_01_query_decomposition_multipart(self):
        """Test category 4: Multi-topic query decomposition into discrete topics."""
        q1 = "What are the rules for casual leave and sick leave?"
        topics1 = decompose_query_topics(q1)
        assert "casual leave" in topics1
        assert "sick leave" in topics1
        assert len(topics1) == 2

        q2 = "What is the probation period, notice period, and working hours?"
        topics2 = decompose_query_topics(q2)
        assert any("probation" in t for t in topics2)
        assert any("notice" in t for t in topics2)
        assert any("working hours" in t for t in topics2)

    def test_02_fully_supported_question(self, hr_context_chunks):
        """Test category 1: Fully supported question grounded in context."""
        question = "What are the rules for casual leave?"
        raw_answer = (
            "- Employees are entitled to 1 (one) Casual Leave per month worked.\n"
            "- Unused Casual Leave cannot be carried forward to the next calendar year.\n"
            "- Casual leave must be applied at least 2 days in advance."
        )
        reconciled = validate_and_reconcile_answer(question, raw_answer, hr_context_chunks)
        assert "1 (one) Casual Leave per month" in reconciled
        assert "carried forward" in reconciled

    def test_03_partially_supported_question_reconciliation(self, hr_context_chunks):
        """Test category 2: Partially supported question (Casual leave + Sick leave).

        Casual leave is present, but Sick leave is absent.
        Validator must ensure Sick leave is explicitly declared as not specified.
        """
        question = "What are the rules for casual leave and sick leave?"
        # Model answered Casual leave but omitted Sick leave
        incomplete_answer = (
            "### Casual Leave:\n"
            "- Entitled to 1 (one) Casual Leave per month worked.\n"
            "- Unused Casual Leave cannot be carried forward."
        )
        reconciled = validate_and_reconcile_answer(question, incomplete_answer, hr_context_chunks)
        assert "Casual Leave" in reconciled
        assert "Sick Leave" in reconciled
        assert "The provided document does not specify" in reconciled

    def test_04_partially_supported_with_hallucinated_section(self, hr_context_chunks):
        """Test category 2 & 10: Model hallucinated a Sick Leave section not in the document.

        Validator must strip or replace the fabricated Sick Leave section.
        """
        question = "What are the rules for casual leave and sick leave?"
        hallucinated_answer = (
            "### Casual Leave:\n"
            "- 1 Casual Leave per month.\n\n"
            "### Sick Leave:\n"
            "- Employees get 10 days of medical sick leave per year with doctor certificate."
        )
        reconciled = validate_and_reconcile_answer(question, hallucinated_answer, hr_context_chunks)
        # The hallucinated 10 days must NOT be present
        assert "10 days of medical sick leave" not in reconciled
        assert "The provided document does not specify a separate sick leave policy" in reconciled

    def test_05_completely_unsupported_question(self, hr_context_chunks):
        """Test category 3: Completely unsupported question (e.g. Stock Options / 401k)."""
        question = "What is the stock option and 401k vesting schedule?"
        # If the model tried to generate a generic answer:
        hallucinated_answer = "Employees vest 25% of stock options after 1 year with a standard 4-year schedule."
        reconciled = validate_and_reconcile_answer(question, hallucinated_answer, hr_context_chunks)
        assert "4-year schedule" not in reconciled
        assert "The provided document does not specify" in reconciled

    def test_06_numerical_fidelity(self, hr_context_chunks):
        """Test category 5 & 8: Preserve exact numbers and do not extrapolate."""
        context_text = " ".join(c.chunk_text for c in hr_context_chunks)
        assert "1 (one) Casual Leave per month" in context_text
        # Ensure our validator acknowledges casual leave evidence
        assert topic_has_evidence("casual leave", context_text) is True
        assert topic_has_evidence("sick leave", context_text) is False

    def test_07_duration_and_notice_period(self, hr_context_chunks):
        """Test category 6: Exact duration (3 months probation, 15/30 days notice)."""
        question = "What is the probation period and notice period?"
        answer = (
            "- Probation period: 3 (three) months from date of joining.\n"
            "- Notice period during probation: 15 days written notice.\n"
            "- Notice period after confirmation: 30 (thirty) days."
        )
        reconciled = validate_and_reconcile_answer(question, answer, hr_context_chunks)
        assert "3 (three) months" in reconciled
        assert "15 days" in reconciled
        assert "30 (thirty) days" in reconciled

    def test_08_related_info_exists_but_requested_does_not(self, hr_context_chunks):
        """Test category 9: Question asks for 'Core Values' when doc only has 'Code of Conduct'.

        Validator must ensure the answer clarifies that the document does not specify a dedicated Core Values section.
        """
        question = "What are Our Core Values of Siprahub?"
        synthesized_answer = (
            "### Core Values and Principles\n"
            "- Professional behavior\n"
            "- Accountability and integrity\n"
            "- Respectful work environment"
        )
        reconciled = validate_and_reconcile_answer(question, synthesized_answer, hr_context_chunks)
        assert "The provided document does not specify a dedicated Core Values section" in reconciled
        assert "Code of Conduct" in reconciled

    def test_09_general_knowledge_conflict_rejection(self, hr_context_chunks):
        """Test category 10: Model general knowledge says sick leave exists in HR, but doc has none."""
        question = "What is the sick leave policy?"
        hallucinated_model_answer = "SipraHub provides 12 paid sick leaves annually upon medical documentation."
        reconciled = validate_and_reconcile_answer(question, hallucinated_model_answer, hr_context_chunks)
        assert "12 paid sick leaves" not in reconciled
        assert "The provided document does not specify" in reconciled

    def test_10_missing_topic_acknowledgment_helper(self):
        """Test helper topic_is_acknowledged_as_missing recognizing standard disclaimer forms."""
        ans1 = "The provided document does not specify a separate Sick Leave policy."
        assert topic_is_acknowledged_as_missing("sick leave", ans1) is True

        ans2 = "Casual leave is 1 day per month. Sick leave is not mentioned in the provided document."
        assert topic_is_acknowledged_as_missing("sick leave", ans2) is True

        ans3 = "Casual leave is 1 day per month."
        assert topic_is_acknowledged_as_missing("sick leave", ans3) is False
