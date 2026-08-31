import uuid
import pytest
from app.rag.intent_router import Route, classify
from app.prompting.builder import PromptBuilder
from app.retrieval.ranking import RankedResult


class TestRAGPipelineE2E:
    def test_routing_siprahub_working_hours(self) -> None:
        titles = ["SipraHub_Policy.pdf"]
        route = classify("tell about working hours in Sipra hub", document_titles=titles)
        assert route == Route.DOCUMENT_QA

    def test_routing_siprahub_generic_question(self) -> None:
        titles = ["SipraHub_PRD_v11.docx"]
        route = classify("what is SipraHub?", document_titles=titles)
        assert route == Route.DOCUMENT_QA

    def test_routing_without_documents(self) -> None:
        route = classify("what is SipraHub?", document_titles=[])
        assert route == Route.GENERAL_KNOWLEDGE

    def test_prompt_builder_with_retrieved_context(self) -> None:
        builder = PromptBuilder()
        chunks = [
            RankedResult(
                chunk_id=uuid.uuid4(),
                chunk_text="Working hours in Sipra Hub are Monday to Friday, 9:00 AM to 6:00 PM.",
                document_id=uuid.uuid4(),
                document_version_id=uuid.uuid4(),
                similarity_score=0.92,
                rank=1,
                document_title="SipraHub_Policy.pdf",
                section_title="Working Hours",
                page_number=1,
            )
        ]
        prompt_res = builder.build(
            question="tell about working hours in Sipra hub",
            retrieved_chunks=chunks,
        )
        assert "SipraHub_Policy.pdf" in prompt_res.user_prompt
        assert "9:00 AM to 6:00 PM" in prompt_res.user_prompt
        assert len(prompt_res.retrieved_chunks) == 1

    def test_routing_attendance_policy(self) -> None:
        titles = ["SipraHub_Policy.pdf"]
        route = classify("What is the attendance policy?", document_titles=titles)
        assert route in (Route.DOCUMENT_QA, Route.GENERAL_KNOWLEDGE)

    def test_routing_working_hours_and_attendance(self) -> None:
        titles = ["SipraHub_Policy.pdf"]
        route = classify("Siprahub Working Hours & Attendance Policy", document_titles=titles)
        assert route == Route.DOCUMENT_QA

    def test_non_existent_information_grounding(self) -> None:
        builder = PromptBuilder()
        prompt_res = builder.build(
            question="What is the refund policy for space station tickets?",
            retrieved_chunks=[],
        )
        assert "Information not found in document excerpts." in prompt_res.user_prompt
        assert len(prompt_res.retrieved_chunks) == 0

