"""Integration tests for RAG response quality and conflict handling.

These tests use the real Ollama LLM to verify that the system prompt
effectively suppresses internal reasoning and correctly handles conflicts.
"""
import uuid
import pytest

from app.core.config import get_settings
from app.llm.ollama_client import OllamaLLMClient
from app.prompting.builder import PromptBuilder
from app.retrieval.ranking import RankedResult
from app.llm.sanitize import sanitize_response


def _verify_no_reasoning(answer: str):
    """Verify exact banned phrases are not present."""
    banned = [
        "let me check", 
        "i think", 
        "actually, let me", 
        "the most likely explanation", 
        "let's reason", 
        "i need to determine",
        "let's check",
        "my reasoning",
        "i will now"
    ]
    lower_ans = answer.lower()
    for phrase in banned:
        assert phrase not in lower_ans, f"Found banned reasoning phrase: {phrase}"
        
    # Also loosely check for document-by-document narration
    assert "document 1 says" not in lower_ans
    assert "document 2 says" not in lower_ans


@pytest.fixture
def llm_client():
    return OllamaLLMClient(model=get_settings().ollama_chat_model)


@pytest.fixture
def prompt_builder():
    return PromptBuilder()


@pytest.mark.asyncio
async def test_simple_factual_rag_question(llm_client, prompt_builder):
    chunks = [
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text="The SipraOne frontend uses port 4173.",
            document_id=uuid.uuid4(),
            document_version_id=uuid.uuid4(),
            similarity_score=0.9,
            rank=1,
            document_title="Deployment Guide",
            section_title="Frontend"
        )
    ]
    prompt = prompt_builder.build("What port does the Sipraone frontend use?", chunks)
    response = await llm_client.generate(prompt.system_prompt, prompt.user_prompt)
    answer = sanitize_response(response.answer)
    
    _verify_no_reasoning(answer)
    assert "4173" in answer
    assert len(answer.split(".")) <= 6  # concise


@pytest.mark.asyncio
async def test_compound_rag_question(llm_client, prompt_builder):
    chunks = [
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text="The SipraOne frontend uses port 4173. The backend uses port 8000.",
            document_id=uuid.uuid4(),
            document_version_id=uuid.uuid4(),
            similarity_score=0.9,
            rank=1,
            document_title="Deployment Guide",
            section_title="Architecture"
        )
    ]
    prompt = prompt_builder.build("What ports do the frontend and backend use?", chunks)
    response = await llm_client.generate(prompt.system_prompt, prompt.user_prompt)
    answer = sanitize_response(response.answer)
    
    _verify_no_reasoning(answer)
    assert "4173" in answer
    assert "8000" in answer


@pytest.mark.asyncio
async def test_conflicting_documents(llm_client, prompt_builder):
    chunks = [
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text="The SipraOne frontend uses port 4173.",
            document_id=uuid.uuid4(),
            document_version_id=uuid.uuid4(),
            similarity_score=0.9,
            rank=1,
            document_title="Frontend Deployment Guide",
            section_title="Frontend"
        ),
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text="The SipraOne frontend uses port 8001.",
            document_id=uuid.uuid4(),
            document_version_id=uuid.uuid4(),
            similarity_score=0.88,
            rank=2,
            document_title="Old Architecture Overview",
            section_title="Frontend Setup"
        )
    ]
    prompt = prompt_builder.build("What port does the Sipraone frontend use?", chunks)
    response = await llm_client.generate(prompt.system_prompt, prompt.user_prompt)
    answer = sanitize_response(response.answer).lower()
    
    _verify_no_reasoning(answer)
    assert "4173" in answer, f"Expected 4173 in answer: {answer}"
    assert "8001" in answer, f"Expected 8001 in answer: {answer}"
    # Must semantically mention the conflict or discrepancy
    conflict_indicators = [
        "conflict", "inconsistent", "discrepancy", "another document", 
        "however", "although", "while", "but", "contradict", "differs",
        "different", "varies", "or", "and"
    ]
    assert any(indicator in answer for indicator in conflict_indicators), f"Expected conflict phrasing, got: {answer}"


@pytest.mark.asyncio
async def test_missing_information(llm_client, prompt_builder):
    chunks = [
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text="The SipraOne backend uses Python and FastAPI.",
            document_id=uuid.uuid4(),
            document_version_id=uuid.uuid4(),
            similarity_score=0.9,
            rank=1,
            document_title="Deployment Guide",
            section_title="Backend"
        )
    ]
    prompt = prompt_builder.build("What port does the Sipraone frontend use?", chunks)
    response = await llm_client.generate(prompt.system_prompt, prompt.user_prompt)
    answer = sanitize_response(response.answer).lower()
    
    _verify_no_reasoning(answer)
    # The prompt instructs it to explicitly state the information is missing
    missing_indicators = [
        "not found", "cannot be determined", "could not find", 
        "does not specify", "do not specify", "not specified", 
        "no information", "unavailable", "does not state", 
        "not mentioned", "doesn't say", "do not contain", "does not contain"
    ]
    assert any(indicator in answer for indicator in missing_indicators), f"Expected missing info phrasing, got: {answer}"
    assert "4173" not in answer  # Should not invent the fact


@pytest.mark.asyncio
async def test_image_and_rag(llm_client, prompt_builder):
    chunks = [
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text="The dashboard shows error code 500 when the DB is down.",
            document_id=uuid.uuid4(),
            document_version_id=uuid.uuid4(),
            similarity_score=0.9,
            rank=1,
            document_title="Troubleshooting",
            section_title="Errors"
        )
    ]
    prompt = prompt_builder.build("What does this error mean?", chunks, is_vision=True)
    
    # We verify that VISION_RAG_SYSTEM_PROMPT was selected
    assert "visual assistant" in prompt.system_prompt.lower()
    
    response = await llm_client.generate(prompt.system_prompt, prompt.user_prompt)
    answer = sanitize_response(response.answer)
    
    _verify_no_reasoning(answer)


@pytest.mark.asyncio
async def test_no_retrieved_context(llm_client, prompt_builder):
    prompt = prompt_builder.build("What port does the Sipraone frontend use?", [])
    response = await llm_client.generate(prompt.system_prompt, prompt.user_prompt)
    answer = sanitize_response(response.answer).lower()
    
    _verify_no_reasoning(answer)
    # The generic prompt without context will likely answer that no context was provided or hallucinate if allowed,
    # but the system prompt should restrict it if it's the RAG system prompt.
    assert "not" in answer or "cannot" in answer or "no document" in answer or "no information" in answer or "no context" in answer or "do not" in answer or "unavailable" in answer or "missing" in answer


@pytest.mark.asyncio
async def test_generic_chat_no_regression(llm_client):
    system_prompt = "You are a helpful assistant. Respond with the answer only — no reasoning, no commentary, no self-talk."
    user_prompt = "Hello, how are you?"
    response = await llm_client.generate(system_prompt, user_prompt)
    answer = sanitize_response(response.answer).lower()
    
    _verify_no_reasoning(answer)
    assert len(answer) > 0
    assert len(answer.split()) < 40, f"Expected short chat response (<40 words), got: {answer}"


@pytest.mark.asyncio
async def test_no_meta_commentary_on_broken_grammar(llm_client, prompt_builder):
    chunks = [
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text="Frontend talks only to FastAPI. Frontend is built with React and Vite. Deployment uses VITE_BACKEND_URL.",
            document_id=uuid.uuid4(),
            document_version_id=uuid.uuid4(),
            similarity_score=0.9,
            rank=1,
            document_title="PRD_Talk_to_My_Data.docx",
            section_title="Architecture"
        )
    ]
    
    # Simulating the exact broken grammar prompt that caused the issue
    prompt = prompt_builder.build("what backend and frontend use in talk to my data", chunks)
    response = await llm_client.generate(prompt.system_prompt, prompt.user_prompt)
    answer = sanitize_response(response.answer)
    lower_ans = answer.lower()
    
    # 1. Verify exact factual data is present
    assert "react" in lower_ans
    assert "fastapi" in lower_ans
    assert "vite" in lower_ans
    
    # 2. Verify all reasoning and citation leakage is absent
    banned_meta_phrases = [
        "according to",
        "the document says",
        "the documents state",
        "this confirms",
        "this tells us",
        "this indicates",
        "putting it together",
        "therefore",
        "document",
        "section",
        "from \"",
        "page",
        "prd_talk_to_my_data",
        "let me think",
    ]
    for phrase in banned_meta_phrases:
        assert phrase not in lower_ans, f"Found banned meta-commentary: '{phrase}'"
    
    # 3. Ensure the answer is reasonably direct (should be just the facts, not a huge paragraph)
    assert len(answer.split()) < 30, f"Answer is too verbose: {answer}"
