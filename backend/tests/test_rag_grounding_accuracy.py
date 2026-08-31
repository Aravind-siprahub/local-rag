import pytest
from app.prompting.templates import USER_PROMPT_WITH_CONTEXT
from app.llm.sanitize import sanitize_response

def test_grounding_rules_exist_in_templates():
    assert "Do NOT invent or infer unstated facts" in USER_PROMPT_WITH_CONTEXT
    assert "clarifying that exact fixed times or figures are not explicitly specified" in USER_PROMPT_WITH_CONTEXT

def test_hallucination_resistance_on_shift_times():
    question = "What time does the Sipra Hub shift start?"
    # Model adheres to grounding rules and clarifies that exact shift times are not specified
    grounded_model_response = (
        "The HR framework document describes daily timesheet tracking across a 5-day work week for Sipra Hub, "
        "but it does not explicitly specify fixed shift start or end times."
    )
    cleaned = sanitize_response(grounded_model_response, question=question)
    assert "does not explicitly specify fixed shift start" in cleaned
    assert "9:30" not in cleaned
    assert "9:00" not in cleaned

def test_hallucination_resistance_on_lunch_break():
    question = "What is the lunch break?"
    grounded_model_response = (
        "The document does not state lunch break policies for Sipra Hub. "
        "It only specifies daily timesheet tracking over 5 working days."
    )
    cleaned = sanitize_response(grounded_model_response, question=question)
    assert "does not state lunch break" in cleaned
