import sys
import os

sys.path.insert(0, r"c:\Users\ARAVIND\Desktop\local-rag\backend")

from app.rag.intent_router import classify, Route
from app.prompting.builder import PromptBuilder

def run_datetime_verification():
    print("=== VERIFYING DATETIME ROUTING & PROMPT INJECTION ===")

    test_queries = [
        "what is today date",
        "what is the date today",
        "what time is it",
        "current date",
        "what day is today",
    ]

    for q in test_queries:
        r = classify(q)
        print(f"Query: '{q}' -> Route: {r}")
        assert r == Route.WEB, f"Failed routing for '{q}': got {r}"

    # Verify prompt builder injects temporal context
    builder = PromptBuilder()
    prompt = builder.build("what is today date", [])
    print("\nGenerated System Prompt:\n", prompt.system_prompt)
    assert "Today's Date:" in prompt.system_prompt
    assert "Current Temporal Context:" in prompt.system_prompt

    print("\nALL DATETIME ROUTING & PROMPT INJECTION TESTS PASSED! 🎉")

if __name__ == "__main__":
    run_datetime_verification()
