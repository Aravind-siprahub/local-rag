import sys
import os

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath("backend"))

from app.rag.intent_router import Route, classify

def test_intent_matrix():
    test_cases = [
        ("earth is 2 planet or 3 planet", Route.GENERAL_KNOWLEDGE),
        ("earth which planet", Route.GENERAL_KNOWLEDGE),
        ("earth 2nd or 3rd", Route.GENERAL_KNOWLEDGE),
        ("what is 2 + 2", [Route.CALCULATOR, Route.GENERAL_KNOWLEDGE]),
        ("hello", Route.GENERIC_CHAT),
        ("good morning", Route.GENERIC_CHAT),
        ("what is Python", Route.GENERAL_KNOWLEDGE),
        ("who invented the telephone", Route.GENERAL_KNOWLEDGE),
        ("what is the capital of France", Route.GENERAL_KNOWLEDGE),
        ("what is the leave policy in my document?", Route.DOCUMENT_QA),
        ("leave policy what say", Route.DOCUMENT_QA),
        ("what does my document say about leave?", Route.DOCUMENT_QA),
        ("what documents do I have?", Route.DOCUMENT_LIST),
        ("what doc u have", Route.DOCUMENT_LIST),
        ("when was PRD_Talk_to_My_Data.docx uploaded?", Route.DOCUMENT_METADATA),
        ("when this file upload", Route.DOCUMENT_METADATA),
    ]

    passed = 0
    failed = 0

    print("=== INTENT ROUTER TEST MATRIX ===")
    for query, expected in test_cases:
        actual = classify(query)
        if isinstance(expected, list):
            ok = actual in expected
        else:
            ok = actual == expected

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"[{status}] Question: '{query}' => Expected: {expected}, Got: {actual}")

    print(f"\nTotal: {len(test_cases)}, Passed: {passed}, Failed: {failed}")
    return failed == 0

if __name__ == "__main__":
    success = test_intent_matrix()
    if not success:
        sys.exit(1)
