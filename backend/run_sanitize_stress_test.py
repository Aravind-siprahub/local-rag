"""
Stress-test script for sanitize_response().
Run from backend/ with: python run_sanitize_stress_test.py
No live model call needed.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.llm.sanitize import sanitize_response, _strip_reasoning_paragraphs

CLEAN_ANSWER = "Talk to My Data is a platform."

test_cases = [
    # (description, input, expected_contains, expected_NOT_contains, must_preserve_full)
    (
        "TC1: Let me analyze prefix",
        f"Let me analyze the documents. {CLEAN_ANSWER}",
        [CLEAN_ANSWER],
        ["Let me analyze"],
        False,
    ),
    (
        "TC2: Looking at Chunk reference",
        f"Looking at Chunk 1, I can see that {CLEAN_ANSWER}",
        [CLEAN_ANSWER],
        ["Looking at Chunk"],
        False,
    ),
    (
        "TC3: Based on context prefix",
        f"Based on the context provided, {CLEAN_ANSWER}",
        [CLEAN_ANSWER],
        ["Based on the context"],
        False,
    ),
    (
        "TC4: From the chunks above prefix",
        f"From the chunks above, {CLEAN_ANSWER}",
        [CLEAN_ANSWER],
        ["From the chunks"],
        False,
    ),
    (
        "TC5: To answer this question prefix",
        f"To answer this question, I need to review the sources. {CLEAN_ANSWER}",
        [CLEAN_ANSWER],
        ["To answer this question"],
        False,
    ),
    (
        "TC6: First note (MUST NOT strip - legitimate answer)",
        f"First, note that {CLEAN_ANSWER} with three key features.",
        [f"First, note that {CLEAN_ANSWER}"],
        [],
        True,  # full content must be preserved
    ),
    (
        "TC7: Original investigation example with Chunk labels + think monologue",
        "Let me analyze the documents provided. I need to check Chunk 1. Looking at Chunk 1, the user asks about Talk to My Data... I'll formulate the response.\n\nTalk to My Data is an enterprise AI platform.",
        ["Talk to My Data is an enterprise AI platform."],
        ["Let me analyze", "Looking at Chunk", "I'll formulate"],
        False,
    ),
    (
        "TC8: I need to prefix (single-paragraph)",
        f"I need to check the relevant sections.\n\n{CLEAN_ANSWER}",
        [CLEAN_ANSWER],
        ["I need to"],
        False,
    ),
    (
        "TC9: I should prefix",
        f"I should note the following before answering.\n\n{CLEAN_ANSWER}",
        [CLEAN_ANSWER],
        ["I should"],
        False,
    ),
    (
        "TC10: Clean direct answer (must be fully preserved)",
        "Talk to My Data enables natural language queries over document repositories.",
        ["Talk to My Data enables natural language queries over document repositories."],
        [],
        True,
    ),
    (
        "TC11: First, I'll prefix (should strip)",
        f"First, I'll review the chunks provided.\n\n{CLEAN_ANSWER}",
        [CLEAN_ANSWER],
        ["First, I'll"],
        False,
    ),
    (
        "TC12: Hmm prefix",
        f"Hmm, this is an interesting question.\n\n{CLEAN_ANSWER}",
        [CLEAN_ANSWER],
        ["Hmm,"],
        False,
    ),
]

passed = 0
failed = 0
results = []

for desc, inp, must_contain, must_not_contain, must_preserve_full in test_cases:
    out = sanitize_response(inp)
    ok = True
    issues = []

    for phrase in must_contain:
        if phrase not in out:
            ok = False
            issues.append(f"FALSE NEGATIVE: '{phrase}' missing from output")

    for phrase in must_not_contain:
        if phrase in out:
            ok = False
            issues.append(f"FALSE POSITIVE: '{phrase}' leaked into output")

    if must_preserve_full and inp.strip() != out.strip():
        # Check if content was wrongly altered
        if out != inp.strip():
            # Acceptable only if output fully contains the meaningful content
            for phrase in must_contain:
                if phrase not in out:
                    ok = False
                    issues.append(f"FALSE POSITIVE (preservation): full content was altered unexpectedly")
                    break

    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1

    results.append((status, desc, inp[:80].replace('\n', ' '), out[:80].replace('\n', ' '), issues))

print()
print("=" * 100)
print("  SANITIZE_RESPONSE STRESS TEST RESULTS")
print("=" * 100)
print(f"{'STATUS':<6}  {'DESCRIPTION':<45}  {'INPUT[:80]':<40}")
print("-" * 100)
for status, desc, inp_preview, out_preview, issues in results:
    print(f"  {status:<6}  {desc:<45}  {inp_preview}")
    print(f"          {'':45}  → {out_preview}")
    for issue in issues:
        print(f"          *** {issue}")
    print()

print("=" * 100)
print(f"  TOTAL: {passed} PASSED, {failed} FAILED out of {len(test_cases)} test cases")
print("=" * 100)
