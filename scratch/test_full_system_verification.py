import sys
import os

sys.path.insert(0, os.path.abspath("backend"))

from app.llm.sanitize import sanitize_response

q = "what fronted and backend are using talk to my data"

# Test Case 1: Pure Factual Answer
tc1_input = "Backend: FastAPI Frontend: React (with Vite)"
tc1_out = sanitize_response(tc1_input, question=q)
print(f"Test 1 (Factual Answer): {tc1_out!r}")
assert tc1_out == "Backend: FastAPI Frontend: React (with Vite)"

# Test Case 2: Multi-document Section Breakdown + Final Answer
tc2_input = """1. In "PRD_Talk_to_My_Data.docx" section 21: "React talks only to FastAPI..."
2. In "Deployment_Guide.docx" section 14.2...
Therefore, the answer to "what fronted and backend are using talk to my data" is:
Backend: FastAPI Frontend: React (with Vite)"""
tc2_out = sanitize_response(tc2_input, question=q)
print(f"Test 2 (Multi-Doc Breakdown + Concluding Marker): {tc2_out!r}")
assert tc2_out == "Backend: FastAPI Frontend: React (with Vite)"

# Test Case 3: Exact Question Echo
tc3_input = '"what fronted and backend are using talk to my data"'
tc3_out = sanitize_response(tc3_input, question=q)
print(f"Test 3 (Question Echo): {tc3_out!r}")
assert tc3_out == "", "Question echo must return empty string"

# Test Case 4: Answer: <Echo Question>
tc4_input = 'Answer: "what fronted and backend are using talk to my data"'
tc4_out = sanitize_response(tc4_input, question=q)
print(f"Test 4 (Answer Prefix + Question Echo): {tc4_out!r}")
assert tc4_out == "", "Answer + question echo must return empty string"

# Test Case 5: Final Answer Prefix
tc5_input = "Final answer: Frontend: React with Vite. Backend: FastAPI."
tc5_out = sanitize_response(tc5_input, question=q)
print(f"Test 5 (Final Answer Prefix): {tc5_out!r}")
assert tc5_out == "Frontend: React with Vite. Backend: FastAPI."

# Test Case 6: XML Thinking Tags (<think>...</think>)
tc6_input = "<think>Analyzing documents...\nReact and FastAPI found.</think>\nFrontend: React with Vite. Backend: FastAPI."
tc6_out = sanitize_response(tc6_input, question=q)
print(f"Test 6 (XML Thinking Tags): {tc6_out!r}")
assert tc6_out == "Frontend: React with Vite. Backend: FastAPI."

print("\n" + "="*60)
print("🎉 ALL 6 RIGOROUS SYSTEM TESTS PASSED PERFECTLY ON 2ND DOUBLE-CHECK!")
print("="*60)
