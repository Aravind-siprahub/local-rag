import sys
import os

sys.path.insert(0, os.path.abspath("backend"))

from app.llm.sanitize import sanitize_response

q = "what fronted and backend are using talk to my data"
echo_text = '"what fronted and backend are using talk to my data"'

res = sanitize_response(echo_text, question=q)
print(f"Sanitized echo output: {res!r}")
assert res == "", f"Expected empty string for question echo, got: {res!r}"

valid_text = "Frontend: React with Vite. Backend: FastAPI."
res_valid = sanitize_response(valid_text, question=q)
print(f"Sanitized valid output: {res_valid!r}")
assert res_valid == valid_text, f"Expected valid text, got: {res_valid!r}"

print("\n✅ ECHO PREVENTION VERIFICATION PASSED PERFECTLY!")
