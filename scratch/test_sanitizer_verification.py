import sys
import os

sys.path.insert(0, os.path.abspath("backend"))

from app.llm.sanitize import sanitize_response

# Sample 1: The exact output from the user's first screenshot
sample_1 = """1. In "PRD_Talk_to_My_Data.docx" section 21: "React talks only to FastAPI — no component below the backend is ever exposed directly to the browser." -> This indicates the frontend is React and the backend is FastAPI.
2. In "PRD_Talk_to_My_Data.docx" section 5: "Frontend — the chat interface; also renders citations, source snippets, and (for data questions) the generated SQL and any resulting chart." -> This confirms the frontend is a chat interface (so React) and the backend is FastAPI (as per the first document).
3. In "Deployment_Guide.docx" section 14.2: "VITE_BACKEND_URL=https://<domain>:<backend-port>" and "VITE_REDIRECT_URI=https://<domain>:<frontend-port>" -> This shows the frontend uses Vite (a build tool for frontend) and the backend port is not specified but we know from the context that the backend is FastAPI.
4. In "Deployment_Guide.docx" section 8.2: It mentions frontend variables prefixed with NEXT_PUBLIC_ but note: the document says "the equivalent convention for your framework". However, the context of the project (from the other documents) indicates that the frontend is React (which typically uses Vite, not Next.js). But the document says "VITE_BACKEND_URL", so the frontend is built with Vite.
5. The VM_Setup_Guide doesn't give direct information about the framework but confirms the presence of frontend and backend folders.

The key points:
- The frontend is built with React (as per the PRD document) and uses Vite (as per the deployment guide's VITE_BACKEND_URL).
- The backend is FastAPI (as per the PRD document).

Therefore, the answer to "what backend and frontend use in talk to my data" is:

Backend: FastAPI Frontend: React (with Vite)"""

# Sample 2: The output from the user's second screenshot
sample_2 = """"Frontend variables prefixed NEXT_PUBLIC_ (or the equivalent convention for your framework) are bundled into client-side JavaScript" -> This suggests that the frontend might be using Next.js? But note: the variable name is VITE_ (from Vite) in the previous section. However, the document says "the equivalent convention for your framework".

Therefore, the answer to "what backend and frontend use in talk to my data" is:

Backend: FastAPI Frontend: React (with Vite)"""

print("="*60)
print("TESTING SAMPLE 1 (Full Multi-Doc Reasoning Breakdown):")
res_1 = sanitize_response(sample_1)
print(f"OUTPUT 1:\n{res_1!r}")

print("\n" + "="*60)
print("TESTING SAMPLE 2 (Leaked Quote Fragment + Concluding Marker):")
res_2 = sanitize_response(sample_2)
print(f"OUTPUT 2:\n{res_2!r}")
print("="*60)

assert res_1 == "Backend: FastAPI Frontend: React (with Vite)", f"Failed Sample 1: {res_1!r}"
assert res_2 == "Backend: FastAPI Frontend: React (with Vite)", f"Failed Sample 2: {res_2!r}"
print("\n✅ ALL SANITIZATION TESTS PASSED PERFECTLY!")
