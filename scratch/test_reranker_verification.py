import sys
import os

sys.path.insert(0, os.path.abspath("backend"))

from app.retrieval.ranking import RankedResult, _fallback_heuristic_rerank
import uuid

q = "what frontend and backend are using talk to my data"

c1 = RankedResult(
    chunk_id=uuid.uuid4(),
    chunk_text='PRD Section 21: "React talks only to FastAPI — no component below the backend is ever exposed directly to the browser." Section 5: "Frontend — the chat interface..."',
    document_id=uuid.uuid4(),
    document_version_id=None,
    document_title="PRD_Talk_to_My_Data.docx",
    similarity_score=0.45,
    rank=1,
    section_title="Architecture",
    page_number=1,
)

c2 = RankedResult(
    chunk_id=uuid.uuid4(),
    chunk_text='Deployment Guide Section 14.2: "VITE_BACKEND_URL=https://<domain>:<backend-port>" Nginx setup and port configurations.',
    document_id=uuid.uuid4(),
    document_version_id=None,
    document_title="Deployment_Guide.docx",
    similarity_score=0.42,
    rank=2,
    section_title="Nginx Setup",
    page_number=2,
)

c3 = RankedResult(
    chunk_id=uuid.uuid4(),
    chunk_text='VM Setup Guide: Frontend and backend folders installation.',
    document_id=uuid.uuid4(),
    document_version_id=None,
    document_title="VM_Setup_Guide.docx",
    similarity_score=0.40,
    rank=3,
    section_title="Installation",
    page_number=1,
)

candidates = [c2, c1, c3]
reranked = _fallback_heuristic_rerank(q, candidates)

print("="*60)
print("RERANKED RESULTS:")
for idx, (score, cand) in enumerate(reranked, 1):
    print(f"Rank {idx}: score={score:.4f} doc={cand.document_title} section={cand.section_title}")
print("="*60)

assert reranked[0][1].document_title == "PRD_Talk_to_My_Data.docx", f"PRD chunk failed to rank #1! Top was: {reranked[0][1].document_title}"
print("\n✅ PRD ARCHITECTURE CHUNK RANKED #1 PERFECTLY!")
