import asyncio
import logging
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_version import DocumentVersion
from app.models.enums import DocumentStatus
from app.rag.intent_router import classify, Route
from app.retrieval.search import SearchFilters

logging.basicConfig(level=logging.INFO)

async def trace():
    q = "what is problem statement in my talk to my data"
    route = classify(q)
    print(f"=== INTENT ROUTER RESULT ===")
    print(f"Query: '{q}'")
    print(f"Route: {route}")
    
    async with AsyncSessionLocal() as session:
        # Fetch document by title matching 'talk to my data' or 'prd'
        stmt = select(Document).where(Document.deleted_at.is_(None))
        res = await session.execute(stmt)
        docs = list(res.scalars().all())
        print(f"\n=== DOCUMENTS IN DATABASE ({len(docs)}) ===")
        for d in docs:
            print(f"Doc ID: {d.id} | Title: '{d.title}' | Status: {d.status} | User: {d.user_id}")
            
        # Try finding chunks matching PRD document
        prd_docs = [d for d in docs if "talk" in d.title.lower() or "prd" in d.title.lower() or "data" in d.title.lower()]
        print(f"\n=== MATCHING DOCS FOR QUERY ({len(prd_docs)}) ===")
        for d in prd_docs:
            stmt_chunks = (
                select(DocumentChunk)
                .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
                .where(DocumentVersion.document_id == d.id)
            )
            res_c = await session.execute(stmt_chunks)
            chunks = list(res_c.scalars().all())
            print(f"Doc '{d.title}' has {len(chunks)} chunks:")
            for idx, c in enumerate(chunks[:3], 1):
                print(f"  Chunk {idx} (ID: {c.id}): {c.content[:200]}...")

if __name__ == "__main__":
    asyncio.run(trace())
