import asyncio
import logging
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from sqlalchemy import select, func
from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.models.document_chunk import DocumentChunk
from app.models.enums import DocumentStatus

logging.basicConfig(level=logging.INFO)

async def inspect_db():
    async with AsyncSessionLocal() as session:
        # 1. Fetch documents
        stmt_docs = select(Document).where(Document.deleted_at.is_(None))
        res_docs = await session.execute(stmt_docs)
        docs = list(res_docs.scalars().all())
        print(f"=== TOTAL ACTIVE DOCUMENTS: {len(docs)} ===")
        for d in docs:
            print(f"Doc ID: {d.id} | Title: {d.title} | Status: {d.status} | User ID: {d.user_id}")
            
            # Fetch versions
            stmt_v = select(DocumentVersion).where(DocumentVersion.document_id == d.id)
            res_v = await session.execute(stmt_v)
            versions = list(res_v.scalars().all())
            print(f"  Versions count: {len(versions)}")
            for v in versions:
                # Fetch chunks count
                stmt_c = select(func.count(DocumentChunk.id)).where(DocumentChunk.document_version_id == v.id)
                res_c = await session.execute(stmt_c)
                chunk_count = res_c.scalar()
                
                # Fetch first 2 chunks text
                stmt_chunks = select(DocumentChunk).where(DocumentChunk.document_version_id == v.id).limit(2)
                res_chunks = await session.execute(stmt_chunks)
                sample_chunks = list(res_chunks.scalars().all())
                
                print(f"    Version ID: {v.id} | Status: {v.status} | Chunks count: {chunk_count}")
                for idx, c in enumerate(sample_chunks, 1):
                    emb_dim = len(c.embeddings) if c.embeddings is not None else 0
                    print(f"      Chunk {idx} ID: {c.id} | Emb Dim: {emb_dim} | Text snippet: {c.content[:120]}...")

if __name__ == "__main__":
    asyncio.run(inspect_db())
