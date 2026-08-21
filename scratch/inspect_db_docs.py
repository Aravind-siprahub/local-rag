import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath("backend"))

from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from sqlalchemy import select

async def main():
    async with AsyncSessionLocal() as session:
        stmt = select(Document).where(Document.deleted_at.is_(None))
        docs = list((await session.execute(stmt)).scalars().all())
        print(f"Total Active Documents in DB: {len(docs)}")
        for d in docs:
            print(f"Doc ID: {d.id} | User ID: {d.user_id} | Title: {d.title!r} | Version ID: {d.current_version_id} | Status: {d.status}")
            
            # Check chunks count
            c_stmt = select(DocumentChunk).where(DocumentChunk.document_version_id == d.current_version_id)
            chunks = list((await session.execute(c_stmt)).scalars().all())
            print(f"  -> Chunks count: {len(chunks)}")
            for c in chunks:
                if "react" in c.content.lower() or "fastapi" in c.content.lower():
                    print(f"     [MATCH CHUNK]: chunk_id={c.id} text={c.content[:100]!r}")

if __name__ == "__main__":
    asyncio.run(main())
