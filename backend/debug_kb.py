import asyncio
import uuid
from sqlalchemy import select, func
from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.models.document_chunk import DocumentChunk
from app.models.embedding import Embedding

async def main():
    async with AsyncSessionLocal() as session:
        docs = list((await session.execute(select(Document).where(Document.deleted_at.is_(None)))).scalars().all())
        print(f"=== Total active documents in DB: {len(docs)} ===")
        for doc in docs:
            print(f"\nDocument ID: {doc.id}")
            print(f"  Title: {doc.title}")
            print(f"  Status: {doc.status}")
            print(f"  User ID: {doc.user_id}")
            
            versions = list((await session.execute(select(DocumentVersion).where(DocumentVersion.document_id == doc.id))).scalars().all())
            print(f"  Versions count: {len(versions)}")
            for v in versions:
                print(f"    Version ID: {v.id}, status={v.status}, error={v.error_message}")
                chunks = list((await session.execute(select(DocumentChunk).where(DocumentChunk.document_version_id == v.id))).scalars().all())
                print(f"    Chunks count: {len(chunks)}")
                if chunks:
                    chunk_ids = [c.id for c in chunks]
                    vecs = (await session.execute(select(func.count(Embedding.id)).where(Embedding.chunk_id.in_(chunk_ids)))).scalar_one()
                    print(f"    Embeddings count: {vecs}")
                    print(f"    First chunk preview: {chunks[0].content[:100]!r}")

if __name__ == "__main__":
    asyncio.run(main())
