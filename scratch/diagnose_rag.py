import asyncio
import sys
from pathlib import Path

# Add backend to sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from sqlalchemy import select, func
from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.models.document_chunk import DocumentChunk
from app.models.embedding import Embedding
from app.models.user import User


async def diagnose():
    async with AsyncSessionLocal() as session:
        print("=== 1. USERS IN DATABASE ===")
        users = (await session.execute(select(User))).scalars().all()
        print(f"Total Users: {len(users)}")
        for u in users:
            print(f"  User ID: {u.id} | Email: {u.email} | Active: {u.is_active}")

        print("\n=== 2. DOCUMENTS IN DATABASE ===")
        docs = (await session.execute(select(Document))).scalars().all()
        print(f"Total Documents: {len(docs)}")
        for d in docs:
            print(f"  Doc ID: {d.id} | Title: {d.title} | Status: {d.status} | User ID: {d.user_id} | Curr Version: {d.current_version_id} | Deleted: {d.deleted_at}")

        print("\n=== 3. DOCUMENT VERSIONS ===")
        versions = (await session.execute(select(DocumentVersion))).scalars().all()
        print(f"Total Versions: {len(versions)}")
        for v in versions:
            print(f"  Version ID: {v.id} | Doc ID: {v.document_id} | Version Num: {v.version_number}")

        print("\n=== 4. CHUNKS COUNT PER DOCUMENT ===")
        chunks_count = (await session.execute(select(DocumentChunk.document_version_id, func.count(DocumentChunk.id)).group_by(DocumentChunk.document_version_id))).all()
        print(f"Chunks Count: {chunks_count}")

        print("\n=== 5. EMBEDDINGS COUNT ===")
        embs_count = (await session.execute(select(Embedding.model_name, func.count(Embedding.id)).group_by(Embedding.model_name))).all()
        print(f"Embeddings by Model: {embs_count}")

        # Sample chunk check for PRD_Talk_to_My_Data.docx
        prd_doc = next((d for d in docs if "Talk_to_My_Data" in d.title or "PRD" in d.title), None)
        if prd_doc:
            print(f"\nFound PRD Document: ID={prd_doc.id}, Title={prd_doc.title}, UserID={prd_doc.user_id}")
            prd_chunks = (await session.execute(select(DocumentChunk).where(DocumentChunk.document_version_id == prd_doc.current_version_id))).scalars().all()
            print(f"PRD Chunks Count: {len(prd_chunks)}")
            if prd_chunks:
                print(f"Sample PRD Chunk 1: {prd_chunks[0].content[:200]}...")
        else:
            print("\nPRD_Talk_to_My_Data.docx WAS NOT FOUND IN DATABASE DOCUMENTS!")


if __name__ == "__main__":
    asyncio.run(diagnose())
