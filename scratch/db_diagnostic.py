"""
Deep DB Diagnostic — checks documents, statuses, and embeddings directly.
Run: python c:\\Users\\ARAVIND\\Desktop\\local-rag\\scratch\\db_diagnostic.py
"""
from __future__ import annotations
import asyncio, os, sys

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)


async def run():
    from sqlalchemy import text
    from app.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        print("\n" + "="*70)
        print("DB DIAGNOSTIC")
        print("="*70)

        # 1. All documents and their statuses
        rows = (await db.execute(text(
            "SELECT id, title, status, user_id, created_at FROM documents WHERE deleted_at IS NULL ORDER BY created_at DESC"
        ))).fetchall()
        print(f"\n[1] DOCUMENTS ({len(rows)} total):")
        for r in rows:
            print(f"  id={r.id}  status={r.status}  title={r.title!r}")

        # 2. All document versions (just id + doc_id)
        rows = (await db.execute(text(
            "SELECT id, document_id FROM document_versions ORDER BY created_at DESC LIMIT 10"
        ))).fetchall()
        print(f"\n[2] DOCUMENT VERSIONS ({len(rows)} shown):")
        for r in rows:
            print(f"  version_id={r.id}  doc_id={r.document_id}")

        # 3. Total chunks
        total_chunks = (await db.execute(text("SELECT COUNT(*) FROM document_chunks"))).scalar()
        print(f"\n[3] TOTAL CHUNKS IN DB: {total_chunks}")

        # 4. Total embeddings + distinct model names
        total_emb = (await db.execute(text("SELECT COUNT(*) FROM embeddings"))).scalar()
        print(f"\n[4] TOTAL EMBEDDINGS IN DB: {total_emb}")
        model_rows = (await db.execute(text(
            "SELECT model_name, COUNT(*) as cnt FROM embeddings GROUP BY model_name"
        ))).fetchall()
        print(f"    Distinct embedding models stored:")
        for r in model_rows:
            print(f"      model={r.model_name!r}  count={r.cnt}")

        # 5. Sample chunks from the PRD doc (if any)
        sample = (await db.execute(text(
            "SELECT dc.id, dc.chunk_index, dc.section_title, LEFT(dc.content, 100) as preview "
            "FROM document_chunks dc "
            "JOIN document_versions dv ON dc.document_version_id = dv.id "
            "JOIN documents d ON dv.document_id = d.id "
            "WHERE d.title ILIKE '%PRD%' OR d.title ILIKE '%Talk%' "
            "ORDER BY dc.chunk_index LIMIT 10"
        ))).fetchall()
        print(f"\n[5] SAMPLE CHUNKS from PRD/Talk docs ({len(sample)} shown):")
        if sample:
            for r in sample:
                print(f"  chunk_index={r.chunk_index}  section={r.section_title!r}  preview={r.preview!r}")
        else:
            print("  ❌ No chunks found for PRD/Talk docs")

        # 6. Check if those chunks have embeddings
        emb_count = (await db.execute(text(
            "SELECT COUNT(*) FROM embeddings e "
            "JOIN document_chunks dc ON e.chunk_id = dc.id "
            "JOIN document_versions dv ON dc.document_version_id = dv.id "
            "JOIN documents d ON dv.document_id = d.id "
            "WHERE d.title ILIKE '%PRD%' OR d.title ILIKE '%Talk%'"
        ))).scalar()
        print(f"\n[6] EMBEDDINGS for PRD/Talk docs: {emb_count}")

        # 7. Check document status = READY
        ready = (await db.execute(text(
            "SELECT COUNT(*) FROM documents WHERE status = 'ready' AND deleted_at IS NULL"
        ))).scalar()
        print(f"\n[7] Documents with status='ready': {ready}")

        print("\nDone.")


if __name__ == "__main__":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run())
