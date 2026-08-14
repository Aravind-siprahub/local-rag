import asyncio
import os
import sys

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)

async def run():
    from sqlalchemy import text
    from app.db.session import AsyncSessionLocal
    import json

    async with AsyncSessionLocal() as db:
        print("Checking failed documents...")
        rows = (await db.execute(text(
            "SELECT id, title, status, last_error FROM documents WHERE status='failed' OR deleted_at IS NULL ORDER BY created_at DESC"
        ))).fetchall()
        for r in rows:
            print(f"Doc {r.id}: status={r.status}, title={r.title}, last_error={r.last_error}")
            
        print("Checking processing jobs...")
        jobs = (await db.execute(text(
            "SELECT id, document_id, job_type, status, error_message FROM processing_jobs ORDER BY created_at DESC LIMIT 10"
        ))).fetchall()
        for j in jobs:
            print(f"Job {j.id}: doc={j.document_id}, type={j.job_type}, status={j.status}, error={j.error_message}")
            
        print("Checking document versions...")
        versions = (await db.execute(text(
            "SELECT id, document_id, status, error_message FROM document_versions ORDER BY created_at DESC LIMIT 10"
        ))).fetchall()
        for v in versions:
            print(f"Version {v.id}: doc={v.document_id}, status={v.status}, error={v.error_message}")

if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run())
