import asyncio
import sys
import os

# Add backend directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.db.session import AsyncSessionLocal
from sqlalchemy import text
from app.rag.service import RAGService
from app.services.chat_session_service import ChatSessionService
from app.services.user_service import UserService

async def main():
    async with AsyncSessionLocal() as session:
        print("=== DATABASE INSPECTION ===")
        # 1. Inspect users
        users_res = await session.execute(text("SELECT id, email, full_name FROM users"))
        users = users_res.fetchall()
        print("USERS in DB:")
        for u in users:
            print(f"  User ID: {u[0]} | Email: {u[1]} | Name: {u[2]}")
        
        # 2. Inspect documents
        docs_res = await session.execute(text("SELECT id, user_id, filename, title, status FROM documents"))
        docs = docs_res.fetchall()
        print("\nDOCUMENTS in DB:")
        for d in docs:
            print(f"  Doc ID: {d[0]} | User ID: {d[1]} | Title: {d[3]} | Filename: {d[2]} | Status: {d[4]}")
            
        # 3. Inspect chunks count
        chunks_res = await session.execute(text("SELECT count(*), document_id FROM document_chunks GROUP BY document_id"))
        chunks_counts = chunks_res.fetchall()
        print("\nDOCUMENT CHUNKS per Doc:")
        for cc in chunks_counts:
            print(f"  Doc ID: {cc[1]} | Count: {cc[0]}")

        # 4. Check chunks text for any mention of talk to my data / frontend / backend
        kw_res = await session.execute(text("SELECT id, document_id, substring(chunk_text from 1 for 150) FROM document_chunks WHERE lower(chunk_text) LIKE '%frontend%' OR lower(chunk_text) LIKE '%backend%' OR lower(chunk_text) LIKE '%framework%' OR lower(chunk_text) LIKE '%talk%'"))
        kw_chunks = kw_res.fetchall()
        print(f"\nCHUNKS WITH KEYWORDS ({len(kw_chunks)} found):")
        for kc in kw_chunks:
            print(f"  Chunk ID: {kc[0]} | Doc ID: {kc[1]} | Text snippet: {repr(kc[2])}")

        # 5. Check active chat sessions
        sess_res = await session.execute(text("SELECT id, user_id, title FROM chat_sessions ORDER BY created_at DESC LIMIT 5"))
        sessions = sess_res.fetchall()
        print("\nRECENT CHAT SESSIONS:")
        for s in sessions:
            print(f"  Session ID: {s[0]} | User ID: {s[1]} | Title: {s[2]}")

if __name__ == "__main__":
    asyncio.run(main())
