import pytest
from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.chat_session import ChatSession
from app.models.chat_message import ChatMessage
from app.models.user import User
from sqlalchemy import select

@pytest.mark.asyncio
async def test_check_user_id_and_document_ownership():
    async with AsyncSessionLocal() as session:
        print("\n============================================================")
        print("DIAGNOSTIC CHECK: USERS, DOCUMENTS & CHAT SESSIONS IN DB")
        print("============================================================\n")

        # 1. Users
        users = (await session.execute(select(User))).scalars().all()
        print(f"--- USERS ({len(users)}) ---")
        for u in users:
            print(f" User ID: {u.id} | Email: {u.email} | Name: {u.full_name}")

        # 2. Active Documents
        doc_stmt = select(Document).where(Document.deleted_at.is_(None))
        docs = (await session.execute(doc_stmt)).scalars().all()
        print(f"\n--- ACTIVE DOCUMENTS ({len(docs)}) ---")
        for d in docs:
            print(f" Doc ID: {d.id} | User ID: {d.user_id} | Title: {d.title} | Status: {d.status} | CurrentVer: {d.current_version_id}")

        # 3. Chat Sessions
        sess_stmt = select(ChatSession).order_by(ChatSession.created_at.desc())
        sessions = (await session.execute(sess_stmt)).scalars().all()
        print(f"\n--- RECENT CHAT SESSIONS ({len(sessions[:10])}) ---")
        for s in sessions[:10]:
            print(f" Session ID: {s.id} | User ID: {s.user_id} | Title: {s.title} | Created: {s.created_at}")

        # 4. Check if any Chat Session user_id DOES NOT MATCH Document user_id
        if docs and sessions:
            doc_user_ids = {str(d.user_id) for d in docs}
            sess_user_ids = {str(s.user_id) for s in sessions[:10]}
            print(f"\nDocument User IDs: {doc_user_ids}")
            print(f"Session User IDs:  {sess_user_ids}")
            
            common = doc_user_ids.intersection(sess_user_ids)
            print(f"Common User IDs:   {common}")
            if not common:
                print("\n[ALERT] CRITICAL USER ID MISMATCH DETECTED!")
                print("The user_id of the chat session created by the frontend DOES NOT MATCH the user_id of the uploaded documents in PostgreSQL!")
