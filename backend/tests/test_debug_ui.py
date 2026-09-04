import pytest
import asyncio
import json
from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.chat_session import ChatSession
from app.services.chat_session_service import ChatSessionService
from app.services.chat_message_service import ChatMessageService
from app.rag.service import RAGService
from app.retrieval.search import SearchFilters
from sqlalchemy import select

@pytest.mark.asyncio
async def test_debug_ask_stream_response():
    async with AsyncSessionLocal() as session:
        doc_stmt = select(Document).where(Document.deleted_at.is_(None))
        docs = (await session.execute(doc_stmt)).scalars().all()
        print(f"\nActive Documents in DB: {[d.title for d in docs]}")
        assert len(docs) > 0, "No active docs in DB!"

        user_id = docs[0].user_id

        sess_stmt = select(ChatSession).where(ChatSession.user_id == user_id).order_by(ChatSession.created_at.desc())
        chat_sess = (await session.execute(sess_stmt)).scalars().first()
        if not chat_sess:
            sess_svc = ChatSessionService(session)
            chat_sess = await sess_svc.create_session(user_id=user_id, title="Debug Session")

        print(f"Testing with Session ID: {chat_sess.id}, User ID: {user_id}")

        msg_svc = ChatMessageService(session)
        sess_svc = ChatSessionService(session)
        rag_svc = RAGService(session, message_service=msg_svc, session_service=sess_svc)

        test_questions = [
            "What are SipraHub's core values?",
            '"What are SipraHub\'s core values?"',
        ]

        for q in test_questions:
            print(f"\n============================================================")
            print(f"TRACING QUESTION: {q!r}")
            print(f"============================================================")

            events = []
            async for chunk in rag_svc.ask_stream(
                session_id=chat_sess.id,
                question=q,
                filters=SearchFilters(),
                request_id="debug-req-123"
            ):
                events.append(chunk)

            full_tokens = []
            for ev in events:
                print(f"SSE EVENT: {ev.strip()!r}")
                if '"type": "token"' in ev:
                    try:
                        data = json.loads(ev.replace("data: ", "").strip())
                        full_tokens.append(data.get("content", ""))
                    except Exception:
                        pass

            complete_answer = "".join(full_tokens)
            print(f"\n---> FINAL ASSEMBLED STREAMED ANSWER:\n{complete_answer}\n")
