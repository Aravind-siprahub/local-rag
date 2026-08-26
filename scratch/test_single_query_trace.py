import asyncio
import sys
import uuid
from pathlib import Path

# Ensure backend in sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.db.session import AsyncSessionLocal
from app.rag.service import RAGService
from app.retrieval.search import SearchFilters
from app.repositories.user_repository import UserRepository
from app.repositories.chat_session_repository import ChatSessionRepository
from app.services.chat_session_service import ChatSessionService
from app.services.session_resolution import get_or_create_swagger_demo_session


async def test_trace():
    async with AsyncSessionLocal() as session:
        user_repo = UserRepository(session)
        users = await user_repo.list_active(limit=1)
        if not users:
            print("No active user found!")
            return
        user = users[0]

        session_repo = ChatSessionRepository(session)
        session_svc = ChatSessionService(session)
        demo_session_id = await get_or_create_swagger_demo_session(
            users=user_repo,
            sessions=session_repo,
            session_service=session_svc,
            user_id=user.id,
        )

        rag = RAGService(session)

        print("\n=======================================================")
        print("  TRACE 1: In-Scope Question")
        print("  Query: 'What is RAG in this system?'")
        print("=======================================================")

        res1 = await rag.ask(
            session_id=demo_session_id,
            question="What is RAG in this system?",
            filters=SearchFilters(user_id=user.id),
        )

        print(f"Processing Time: {res1.processing_time_ms} ms")
        print(f"Model Used:      {res1.model}")
        print(f"Answer:          {res1.answer}")
        print(f"Sources Count:   {len(res1.sources)}")
        for s in res1.sources:
            print(f"  - Source Doc: {s.document_title} | Score: {s.similarity_score:.4f}")

        print("\n=======================================================")
        print("  TRACE 2: Out-of-Context Question")
        print("  Query: 'What is the stock price of Apple Inc today?'")
        print("=======================================================")

        res2 = await rag.ask(
            session_id=demo_session_id,
            question="What is the stock price of Apple Inc today?",
            filters=SearchFilters(user_id=user.id),
        )

        print(f"Processing Time: {res2.processing_time_ms} ms")
        print(f"Answer:          {res2.answer}")
        print(f"Sources Count:   {len(res2.sources)}")


if __name__ == "__main__":
    asyncio.run(test_trace())
