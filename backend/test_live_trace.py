import asyncio
import sys

from app.db.session import AsyncSessionLocal
from app.rag.service import RAGService
from app.repositories.chat_session_repository import ChatSessionRepository
from app.repositories.user_repository import UserRepository
from app.services.chat_session_service import ChatSessionService
from app.services.session_resolution import get_or_create_swagger_demo_session


async def main() -> None:
    async with AsyncSessionLocal() as session:
        session_id = await get_or_create_swagger_demo_session(
            users=UserRepository(session),
            sessions=ChatSessionRepository(session),
            session_service=ChatSessionService(session),
        )

        rag = RAGService(session)
        q = "What is inside Deployment_Guide.docx?"
        res = await rag.ask(session_id, q, top_k=2, similarity_threshold=0.0)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
