import asyncio
import json
import os
import sys

# Ensure backend path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from app.db.session import AsyncSessionLocal
from app.rag.service import RAGService
from app.rag.intent_router import classify
from app.repositories.user_repository import UserRepository
from app.services.chat_session_service import ChatSessionService

async def check():
    with open("scripts/benchmark_data.json", "r") as f:
        data = json.load(f)
        
    async with AsyncSessionLocal() as db_session:
        user_repo = UserRepository(db_session)
        users = await user_repo.list_active()
        user_id = users[0].id
        session_service = ChatSessionService(db_session)
        chat_session = await session_service.create_session(user_id=user_id, title="Benchmark Test")
        rag_service = RAGService(db_session)
        
        for idx, item in enumerate(data, 1):
            q = item["question"]
            expected = item["expected"]
            route = classify(q)
            resp = await rag_service.ask(chat_session.id, q)
            actual = resp.answer
            
            # evaluate
            is_in = expected.lower() in actual.lower()
            print(f"Q{idx}: {q}")
            print(f"Expected: {expected}")
            print(f"Actual: {actual}")
            print(f"Contains Expected? {is_in}\n")

if __name__ == "__main__":
    asyncio.run(check())
