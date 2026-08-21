import sys
import os
import json
import asyncio
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import AsyncSessionLocal
from app.rag.service import RAGService
from app.repositories.user_repository import UserRepository
from app.api.dependencies import get_chat_session_service

# Configure logging to see internal RAG logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

async def trace_benchmark():
    async with AsyncSessionLocal() as session:
        user_repo = UserRepository(session)
        users = await user_repo.list_active()
        if not users:
            print("No users found.")
            return
        user_id = users[0].id

        from app.services.chat_session_service import ChatSessionService
        session_service = ChatSessionService(session)
        chat_session = await session_service.create_session(user_id=user_id, title="Trace Session")
        session_id = chat_session.id

        rag_service = RAGService(session)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, "benchmark_data.json")
        with open(data_path, "r", encoding="utf-8") as f:
            benchmark_data = json.load(f)

        print("=== RAG PIPELINE TRACE ===")
        for item in benchmark_data:
            q = item["question"]
            print(f"\n\n--- QUESTION: {q} ---")
            
            # Since ask() yields server-sent events for streaming, we should use the non-streaming internal method 
            # OR we can just iterate over the stream but that won't give us the internal variables easily.
            # Wait, ask() returns a RAGResponse if we call the async methods directly.
            # But wait, in the actual API, it uses streaming or non-streaming?
            # Let's see how `ask()` is defined. If it's a generator, we iterate.
            
            # Actually, `rag_service.ask` is an async generator?
            # Let's check service.py for `ask` signature. It says `async def ask(...) -> RAGResponse:`
            # Wait, the streaming one is called something else or `ask` is NOT a generator.
            try:
                resp = await rag_service.ask(
                    session_id=session_id,
                    question=q
                )
                print(f"MODEL USED: {resp.model}")
                print(f"ANSWER: {resp.answer}")
                print(f"SOURCES: {len(resp.sources)}")
                for s in resp.sources:
                    print(f" - {s.document_title}: {s.chunk_text[:100]}...")
            except Exception as e:
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info < (3, 14):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(trace_benchmark())
