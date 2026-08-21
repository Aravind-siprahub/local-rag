import sys
import os
import asyncio
import logging

logging.basicConfig(level=logging.INFO)

sys.path.insert(0, os.path.abspath("backend"))

from app.db.session import AsyncSessionLocal
from app.rag.service import RAGService
from app.models.chat_session import ChatSession
from sqlalchemy import select

async def main():
    async with AsyncSessionLocal() as session:
        # Fetch the latest chat session
        stmt = select(ChatSession).order_by(ChatSession.created_at.desc()).limit(1)
        res = await session.execute(stmt)
        chat_sess = res.scalars().first()
        if not chat_sess:
            print("No chat session found in DB!")
            return
            
        print(f"Using chat session: id={chat_sess.id} user_id={chat_sess.user_id}")
        
        service = RAGService(session)
        query = "what backend and frontend use in talk to my data"
        print(f"\nCalling RAGService.ask with query: {query!r}\n")
        rag_res = await service.ask(chat_sess.id, query)
        print("\n" + "="*50)
        print("ANSWER RETURNED BY RAGService.ask:")
        print(rag_res.answer)
        print("="*50)
        print(f"Sources count: {len(rag_res.sources)}")
        for s in rag_res.sources:
            print(f"  Source chunk_id={s.chunk_id} score={s.similarity_score}")

if __name__ == "__main__":
    asyncio.run(main())
