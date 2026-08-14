import asyncio
import sys
import os
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
os.chdir(backend_dir)

from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.models.chat_message import ChatMessage

async def main():
    print("Inspecting database messages for attachments...")
    async with AsyncSessionLocal() as session:
        stmt = select(ChatMessage).order_by(ChatMessage.created_at.desc()).limit(15)
        res = await session.execute(stmt)
        messages = list(res.scalars().all())
        
        print(f"Found {len(messages)} recent messages:")
        for idx, msg in enumerate(messages, 1):
            has_attachments = hasattr(msg, "attachments") and msg.attachments
            print(f"#{idx} | Role: {msg.role.value if hasattr(msg.role, 'value') else msg.role} | Content: {msg.content!r} | Attachments: {msg.attachments if has_attachments else 'None'} | Created: {msg.created_at}")

if __name__ == "__main__":
    asyncio.run(main())
