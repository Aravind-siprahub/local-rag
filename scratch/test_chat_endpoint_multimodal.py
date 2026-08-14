import asyncio
import logging
import sys
import uuid
import os
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
os.chdir(backend_dir)

import httpx
from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import AsyncSessionLocal
from app.models.user import User
from app.models.chat_session import ChatSession

# 100x100 Red PNG
def get_test_image_bytes() -> bytes:
    from PIL import Image
    import io
    img = Image.new("RGB", (100, 100), color="red")
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()

async def main():
    settings = get_settings()
    print("Connecting to DB to find an active user and chat session...")
    
    async with AsyncSessionLocal() as session:
        # Find user
        stmt_user = select(User).limit(1)
        res_user = await session.execute(stmt_user)
        user = res_user.scalars().first()
        if not user:
            print("No users found in database.")
            return
            
        # Find or create a session
        stmt_sess = select(ChatSession).where(ChatSession.user_id == user.id).limit(1)
        res_sess = await session.execute(stmt_sess)
        chat_sess = res_sess.scalars().first()
        if not chat_sess:
            chat_sess = ChatSession(user_id=user.id, title="Test Chat")
            session.add(chat_sess)
            await session.commit()
            await session.refresh(chat_sess)
            
        user_id = str(user.id)
        session_id = str(chat_sess.id)
        
    print(f"User ID: {user_id}")
    print(f"Session ID: {session_id}")
    
    # Generate a JWT token
    import jwt
    from datetime import datetime, timedelta, timezone
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(days=1)
    }
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    
    # Send HTTP Multipart Request
    url = "http://localhost:8000/api/chat"
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Request-ID": f"test-{uuid.uuid4().hex[:8]}"
    }
    
    image_bytes = get_test_image_bytes()
    
    files = {
        "file": ("test_red.png", image_bytes, "image/png")
    }
    data = {
        "question": "What color is this image? Keep it extremely short.",
        "session_id": session_id
    }
    
    print("\nSending POST request to /api/chat with multipart/form-data...")
    print(f"Headers: {headers}")
    print(f"Data: {data}")
    print(f"Files: {files.keys()}")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(url, headers=headers, data=data, files=files)
            print("\nResponse Received!")
            print(f"Status Code: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            print(f"Body: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
