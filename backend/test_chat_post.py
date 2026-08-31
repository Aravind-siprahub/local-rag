import asyncio
import httpx
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db.session import AsyncSessionLocal
from app.repositories.user_repository import UserRepository
from app.api.security import create_access_token

async def get_test_token() -> str:
    token = os.environ.get("AUTH_TOKEN")
    if token and token.strip():
        return token.strip()
    try:
        async with AsyncSessionLocal() as session:
            user_repo = UserRepository(session)
            users = await user_repo.list_active()
            if users:
                user = users[0]
                t = create_access_token(user.id)
                print(f"Auto-generated Bearer token for user: {user.email}")
                return t
    except Exception as err:
        print(f"Failed to auto-generate user token from database: {err}")
    return ""

async def test_live_chat(question: str | None = None):
    if not question:
        question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is inside PRD_Talk_to_My_Data.docx?"

    url = "http://127.0.0.1:8000/api/chat"
    payload = {
        "question": question
    }

    token = await get_test_token()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    print("=== SENDING POST REQUEST TO /api/chat ===")
    print(f"URL: {url}")
    print(f"Question: {question}")
    print(f"Auth Header: {'Present' if token else 'Missing'}")

    async with httpx.AsyncClient(timeout=300.0) as client:
        res = await client.post(url, json=payload, headers=headers)
        print(f"\nResponse Status: {res.status_code}")
        data = res.json()
        print("\n=== FINAL RESPONSE RETURNED TO FRONTEND ===")
        print(json.dumps(data, indent=2))

if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info < (3, 12):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
    asyncio.run(test_live_chat())
