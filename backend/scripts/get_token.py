import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import AsyncSessionLocal
from app.repositories.user_repository import UserRepository
from app.api.security import create_access_token

async def main():
    async with AsyncSessionLocal() as session:
        user_repo = UserRepository(session)
        # get all active users
        users = await user_repo.list_active()
        if not users:
            print("No users found in database!")
            return
        
        user = users[0]
        token = create_access_token(user.id)
        print(f"Token for {user.email}:")
        print("--------------------------")
        print(token)
        print("--------------------------")
        print(f"Run benchmark with:")
        print(f"python scripts\\benchmark_latency.py --warm --token {token}")

if __name__ == "__main__":
    import sys
    if sys.platform == "win32" and sys.version_info < (3, 14):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
