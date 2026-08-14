import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
from dotenv import load_dotenv

# Load environment variables from the backend/.env file if it exists
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=dotenv_path)

# Parser/cleaner/chunker tests do not need a database, but importing the
# processor or services pulls in SQLAlchemy settings validation.
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres",
)

import pytest
import pytest_asyncio
from app.db.session import AsyncSessionLocal


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_database():
    """Ensure all tables (including RAGTrace) exist before running any tests."""
    from app.db.base import Base
    from app.db.session import engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


@pytest_asyncio.fixture
async def db_session():
    """Provide a transactional AsyncSession for test setup and override the FastAPI get_db dependency."""
    from app.api.dependencies import get_db
    from app.main import app

    async with AsyncSessionLocal() as session:
        async def override_get_db():
            yield session

        app.dependency_overrides[get_db] = override_get_db
        try:
            yield session
        finally:
            app.dependency_overrides.pop(get_db, None)
            await session.rollback()


@pytest.fixture(autouse=True)
def clear_embedding_cache():
    """Clear embedding cache between tests to prevent cache-pollution."""
    from app.embeddings.client import _QUERY_EMBEDDING_CACHE
    _QUERY_EMBEDDING_CACHE.clear()

