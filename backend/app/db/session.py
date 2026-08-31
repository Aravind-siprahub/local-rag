"""Async SQLAlchemy engine, session factory, and FastAPI dependency.

This module owns the single `AsyncEngine` instance for the process.
Everything else (routers, services, repositories) imports `AsyncSessionLocal`
or `get_db` from here — nothing else calls `create_async_engine()`.
"""
import logging
from collections.abc import AsyncGenerator

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import get_settings

import asyncio
import sys

if sys.platform == "win32" and sys.version_info < (3, 14):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore  # noqa

logger = logging.getLogger(__name__)
settings = get_settings()


def _build_engine() -> AsyncEngine:
    """Create the async engine with production connection-pool settings.

    Notes specific to Supabase (apply identically to sync or async engines):
    - `pool_pre_ping=True` validates a pooled connection with a cheap ping
      before handing it out, so a connection Supabase silently closed (idle
      timeout, restart, failover) is replaced instead of surfacing as a
      request-time error.
    - `pool_recycle` proactively retires connections before Supabase/PgBouncer
      closes them.
    - `connect_args={"sslmode": "require"}` enforces TLS.
    - A pooler URL (port 6543, PgBouncer transaction mode) disables
      SQLAlchemy's own pooling (`NullPool`) and server-side prepared
      statements, since PgBouncer already pools upstream and transaction
      mode doesn't support prepared statements across pooled connections.
    """
    is_pgbouncer = ":6543" in settings.async_database_url or "pooler.supabase.com" in settings.async_database_url

    engine_kwargs: dict = {
        "pool_pre_ping": True,
        "echo": settings.DB_ECHO,
        "connect_args": {"sslmode": "require"},
        "pool_size": getattr(settings, "DB_POOL_SIZE", 5),
        "max_overflow": getattr(settings, "DB_MAX_OVERFLOW", 10),
        "pool_timeout": getattr(settings, "DB_POOL_TIMEOUT", 30),
        "pool_recycle": 120,
    }

    if is_pgbouncer:
        engine_kwargs["connect_args"]["prepare_threshold"] = None

    return create_async_engine(settings.async_database_url, **engine_kwargs)


engine: AsyncEngine = _build_engine()

# `expire_on_commit=False` so ORM objects stay usable for response
# serialization after a request's transaction has committed.
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


@event.listens_for(engine.sync_engine, "connect")
def _log_new_connection(dbapi_connection: object, connection_record: object) -> None:
    logger.debug("New database connection established")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a request-scoped `AsyncSession`."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except SQLAlchemyError as exc:
            try:
                await session.rollback()
            except Exception:
                pass
            logger.exception("Database error during request; transaction rolled back: %s", exc)
            raise
        except (GeneratorExit, Exception):
            try:
                await session.rollback()
            except Exception:
                pass
            raise


async def check_database_connection() -> None:
    """Run a trivial round-trip query against the database.

    Raises the underlying `SQLAlchemyError` on failure — callers (app
    startup, the `/health` endpoint) decide how to react.
    """
    async with AsyncSessionLocal() as session:
        await session.execute(text("SELECT 1"))
