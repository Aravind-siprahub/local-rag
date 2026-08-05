"""Alembic environment configuration.

Wired to the same `Settings`/`Base` the app uses, so migrations always run
against the DATABASE_URL from `.env` — never a URL hardcoded in alembic.ini.

Runs migrations through an `AsyncEngine` (the standard Alembic pattern for
async projects: open an async connection, then run the actual sync-style
migration machinery via `connection.run_sync`) so this stays consistent
with `app/db/session.py` using `create_async_engine` — no second, sync-only
driver dependency is needed just for migrations.
"""
import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.core.config import get_settings
from app.db.base import Base

# Import every model module so Base.metadata is fully populated for
# autogenerate. Add new model modules to app/models/__init__.py, not here.
import app.models  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# NOTE: the schema already exists (deployed via db/sql/*.sql). Once models
# are added to app/models/, generate the first revision with
# `alembic revision --autogenerate` for review, then apply it with
# `alembic stamp head` — NOT `alembic upgrade head` — so Alembic records the
# database as current without re-running DDL against tables that already
# exist.
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.async_database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Generate SQL scripts without a live DB connection (`alembic upgrade --sql`)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations against a live database connection, asynchronously."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(_do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
