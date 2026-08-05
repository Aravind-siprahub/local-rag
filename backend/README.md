# Local RAG — Backend

FastAPI + async SQLAlchemy 2.x + Alembic, Clean Architecture layout, backed
by PostgreSQL (Supabase) via psycopg3's native asyncio support.

This step covers the **foundation only**: config, async DB connection,
health check, logging, error handling. No auth, RAG, embeddings, document
processing, or ORM models yet — those are later steps.

## Folder structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # composition root: creates the FastAPI app
│   ├── core/                    # cross-cutting concerns, no FastAPI/DB imports
│   │   ├── config.py             # Pydantic Settings
│   │   ├── logging.py             # logging setup
│   │   └── exceptions.py           # AppError + global exception handlers
│   ├── db/                        # infrastructure layer: persistence
│   │   ├── base.py                 # DeclarativeBase
│   │   └── session.py               # async engine, session factory, get_db
│   ├── api/                          # interface layer: HTTP routing only
│   │   ├── router.py                  # aggregates endpoint routers
│   │   └── endpoints/
│   │       └── health.py                # GET /health
│   ├── schemas/                          # Pydantic request/response models
│   │   └── health.py
│   ├── models/                            # ORM models (added in a later step —
│   │                                        #   maps to the *existing* Supabase schema)
│   ├── services/                            # business logic (later step)
│   └── repositories/                         # data-access abstractions (later step)
├── alembic/
│   ├── env.py                                 # async-aware, wired to Settings/Base
│   └── versions/
├── alembic.ini
├── run.py                                       # dev entry point (see "Running on Windows" below)
├── requirements.txt
├── .env / .env.example
└── .gitignore
```

**Why this layout:** `core` has no dependency on FastAPI or SQLAlchemy specifics
beyond typed exceptions, so it stays testable in isolation. `db` is the only
place that talks to SQLAlchemy's engine/session machinery. `api` is a thin
HTTP-translation layer — routers depend on `db.get_db` and `schemas`, nothing
routes directly touches the database driver. `services`/`repositories` are
scaffolded but empty: the next step (mapping the existing schema) fills
`models` and `repositories`; the step after that fills `services`.

## File-by-file explanation

**`app/core/config.py`** — `Settings(BaseSettings)`, the single source of
config truth. `DATABASE_URL` is required and validated (non-empty, correct
scheme, no leftover `.env.example` placeholders). `async_database_url`
normalizes it to `postgresql+psycopg://` — the same URL scheme SQLAlchemy
uses for both sync and async engines with the psycopg3 driver.

**`app/core/logging.py`** — one `setup_logging()` call, made in `main.py`
before anything else runs. Ties the `sqlalchemy.engine` logger's verbosity
to `DB_ECHO` so you don't have two separate toggles for the same thing.

**`app/core/exceptions.py`** — `AppError` (and `DatabaseUnavailableError`)
for the service/domain layer to raise without importing FastAPI, plus
`register_exception_handlers()`, the single place that turns any exception
(`AppError`, `SQLAlchemyError`, or anything unhandled) into a consistent
JSON error response.

**`app/db/base.py`** — the shared `DeclarativeBase`. Every future ORM model
inherits from this alone, so `Base.metadata` (what Alembic diffs against)
sees every table.

**`app/db/session.py`** — the core deliverable of this step:
- `create_async_engine` with the same Supabase-aware pooling logic as
  before (pgbouncer pooler detection → `NullPool` + disabled prepared
  statements; direct connection → tuned `pool_size`/`max_overflow`/
  `pool_recycle`/`pool_pre_ping`).
- `AsyncSessionLocal` — `async_sessionmaker`, `expire_on_commit=False`.
- `get_db()` — async generator FastAPI dependency; rolls back on
  `SQLAlchemyError`, otherwise leaves commit to the caller.
- `check_database_connection()` — used by both the startup lifespan check
  and the `/health` endpoint.

**`app/api/endpoints/health.py` + `app/api/router.py`** — `GET /health` runs
`SELECT 1` through `get_db()`; success returns
`{"status": "ok", "database": "connected"}`, failure raises `HTTPException`
(500) with a descriptive, non-leaking error body. `router.py` is the one
place new endpoint modules get registered as they're added.

**`app/main.py`** — `create_app()` factory (keeps the module import-safe for
tests) wires logging, exception handlers, and the API router; a `lifespan`
context manager runs `check_database_connection()` once at boot and logs a
clear error without crash-looping the process if it fails.

**`alembic/env.py`** — uses `async_engine_from_config` + `connection.run_sync`,
the standard Alembic pattern for async SQLAlchemy projects, so migrations
run through the same async psycopg3 driver as the app. **Because the schema
already exists** (deployed via `db/sql/*.sql`), once models are added: run
`alembic revision --autogenerate` to generate and review a migration, then
apply it with `alembic stamp head` — **not** `alembic upgrade head` — so
Alembic marks the database current without re-running DDL against tables
that already exist.

## Running on Windows — a real gotcha, not optional reading

psycopg3's async mode cannot run on Windows' default `ProactorEventLoop` —
it raises `psycopg.InterfaceError` on the first connection attempt. The fix
(`asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())`)
**must run before `uvicorn.run()`'s internal `asyncio.run()` creates the
event loop** — uvicorn only applies this itself in its `--reload`
subprocess/worker path, and it imports your app *after* the loop already
exists, so setting the policy inside `app/main.py` is too late. This is why
[`run.py`](run.py) exists: it sets the policy first, then calls
`uvicorn.run(...)`.

**Verified locally:** running with the plain `uvicorn app.main:app` CLI
reproduces `psycopg.InterfaceError: Psycopg cannot use the
'ProactorEventLoop'...`; running via `python run.py` (or the equivalent
policy-then-import order) does not — confirmed both with and without
`--reload`.

On Linux/macOS (and inside Docker in production) this is a no-op — the
`sys.platform == "win32"` guard means `run.py` behaves identically to
calling `uvicorn app.main:app` directly there.

## Installation

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env: paste your real DATABASE_URL
```

## Running

```bash
# Windows (required, see gotcha above):
python run.py

# Linux/macOS (either works):
python run.py
# or
uvicorn app.main:app --reload
```

API at `http://127.0.0.1:8000`, docs at `http://127.0.0.1:8000/docs`.

## Verifying the database connection

```bash
curl http://127.0.0.1:8000/health
```
Success:
```json
{"status": "ok", "database": "connected"}
```
Failure — HTTP 500 with a body describing the failure (never the raw driver
exception or connection string).

Standalone, without starting the server (verified on Windows — this **does**
still need the event loop policy fix, since plain `asyncio.run()` also
defaults to `ProactorEventLoop` on Windows regardless of uvicorn):
```bash
python -c "import asyncio, sys
if sys.platform == 'win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from app.db.session import check_database_connection
asyncio.run(check_database_connection())
print('OK')"
```
