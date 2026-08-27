"""FastAPI application entry point / composition root.

Run with:
"""
from __future__ import annotations

import asyncio
import logging
import sys
import uuid as uuid_lib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.exc import SQLAlchemyError

if sys.platform == "win32":
    # psycopg3's async mode cannot run on Windows' default ProactorEventLoop
    # (raises psycopg.InterfaceError at the first async connection attempt).
    # This must be set before uvicorn's asyncio.run() creates the event loop,
    # i.e. at import time of this module, not inside an async function.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore
    try:
        loop = asyncio.get_running_loop()
        if isinstance(loop, asyncio.ProactorEventLoop):
            logging.error(
                "CRITICAL WINDOWS ERROR: ProactorEventLoop is active. "
                "Do NOT run 'uvicorn app.main:app --reload'. "
                "Instead run: python run.py"
            )
    except RuntimeError:
        pass

from app.api.router import api_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging
from app.core.openapi import patch_openapi_with_examples
from app.db.session import AsyncSessionLocal, check_database_connection
from app.repositories.chat_session_repository import ChatSessionRepository
from app.repositories.user_repository import UserRepository
from app.services.chat_session_service import ChatSessionService
from app.services.session_resolution import get_or_create_swagger_demo_session

setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook.

    Verifies the database is reachable once, at boot, so a misconfigured
    DATABASE_URL is caught immediately in the logs. The app still finishes
    starting even if this check fails — that keeps `/health` available to
    report the failure (and recover automatically once the database comes
    back) instead of crash-looping a process an orchestrator would just
    keep restarting into the same error.
    """
    logger.info("DATABASE_URL (masked): %s", settings.masked_database_url)
    logger.info(
        "JWT Authentication: algorithm=%s expire_minutes=%d secret_configured=%s",
        settings.JWT_ALGORITHM,
        settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
        "configured (masked)" if settings.JWT_SECRET_KEY else "NO",
    )
    logger.info(
        "Embedding config: model=%s dimensions=%d",
        settings.EMBEDDING_MODEL,
        settings.EMBEDDING_DIMENSIONS,
    )
    logger.info(
        "LLM config: model=%s timeout_seconds=%.1f max_retries=%d temperature=%.2f",
        settings.ollama_chat_model,
        settings.LLM_TIMEOUT_SECONDS,
        settings.LLM_MAX_RETRIES,
        settings.LLM_TEMPERATURE,
    )
    if settings.OLLAMA_USE_GPU:
        logger.info(
            "LLM execution: GPU enabled (host=%s num_gpu=%s num_thread=%s)",
            settings.ollama_host,
            settings.OLLAMA_NUM_GPU if settings.OLLAMA_NUM_GPU is not None else "default",
            settings.OLLAMA_NUM_THREAD if settings.OLLAMA_NUM_THREAD is not None else "default",
        )
    else:
        logger.info(
            "LLM execution: CPU fallback (host=%s num_gpu=0 num_thread=%s)",
            settings.ollama_host,
            settings.OLLAMA_NUM_THREAD if settings.OLLAMA_NUM_THREAD is not None else "default",
        )

    try:
        await asyncio.wait_for(check_database_connection(), timeout=5.0)
    except (SQLAlchemyError, asyncio.TimeoutError, Exception) as exc:
        logger.error(
            "STARTUP CHECK FAILED: could not connect to the database within 5s. "
            "Verify DATABASE_URL in .env and that the database is reachable. "
            "Error: %s",
            exc,
        )
    else:
        logger.info("Startup check passed: database connection OK")

        # Auto-ensure missing database columns exist safely
        try:
            async with AsyncSessionLocal() as db_sess:
                from sqlalchemy import text
                await db_sess.execute(text("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS working_memory_summary TEXT;"))
                await db_sess.commit()
                logger.info("[DB MIGRATION] Column chat_sessions.working_memory_summary verified OK.")
        except Exception as mig_exc:
            logger.warning("[DB MIGRATION WARNING] %s", mig_exc)

        app.state.swagger_example_user_email = (
            f"swagger-{uuid_lib.uuid4().hex[:8]}@example.com"
        )
        
        # Standalone search provider test on boot/reload
        async def run_search_test():
            await asyncio.sleep(1.0) # Let server finish loading
            logger.info("=== STARTING STARTUP WEB SEARCH TEST ===")
            from app.tools.web_search import get_web_search_provider
            try:
                provider = get_web_search_provider()
                res = await provider.search("When is Deepawali in 2026")
                logger.info(f"=== WEB SEARCH TEST PASSED: {len(res.hits)} hits ===")
            except Exception as e:
                logger.error(f"=== WEB SEARCH TEST FAILED: {e} ===", exc_info=True)
                
        asyncio.create_task(run_search_test())
        logger.info(
            "Swagger example user email set to %s",
            app.state.swagger_example_user_email,
        )
        try:
            async with AsyncSessionLocal() as session:
                users = await UserRepository(session).list_active(limit=1)
                if users:
                    app.state.swagger_example_user_id = str(users[0].id)
                    logger.info(
                        "Swagger example user_id set to first active user %s",
                        app.state.swagger_example_user_id,
                    )
                    demo_session_id = await get_or_create_swagger_demo_session(
                        users=UserRepository(session),
                        sessions=ChatSessionRepository(session),
                        session_service=ChatSessionService(session),
                        user_id=users[0].id,
                    )
                    app.state.swagger_example_session_id = str(demo_session_id)
                    logger.info(
                        "Swagger example session_id set to %s",
                        app.state.swagger_example_session_id,
                    )
        except SQLAlchemyError as exc:
            logger.warning("Could not load Swagger example ids: %s", exc)

        # Warm up Ollama chat model in background ONLY if Ollama is the selected provider
        async def _warmup_ollama() -> None:
            try:
                settings = get_settings()
                if (getattr(settings, "LLM_PROVIDER", "") or "").strip().lower() != "ollama":
                    logger.info("stage=ollama_warmup request_id=N/A action=SKIPPED reason=provider_not_selected provider=%s", getattr(settings, "LLM_PROVIDER", "omniroute"))
                    return
                from app.llm.ollama_client import get_global_ollama_client
                client = get_global_ollama_client()
                await client.warmup()
            except Exception as exc:
                logger.warning("Ollama background warmup error: %s", exc)

        # Auto-ingest any un-processed or pending documents on boot
        async def _ingest_pending_documents() -> None:
            try:
                async with AsyncSessionLocal() as session:
                    from app.models.document import Document
                    from app.models.enums import DocumentStatus
                    from app.services.ingestion_service import IngestionService
                    from sqlalchemy import select

                    stmt = select(Document).where(
                        Document.deleted_at.is_(None),
                        Document.status.in_([DocumentStatus.UPLOADED, DocumentStatus.PROCESSING, DocumentStatus.FAILED]),
                    )
                    docs = list((await session.execute(stmt)).scalars().all())
                    doc_ids = [d.id for d in docs]
                    if doc_ids:
                        logger.info("Startup: found %d pending/uploaded/failed document(s) to ingest", len(doc_ids))
                        for doc_id in doc_ids:
                            try:
                                async with AsyncSessionLocal() as ing_session:
                                    ingestion = IngestionService(ing_session)
                                    logger.info("Startup auto-ingesting document %s...", doc_id)
                                    await ingestion.run_pipeline(doc_id)
                                    await ing_session.commit()
                            except Exception as exc:
                                logger.warning("Startup ingestion error for doc %s: %s", doc_id, exc)

            except Exception as exc:
                if "storage_provider" in str(exc) or "UndefinedColumn" in str(exc):
                    logger.warning(
                        "Supabase Storage metadata columns missing in PostgreSQL. "
                        "Run the migration in Supabase SQL Editor: db/sql/007_supabase_storage_columns.sql"
                    )
                else:
                    logger.warning("Startup pending document check error: %s", exc)

        from app.processing.background_runner import start_background_runner

        asyncio.create_task(_warmup_ollama())
        asyncio.create_task(_ingest_pending_documents())
        start_background_runner()


        # Force OpenAPI rebuild after startup examples are known, so the first
        # /docs hit never serves a schema generated without those patches.
        app.openapi_schema = None

    yield

    logger.info("Application shutting down")
    
    # Clean up the global persistent LLM client
    try:
        from app.llm.ollama_client import get_global_ollama_client
        client = get_global_ollama_client()
        await client.close()
    except Exception as exc:
        logger.warning("Error closing global OllamaLLMClient: %s", exc)


from fastapi.middleware.cors import CORSMiddleware

def create_app() -> FastAPI:
    """Application factory.

    A factory (rather than a bare module-level `FastAPI()`) keeps `main.py`
    importable for tests without side effects beyond constructing the app,
    and keeps configuration/wiring in one place.
    """
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.routing import APIRoute

    def custom_generate_unique_id(route: APIRoute) -> str:
        # The api_router is mounted twice (at / and /api), so we must include
        # the path in the operation ID to prevent duplicates in OpenAPI generation.
        return f"{route.tags[0] if route.tags else 'api'}-{route.name}-{route.path_format.strip('/').replace('/', '_')}"

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
        generate_unique_id_function=custom_generate_unique_id,
    )

    # CORS must be registered before routers/handlers so preflight and error
    # responses (404/500) still include Access-Control-* headers for the SPA.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    # Primary mount: frontend VITE_API_BASE_URL=http://localhost:8000/api
    app.include_router(api_router, prefix="/api")
    # Root mount: Vite proxy rewrite strips /api -> /health, /users, etc.
    app.include_router(api_router)

    def custom_openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema

        from fastapi.openapi.utils import get_openapi

        schema = get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            routes=app.routes,
        )
        example_user_id = getattr(app.state, "swagger_example_user_id", None)
        example_user_email = getattr(app.state, "swagger_example_user_email", None)
        example_session_id = getattr(app.state, "swagger_example_session_id", None)
        if example_user_id or example_user_email or example_session_id:
            patch_openapi_with_examples(
                schema,
                example_user_id=example_user_id,
                example_user_email=example_user_email,
                example_session_id=example_session_id,
            )

        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi

    return app


app = create_app()


@app.get("/", tags=["root"])
async def root() -> dict[str, str]:
    return {"service": settings.APP_NAME, "version": settings.APP_VERSION}
