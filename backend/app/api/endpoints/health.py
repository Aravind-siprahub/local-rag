"""Health check endpoint."""
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.health import HealthErrorResponse, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={500: {"model": HealthErrorResponse}},
    summary="Liveness/readiness check including a real database round-trip",
)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """Confirm the API process is up, can reach Postgres, pgvector, and Ollama."""
    import httpx
    from app.core.config import get_settings
    settings = get_settings()
    
    db_status = "disconnected"
    pgvector_status = "disconnected"
    ollama_status = "disconnected"
    models_available = []

    # 1. Check Database
    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except SQLAlchemyError as exc:
        logger.exception("Health check database round-trip failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "database": "disconnected",
                "pgvector": "disconnected",
                "ollama": "unknown",
                "message": "Database connectivity check failed.",
            },
        ) from exc

    # 2. Check pgvector
    try:
        result = await db.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
        ext = result.scalar()
        if ext == 'vector':
            pgvector_status = "connected"
        else:
            pgvector_status = "missing"
    except SQLAlchemyError as exc:
        logger.exception("Health check pgvector check failed")
        pgvector_status = "error"

    # 3. Check Ollama
    try:
        ollama_url = f"{settings.ollama_host}/api/tags"
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(ollama_url)
            if response.status_code == 200:
                ollama_status = "connected"
                data = response.json()
                models_available = [m.get("name") for m in data.get("models", []) if m.get("name")]
            else:
                ollama_status = f"error_{response.status_code}"
    except Exception as exc:
        logger.warning(f"Health check Ollama connection failed: {exc}")
        ollama_status = "disconnected"
        
    return HealthResponse(
        status="ok",
        database=db_status,
        pgvector=pgvector_status,
        ollama=ollama_status,
        models=models_available,
    )
