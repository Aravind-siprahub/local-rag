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
    """Confirm the API process is up *and* can reach Postgres.

    Executes `SELECT 1` through the pooled async engine — a genuine
    connectivity check (bad credentials, network issues, paused Supabase
    project, exhausted pool), not just a "the process is running" ping.
    """
    try:
        await db.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        logger.exception("Health check database round-trip failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "database": "disconnected",
                "message": "Could not connect to the database.",
                "error": str(exc),
            },
        ) from exc

    return HealthResponse(status="ok", database="connected")
