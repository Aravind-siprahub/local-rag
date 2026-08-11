"""Application Performance Metrics Endpoint."""
from __future__ import annotations

import time
from typing import Any
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.core.config import get_settings

router = APIRouter(prefix="/metrics", tags=["Metrics"])

_METRICS_STORE = {
    "requests_total": 0,
    "retrieval_latency_ms": [],
    "embedding_latency_ms": [],
    "llm_latency_ms": [],
    "errors_total": 0,
}

def record_metric(category: str, value: float) -> None:
    if category in _METRICS_STORE and isinstance(_METRICS_STORE[category], list):
        _METRICS_STORE[category].append(value)
        if len(_METRICS_STORE[category]) > 100:
            _METRICS_STORE[category].pop(0)

@router.get("", summary="Get system metrics & latency statistics")
async def get_metrics(
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    settings = get_settings()

    def _avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 2) if lst else 0.0

    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "uptime_seconds": int(time.process_time()),
        "requests_total": _METRICS_STORE["requests_total"],
        "errors_total": _METRICS_STORE["errors_total"],
        "latencies": {
            "avg_retrieval_ms": _avg(_METRICS_STORE["retrieval_latency_ms"]),
            "avg_embedding_ms": _avg(_METRICS_STORE["embedding_latency_ms"]),
            "avg_llm_ms": _avg(_METRICS_STORE["llm_latency_ms"]),
        },
    }
