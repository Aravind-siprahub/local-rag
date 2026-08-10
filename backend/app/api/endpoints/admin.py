"""Admin Telemetry & System Statistics Endpoint."""
from __future__ import annotations

import logging
from typing import Any
import httpx
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_version import DocumentVersion
from app.models.embedding import Embedding
from app.models.processing_job import ProcessingJob
from app.models.enums import ProcessingJobStatus, DocumentStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin Dashboard"])


@router.get("/stats", summary="Get comprehensive system statistics & telemetry")
async def get_admin_stats(
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return database totals, job statuses, storage usage, and Ollama health."""
    settings = get_settings()

    # Document counts
    doc_total = (await session.execute(select(func.count(Document.id)).where(Document.deleted_at.is_(None)))).scalar_one()
    doc_ready = (await session.execute(select(func.count(Document.id)).where(Document.deleted_at.is_(None)).where(Document.status == DocumentStatus.READY))).scalar_one()

    # Chunks and Embeddings
    chunk_total = (await session.execute(select(func.count(DocumentChunk.id)))).scalar_one()
    emb_total = (await session.execute(select(func.count(Embedding.id)))).scalar_one()

    # Processing Jobs breakdown
    jobs_total = (await session.execute(select(func.count(ProcessingJob.id)))).scalar_one()
    jobs_pending = (await session.execute(select(func.count(ProcessingJob.id)).where(ProcessingJob.status == ProcessingJobStatus.PENDING))).scalar_one()
    jobs_running = (await session.execute(select(func.count(ProcessingJob.id)).where(ProcessingJob.status == ProcessingJobStatus.RUNNING))).scalar_one()
    jobs_completed = (await session.execute(select(func.count(ProcessingJob.id)).where(ProcessingJob.status == ProcessingJobStatus.COMPLETED))).scalar_one()
    jobs_failed = (await session.execute(select(func.count(ProcessingJob.id)).where(ProcessingJob.status == ProcessingJobStatus.FAILED))).scalar_one()

    # Ollama status check
    ollama_status = "offline"
    ollama_models = []
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            res = await client.get(f"{settings.ollama_host}/api/tags")
            if res.status_code == 200:
                ollama_status = "online"
                data = res.json()
                ollama_models = [m.get("name") for m in data.get("models", [])]
    except Exception as exc:
        logger.warning("Ollama health check failed: %s", exc)

    return {
        "documents": {
            "total": doc_total,
            "ready": doc_ready,
            "processing": doc_total - doc_ready,
        },
        "vectors": {
            "total_chunks": chunk_total,
            "total_embeddings": emb_total,
            "dimension": settings.EMBEDDING_DIMENSIONS,
            "embedding_model": settings.EMBEDDING_MODEL,
        },
        "jobs": {
            "total": jobs_total,
            "pending": jobs_pending,
            "running": jobs_running,
            "completed": jobs_completed,
            "failed": jobs_failed,
        },
        "ollama": {
            "status": ollama_status,
            "host": settings.ollama_host,
            "chat_model": settings.ollama_chat_model,
            "available_models": ollama_models,
        },
        "storage": {
            "provider": settings.STORAGE_PROVIDER,
            "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_MB,
        },
    }
