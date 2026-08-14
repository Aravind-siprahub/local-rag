"""BackgroundJobRunner: reliable non-blocking background job worker for document ingestion."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid

from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.models.document import Document
from app.models.enums import DocumentStatus, DocumentVersionStatus, ProcessingJobStatus, ProcessingJobType
from app.models.processing_job import ProcessingJob
from app.services.document_service import DocumentService
from app.services.document_version_service import DocumentVersionService
from app.services.processing_job_service import ProcessingJobService
from app.processing.processor import DocumentProcessor
from app.embeddings.worker import EmbeddingWorker

logger = logging.getLogger(__name__)

_ACTIVE_DOCUMENT_IDS: set[uuid.UUID] = set()
_RUNNER_TASK: asyncio.Task | None = None


class BackgroundJobRunner:
    """Manages automatic execution of PARSE and EMBED jobs for uploaded documents."""

    @staticmethod
    def enqueue_document(document_id: uuid.UUID) -> None:
        """Trigger background ingestion task for document_id non-blockingly."""
        asyncio.create_task(BackgroundJobRunner._process_document_safely(document_id))

    @staticmethod
    async def _process_document_safely(document_id: uuid.UUID) -> None:
        if document_id in _ACTIVE_DOCUMENT_IDS:
            logger.debug("[BACKGROUND RUNNER] Document %s already in progress, skipping", document_id)
            return

        _ACTIVE_DOCUMENT_IDS.add(document_id)
        try:
            async with AsyncSessionLocal() as session:
                doc_service = DocumentService(session)
                version_service = DocumentVersionService(session)
                job_service = ProcessingJobService(session)

                document = await doc_service.get(document_id)
                if not document or not document.current_version_id:
                    return

                version = await version_service.get(document.current_version_id)
                if not version:
                    return

                logger.info("[BACKGROUND RUNNER] Starting ingestion pipeline for document %s...", document_id)
                await doc_service.update(document_id, status=DocumentStatus.PROCESSING)
                await session.commit()

                # 1. PARSE Stage
                if version.status in (DocumentVersionStatus.UPLOADED, DocumentVersionStatus.PARSING):
                    all_jobs = await job_service.list_by_document_version(version.id)
                    parse_jobs = [j for j in all_jobs if j.job_type == ProcessingJobType.PARSE]
                    active_parse = [j for j in parse_jobs if j.status in (ProcessingJobStatus.PENDING, ProcessingJobStatus.RUNNING)]
                    parse_job = active_parse[-1] if active_parse else (parse_jobs[-1] if parse_jobs else None)
                    
                    if not parse_job or parse_job.status in (ProcessingJobStatus.COMPLETED, ProcessingJobStatus.FAILED):
                        parse_job = await job_service.create_job(document_version_id=version.id, job_type=ProcessingJobType.PARSE)
                        await session.commit()

                    if parse_job.status != ProcessingJobStatus.COMPLETED:
                        processor = DocumentProcessor(session)
                        try:
                            await processor.process_job(parse_job.id)
                            await session.commit()
                        except Exception:
                            await session.rollback()
                            await doc_service.update(document_id, status=DocumentStatus.FAILED)
                            await session.commit()
                            raise

                # Refresh version state
                version = await version_service.get(document.current_version_id)

                # 2. EMBED Stage
                if version.status in (DocumentVersionStatus.CHUNKED, DocumentVersionStatus.EMBEDDING):
                    all_jobs = await job_service.list_by_document_version(version.id)
                    embed_jobs = [j for j in all_jobs if j.job_type == ProcessingJobType.EMBED]
                    active_embed = [j for j in embed_jobs if j.status in (ProcessingJobStatus.PENDING, ProcessingJobStatus.RUNNING)]
                    embed_job = active_embed[-1] if active_embed else (embed_jobs[-1] if embed_jobs else None)

                    if not embed_job or embed_job.status in (ProcessingJobStatus.COMPLETED, ProcessingJobStatus.FAILED):
                        embed_job = await job_service.create_job(document_version_id=version.id, job_type=ProcessingJobType.EMBED)
                        await session.commit()

                    if embed_job.status != ProcessingJobStatus.COMPLETED:
                        worker = EmbeddingWorker(session)
                        try:
                            await worker.process_job(embed_job.id)
                            await session.commit()
                        except Exception:
                            await session.rollback()
                            await doc_service.update(document_id, status=DocumentStatus.FAILED)
                            await session.commit()
                            raise

                # 3. Finalize
                await version_service.update(version.id, status=DocumentVersionStatus.COMPLETED)
                await doc_service.update(document_id, status=DocumentStatus.READY)
                await session.commit()
                logger.info("[BACKGROUND RUNNER] Completed ingestion pipeline for document %s", document_id)

        except Exception as exc:
            logger.warning("[BACKGROUND RUNNER] Error ingesting document %s: %s", document_id, exc, exc_info=True)
            try:
                async with AsyncSessionLocal() as session:
                    doc_service = DocumentService(session)
                    await doc_service.update(document_id, status=DocumentStatus.FAILED)
                    await session.commit()
            except Exception:
                pass
        finally:
            _ACTIVE_DOCUMENT_IDS.discard(document_id)

    @staticmethod
    async def poll_and_process_pending() -> None:
        """Scan DB for documents needing processing or retries."""
        try:
            async with AsyncSessionLocal() as session:
                stmt = select(Document).where(
                    Document.deleted_at.is_(None),
                    Document.status.in_([DocumentStatus.UPLOADED, DocumentStatus.PROCESSING]),
                )
                docs = list((await session.execute(stmt)).scalars().all())
                pending_ids = [d.id for d in docs if d.id not in _ACTIVE_DOCUMENT_IDS]

            for doc_id in pending_ids:
                await BackgroundJobRunner._process_document_safely(doc_id)
        except Exception as exc:
            logger.warning("[BACKGROUND RUNNER] Poll error: %s", exc)

    @staticmethod
    async def start_loop(interval_seconds: float = 4.0) -> None:
        """Start the background polling worker loop."""
        logger.info("[BACKGROUND RUNNER] Starting background worker loop (interval=%.1fs)", interval_seconds)
        while True:
            try:
                await BackgroundJobRunner.poll_and_process_pending()
            except asyncio.CancelledError:
                logger.info("[BACKGROUND RUNNER] Worker loop cancelled")
                break
            except Exception as exc:
                logger.warning("[BACKGROUND RUNNER] Worker loop exception: %s", exc)
            await asyncio.sleep(interval_seconds)


def start_background_runner() -> None:
    """Start background runner task if not already running."""
    global _RUNNER_TASK
    if _RUNNER_TASK is None or _RUNNER_TASK.done():
        _RUNNER_TASK = asyncio.create_task(BackgroundJobRunner.start_loop())
