"""End-to-end Document Ingestion Pipeline Service.

Traces and executes the full document processing lifecycle:
  [UPLOAD] -> [TEXT EXTRACTION] -> [CHUNKING] -> [EMBEDDINGS] -> [VECTOR INSERT] -> [STATUS UPDATE]

Logs structured diagnostics for every stage and guarantees status transitions
(Pending -> Processing -> Ready / Processed OR Failed).
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.embeddings.client import EmbeddingClientError, OllamaEmbeddingClient
from app.embeddings.generator import EmbeddingGenerator
from app.models.document import Document
from app.models.document_version import DocumentVersion
from app.models.embedding import Embedding
from app.models.enums import DocumentStatus, DocumentVersionStatus, ProcessingJobStatus, ProcessingJobType
from app.models.processing_job import ProcessingJob
from app.processing.parser import ParsingError
from app.services.chunker import chunk_document
from app.services.embedding import normalize_text_for_embedding
from app.services.parser import DocumentParser
from app.services.document_chunk_service import ChunkInput, DocumentChunkService
from app.services.document_service import DocumentService
from app.services.document_version_service import DocumentVersionService
from app.services.processing_job_service import ProcessingJobService
from app.storage.s3_storage_service import S3StorageError, S3StorageService
from app.storage.supabase_storage_service import SupabaseStorageError, SupabaseStorageService
from app.storage.local_file_storage import LocalFileStorage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    """Diagnostic outcome of an end-to-end document ingestion run."""

    document_id: uuid.UUID
    version_id: uuid.UUID
    job_id: uuid.UUID
    character_count: int
    chunk_count: int
    embedding_count: int
    vector_count: int
    total_duration_ms: int


class IngestionService:
    """Orchestrates end-to-end document ingestion with detailed stage logging."""

    def __init__(self, session: AsyncSession) -> None:
        settings = get_settings()
        self.session = session
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.document_parser = DocumentParser()
        self.documents = DocumentService(session)
        self.versions = DocumentVersionService(session)
        self.jobs = ProcessingJobService(session)
        self.chunks = DocumentChunkService(session)

    async def run_pipeline(self, document_id: uuid.UUID) -> IngestionResult:
        """Run all 6 ingestion stages for a document."""
        start_pipeline_mono = time.monotonic()

        document = await self.documents.get(document_id)
        if not document:
            raise ValueError(f"Document with id={document_id!r} not found.")

        # Find the latest document version
        version = None
        if hasattr(self.versions, "get_current_version"):
            version = await self.versions.get_current_version(document_id)
        if not version:
            versions = await self.versions.list_by_document(document_id)
            if not versions:
                raise ValueError(f"No DocumentVersion found for document_id={document_id!r}.")
            version = versions[-1]

        # Find or reuse existing processing job (prefer PENDING or RUNNING over creating duplicates)
        all_jobs = await self.jobs.list_by_document_version(version.id)
        active_jobs = [j for j in all_jobs if j.status in (ProcessingJobStatus.PENDING, ProcessingJobStatus.RUNNING)]
        job = active_jobs[-1] if active_jobs else (all_jobs[-1] if all_jobs else None)
        if not job:
            job = await self.jobs.create_job(document_version_id=version.id, job_type=ProcessingJobType.PARSE)

        version_id = version.id
        job_id = job.id if job else None

        current_stage = "[UPLOAD]"
        try:
            # ------------------------------------------------------------------
            # STAGE 1: [UPLOAD]
            # ------------------------------------------------------------------
            current_stage = "[UPLOAD]"
            stage_start = time.monotonic()
            logger.info("[UPLOAD] start: document_id=%s, version_id=%s", document_id, version.id)

            await self.documents.update(document_id, status=DocumentStatus.PROCESSING)
            if job.status == ProcessingJobStatus.PENDING:
                job = await self.jobs.start(job.id)

            # Use the best available storage backend (S3 > REST > local disk)
            settings = get_settings()
            if settings.s3_is_configured:
                storage_service = S3StorageService()
            else:
                storage_service = SupabaseStorageService()

            # ------------------------------------------------------------------
            # Resolve the correct storage path: try storage_path first (the
            # canonical Supabase path: user_id/doc_id/filename), then
            # storage_key as a secondary Supabase candidate, then local disk.
            # ------------------------------------------------------------------
            version_storage_path = (getattr(version, "storage_path", None) or "").strip().lstrip("/")
            version_storage_key = (getattr(version, "storage_key", None) or "").strip().lstrip("/")
            doc_storage_path = (getattr(document, "storage_path", None) or "").strip().lstrip("/")

            logger.info(
                "DB paths: version.storage_path=%r  version.storage_key=%r  document.storage_path=%r  "
                "document_id=%s  version_id=%s",
                version_storage_path, version_storage_key, doc_storage_path, document_id, version.id,
            )

            if not version_storage_path and not version_storage_key and not doc_storage_path:
                raise ValueError(
                    f"No storage path found in database for document_version_id={version.id!r}. "
                    "Both storage_path and storage_key are empty."
                )

            duration_ms = int((time.monotonic() - stage_start) * 1000)
            logger.info(
                "[UPLOAD] success: document_id=%s, storage_path=%r, duration=%dms",
                document_id, version_storage_path or version_storage_key, duration_ms,
            )

            # ------------------------------------------------------------------
            # STAGE 2: [TEXT EXTRACTION]
            # ------------------------------------------------------------------
            current_stage = "[TEXT EXTRACTION]"
            stage_start = time.monotonic()
            logger.info(
                "[TEXT EXTRACTION] start: document_id=%s, filename=%s, mime_type=%s",
                document_id, version.original_filename, version.mime_type,
            )

            await self.versions.update(version.id, status=DocumentVersionStatus.PARSING)

            raw_bytes: bytes | None = None

            if storage_service.is_configured:
                # Build candidate list: prefer explicit storage_path, then storage_key,
                # then document-level fallback — skip duplicates and blanks.
                remote_candidates: list[str] = []
                for candidate in (version_storage_path, version_storage_key, doc_storage_path):
                    if candidate and candidate not in remote_candidates:
                        remote_candidates.append(candidate)

                logger.info(
                    "DOWNLOAD candidates (Supabase bucket=%s): %s",
                    storage_service.bucket_name, remote_candidates,
                )

                for candidate_path in remote_candidates:
                    logger.info("DOWNLOAD attempt: bucket=%s path=%r", storage_service.bucket_name, candidate_path)
                    try:
                        exists = await storage_service.exists_file(storage_path=candidate_path)
                    except Exception as check_exc:
                        logger.warning("exists_file check failed for %r: %s", candidate_path, check_exc)
                        exists = False

                    if exists:
                        try:
                            raw_bytes = await storage_service.download_file(storage_path=candidate_path)
                            logger.info(
                                "DOWNLOAD success: bucket=%s path=%r bytes=%d",
                                storage_service.bucket_name, candidate_path, len(raw_bytes),
                            )
                            # Persist the working path so future runs go straight to it
                            if candidate_path != version_storage_path:
                                logger.info(
                                    "Updating version.storage_path from %r to working path %r",
                                    version_storage_path, candidate_path,
                                )
                                try:
                                    await self.versions.update(version.id, storage_path=candidate_path)
                                except Exception as upd_exc:
                                    logger.warning("Could not update storage_path on version: %s", upd_exc)
                            break
                        except Exception as dl_exc:
                            logger.warning("Download failed for %r: %s", candidate_path, dl_exc)
                            raw_bytes = None
                    else:
                        logger.warning("Object not found in Supabase at path %r", candidate_path)

                # If all remote candidates failed, try local disk as last resort
                if raw_bytes is None:
                    local_candidates = [self.upload_dir / p for p in remote_candidates if p]
                    # Also try the bare storage_key directly under upload_dir
                    if version_storage_key:
                        local_candidates.append(self.upload_dir / version_storage_key)

                    for local_path in local_candidates:
                        if local_path.is_file():
                            logger.warning(
                                "All Supabase paths failed; falling back to local disk: %s", local_path
                            )
                            raw_bytes = await asyncio.to_thread(local_path.read_bytes)
                            break

                if raw_bytes is None:
                    tried_paths = remote_candidates
                    err_msg = (
                        f"Supabase Storage download failed (400): "
                        f'{{"statusCode":"404","error":"not_found","message":"Object not found","code":"NoSuchKey"}} '
                        f"— tried paths {tried_paths!r} in bucket {storage_service.bucket_name!r}. "
                        "Check that the file was successfully uploaded to Supabase Storage."
                    )
                    logger.error(
                        "DOWNLOAD ERROR: bucket=%s tried_paths=%s detail=Object not found",
                        storage_service.bucket_name, tried_paths,
                    )
                    await self.versions.update(version.id, status=DocumentVersionStatus.FAILED, error_message=err_msg)
                    await self.documents.update(document_id, status=DocumentStatus.FAILED)
                    raise SupabaseStorageError(err_msg)

            else:
                # Supabase not configured — local-only mode (dev / offline)
                local_candidates = []
                for p in (version_storage_key, version_storage_path, doc_storage_path):
                    if p:
                        local_candidates.append(self.upload_dir / p)

                for local_path in local_candidates:
                    if local_path.is_file():
                        logger.info("LOCAL download: %s", local_path)
                        raw_bytes = await asyncio.to_thread(local_path.read_bytes)
                        break

                if raw_bytes is None:
                    raise FileNotFoundError(
                        f"Local file not found. Tried paths: "
                        f"{[str(p) for p in local_candidates]}"
                    )

            original_filename = getattr(version, "original_filename", None) or getattr(document, "original_filename", None) or getattr(document, "filename", None) or document.title
            mime_type = getattr(version, "mime_type", None) or getattr(document, "mime_type", None) or "application/octet-stream"

            parsed_doc = await self.document_parser.parse(
                raw_bytes, original_filename, document_id, mime_type
            )
            char_count = sum(len(block.text) for block in parsed_doc.blocks)

            if char_count == 0:
                raise ParsingError(f"Extracted 0 characters from {original_filename!r}.")

            now = datetime.now(timezone.utc)
            version_update: dict[str, Any] = {
                "status": DocumentVersionStatus.PARSED,
                "parsed_at": now,
            }
            if parsed_doc.page_count > 0:
                version_update["page_count"] = parsed_doc.page_count
            await self.versions.update(version.id, **version_update)

            duration_ms = int((time.monotonic() - stage_start) * 1000)
            logger.info(
                "[TEXT EXTRACTION] success: document_id=%s, characters=%d, duration=%dms",
                document_id,
                char_count,
                duration_ms,
            )

            # ------------------------------------------------------------------
            # STAGE 3: [CHUNKING]
            # ------------------------------------------------------------------
            current_stage = "[CHUNKING]"
            stage_start = time.monotonic()
            settings = get_settings()
            logger.info(
                "[CHUNKING] start: document_id=%s, max_tokens=%d, overlap=%d-%d, parser=%s",
                document_id,
                settings.SEMANTIC_CHUNK_MAX_TOKENS,
                settings.SEMANTIC_CHUNK_OVERLAP_MIN,
                settings.SEMANTIC_CHUNK_OVERLAP_MAX,
                parsed_doc.parser_used,
            )

            await self.versions.update(version.id, status=DocumentVersionStatus.CHUNKING)

            semantic_chunks = chunk_document(parsed_doc)
            if not semantic_chunks:
                raise ValueError(f"Chunking produced 0 chunks for document {document_id!r}.")

            chunk_inputs: list[ChunkInput] = [
                {
                    "chunk_index": sc.chunk_index,
                    "content": normalize_text_for_embedding(sc.text),
                    "content_tokens": sc.token_count,
                    "page_number": sc.page_number or None,
                    "section_title": sc.breadcrumb or sc.section or None,
                    "char_start": sc.char_start,
                    "char_end": sc.char_end,
                    "metadata_": sc.to_metadata_dict(),
                }
                for sc in semantic_chunks
            ]

            # Replace any old chunks for this version
            await self.chunks.create_chunks_for_version(version.id, chunk_inputs)
            now = datetime.now(timezone.utc)
            await self.versions.update(version.id, status=DocumentVersionStatus.CHUNKED, chunked_at=now)

            first_preview = semantic_chunks[0].text[:80].replace("\n", " ")
            duration_ms = int((time.monotonic() - stage_start) * 1000)
            logger.info(
                "[CHUNKING] success: document_id=%s, chunk_count=%d, duration=%dms, first_chunk_preview=%r",
                document_id,
                len(semantic_chunks),
                duration_ms,
                first_preview,
            )

            # ------------------------------------------------------------------
            # STAGE 4: [EMBEDDINGS]
            # ------------------------------------------------------------------
            current_stage = "[EMBEDDINGS]"
            stage_start = time.monotonic()
            settings = get_settings()
            logger.info(
                "[EMBEDDINGS] start: document_id=%s, model=%s, dim=%d",
                document_id,
                settings.EMBEDDING_MODEL,
                settings.EMBEDDING_DIMENSIONS,
            )

            await self.versions.update(version.id, status=DocumentVersionStatus.EMBEDDING)

            db_chunks = await self.chunks.list_by_document_version(version.id)
            if not db_chunks:
                raise ValueError(f"No DB chunks found for version {version.id!r}.")

            client = OllamaEmbeddingClient()
            generator = EmbeddingGenerator(self.session, client)
            try:
                gen_result = await generator.embed_chunks(db_chunks)
            finally:
                await client.close()

            if gen_result.embedded_count == 0 and gen_result.skipped_count == 0:
                raise EmbeddingClientError(f"Embedding generation failed: 0 embeddings generated for doc {document_id!r}.")

            now = datetime.now(timezone.utc)
            await self.versions.update(version.id, status=DocumentVersionStatus.EMBEDDED, embedded_at=now)

            duration_ms = int((time.monotonic() - stage_start) * 1000)
            logger.info(
                "[EMBEDDINGS] success: document_id=%s, generated=%d, skipped=%d, duration=%dms",
                document_id,
                gen_result.embedded_count,
                gen_result.skipped_count,
                duration_ms,
            )

            # ------------------------------------------------------------------
            # STAGE 5: [VECTOR INSERT]
            # ------------------------------------------------------------------
            current_stage = "[VECTOR INSERT]"
            stage_start = time.monotonic()
            logger.info("[VECTOR INSERT] start: document_id=%s", document_id)

            # Count total vector rows stored in database for this version
            chunk_ids = [c.id for c in db_chunks]
            stmt = select(func.count(Embedding.id)).where(Embedding.chunk_id.in_(chunk_ids))
            vector_count = (await self.session.execute(stmt)).scalar_one()

            if vector_count == 0:
                raise ValueError(
                    f"Vector storage verification failed: 0 vectors found in database for document {document_id!r}."
                )

            duration_ms = int((time.monotonic() - stage_start) * 1000)
            logger.info(
                "[VECTOR INSERT] success: document_id=%s, vectors_indexed=%d, duration=%dms",
                document_id,
                vector_count,
                duration_ms,
            )

            # ------------------------------------------------------------------
            # STAGE 6: [STATUS UPDATE]
            # ------------------------------------------------------------------
            current_stage = "[STATUS UPDATE]"
            stage_start = time.monotonic()

            await self.versions.update(version.id, status=DocumentVersionStatus.COMPLETED, error_message=None)
            await self.documents.update(document_id, status=DocumentStatus.READY, current_version_id=version.id)
            
            # Complete all active jobs for this version
            for j in all_jobs:
                j_latest = await self.jobs.get(j.id)
                if j_latest and j_latest.status == ProcessingJobStatus.RUNNING:
                    await self.jobs.complete(j.id)
                elif j_latest and j_latest.status == ProcessingJobStatus.PENDING:
                    await self.jobs.start(j.id)
                    await self.jobs.complete(j.id)

            total_duration_ms = int((time.monotonic() - start_pipeline_mono) * 1000)
            logger.info(
                "[STATUS UPDATE] success: document_id=%s, status=READY, total_duration=%dms",
                document_id,
                total_duration_ms,
            )

            return IngestionResult(
                document_id=document_id,
                version_id=version.id,
                job_id=job.id,
                character_count=char_count,
                chunk_count=len(semantic_chunks),
                embedding_count=gen_result.embedded_count + gen_result.skipped_count,
                vector_count=vector_count,
                total_duration_ms=total_duration_ms,
            )

        except Exception as exc:
            logger.exception("%s FAILED for document_id=%s: %s", current_stage, document_id, exc)
            await self._record_pipeline_failure(document_id, version_id, job_id, str(exc))
            raise

    async def _record_pipeline_failure(
        self,
        document_id: uuid.UUID,
        version_id: uuid.UUID | None,
        job_id: uuid.UUID | None,
        error_message: str,
    ) -> None:
        """Mark Document, DocumentVersion, and ProcessingJob as FAILED."""
        try:
            await self.session.rollback()
            await self.documents.update(document_id, status=DocumentStatus.FAILED)
            if version_id is not None:
                await self.versions.update(
                    version_id,
                    status=DocumentVersionStatus.FAILED,
                    error_message=error_message[:1000],
                )
            if job_id is not None:
                job = await self.jobs.get(job_id)
                if job and job.status in (ProcessingJobStatus.PENDING, ProcessingJobStatus.RUNNING):
                    await self.jobs.fail(job_id, error_message[:1000])
        except Exception:
            logger.exception("Failed to record failure state for document_id=%s", document_id)
