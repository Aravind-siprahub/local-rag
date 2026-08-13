"""S3-compatible Storage Service using Supabase's S3 endpoint.

Uses boto3 with the Supabase S3-compatible API:
  endpoint:   https://<ref>.storage.supabase.co/storage/v1/s3
  access_key: S3_ACCESS_KEY  (from Supabase Storage → S3 access keys)
  secret_key: S3_SECRET_KEY
  bucket:     SUPABASE_BUCKET (default: 'documents')

This is more reliable than the REST API because:
 - boto3 handles authentication (Signature V4) automatically.
 - Handles path encoding, retries, and chunked uploads natively.
 - Direct S3 semantics — no false-positive existence checks.
"""
from __future__ import annotations

import hashlib
import logging

from app.core.config import get_settings
from app.storage.base import SavedFile

logger = logging.getLogger(__name__)


class S3StorageError(Exception):
    """Base exception for S3 Storage failures."""


class S3StorageService:
    """Async-friendly S3 storage service backed by Supabase's S3 endpoint.

    boto3 is synchronous; we run it in a thread-pool executor via
    `asyncio.to_thread` so the FastAPI event loop is never blocked.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.endpoint_url = settings.S3_ENDPOINT
        self.region = settings.S3_REGION
        self.access_key = settings.S3_ACCESS_KEY
        self.secret_key = settings.S3_SECRET_KEY
        self.bucket_name = settings.SUPABASE_BUCKET

    @property
    def is_configured(self) -> bool:
        return bool(self.endpoint_url and self.access_key and self.secret_key)

    def _client(self):  # type: ignore[return]
        """Create a fresh boto3 S3 client (thread-safe, cheap to construct)."""
        import boto3
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
        )

    # ------------------------------------------------------------------
    # Bucket lifecycle
    # ------------------------------------------------------------------

    def _ensure_bucket_sync(self) -> None:
        """Create the bucket if it doesn't exist (sync, run inside executor)."""
        client = self._client()
        try:
            client.head_bucket(Bucket=self.bucket_name)
            logger.info("S3 bucket '%s' exists.", self.bucket_name)
        except Exception:
            try:
                client.create_bucket(Bucket=self.bucket_name)
                logger.info("S3 bucket '%s' created.", self.bucket_name)
            except Exception as exc:
                logger.warning("Could not create bucket '%s': %s", self.bucket_name, exc)

    async def ensure_bucket_exists(self) -> None:
        import asyncio
        await asyncio.to_thread(self._ensure_bucket_sync)

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def _upload_sync(self, *, content: bytes, storage_path: str, mime_type: str) -> None:
        """Upload bytes to S3 (sync, run inside executor)."""
        client = self._client()
        clean_key = storage_path.lstrip("/")
        logger.info(
            "S3 UPLOAD: bucket=%s key=%r size=%d mime=%s",
            self.bucket_name, clean_key, len(content), mime_type,
        )
        client.put_object(
            Bucket=self.bucket_name,
            Key=clean_key,
            Body=content,
            ContentType=mime_type,
        )
        logger.info("S3 UPLOAD COMPLETE: bucket=%s key=%r", self.bucket_name, clean_key)

    async def upload_file(
        self,
        *,
        content: bytes,
        storage_path: str,
        mime_type: str = "application/octet-stream",
    ) -> SavedFile:
        """Upload raw file content to Supabase S3 bucket."""
        if not self.is_configured:
            logger.warning("S3 credentials not configured — using mock SavedFile.")
            checksum = hashlib.sha256(content).hexdigest()
            return SavedFile(
                storage_key=storage_path.lstrip("/"),
                size_bytes=len(content),
                checksum_sha256=checksum,
            )

        import asyncio

        await self.ensure_bucket_exists()
        clean_key = storage_path.strip().lstrip("/")

        try:
            await asyncio.to_thread(
                self._upload_sync,
                content=content,
                storage_path=clean_key,
                mime_type=mime_type,
            )
        except Exception as exc:
            logger.error("S3 UPLOAD ERROR: bucket=%s key=%r error=%s", self.bucket_name, clean_key, exc)
            raise S3StorageError(f"S3 upload failed for key {clean_key!r}: {exc}") from exc

        checksum = hashlib.sha256(content).hexdigest()
        return SavedFile(
            storage_key=clean_key,
            size_bytes=len(content),
            checksum_sha256=checksum,
        )

    async def save(self, *, content: bytes, original_filename: str) -> SavedFile:
        """Satisfy FileStorage protocol by delegating to upload_file."""
        return await self.upload_file(content=content, storage_path=original_filename)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download_sync(self, *, storage_path: str) -> bytes:
        """Download bytes from S3 (sync, run inside executor)."""
        client = self._client()
        clean_key = storage_path.lstrip("/")
        logger.info("S3 DOWNLOAD: bucket=%s key=%r", self.bucket_name, clean_key)
        try:
            response = client.get_object(Bucket=self.bucket_name, Key=clean_key)
            data = response["Body"].read()
            logger.info("S3 DOWNLOAD COMPLETE: bucket=%s key=%r bytes=%d", self.bucket_name, clean_key, len(data))
            return data
        except Exception as exc:
            logger.error("S3 DOWNLOAD ERROR: bucket=%s key=%r error=%s", self.bucket_name, clean_key, exc)
            raise S3StorageError(f"S3 download failed for key {clean_key!r}: {exc}") from exc

    async def download_file(self, *, storage_path: str) -> bytes:
        """Download raw file bytes from Supabase S3 bucket."""
        if not self.is_configured:
            raise S3StorageError("S3 credentials not configured.")

        import asyncio
        return await asyncio.to_thread(self._download_sync, storage_path=storage_path)

    # ------------------------------------------------------------------
    # Exists check
    # ------------------------------------------------------------------

    def _exists_sync(self, *, storage_path: str) -> bool:
        client = self._client()
        clean_key = storage_path.lstrip("/")
        try:
            client.head_object(Bucket=self.bucket_name, Key=clean_key)
            return True
        except Exception:
            return False

    async def exists_file(self, *, storage_path: str) -> bool:
        if not self.is_configured:
            return False
        import asyncio
        return await asyncio.to_thread(self._exists_sync, storage_path=storage_path)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def _delete_sync(self, *, storage_path: str) -> None:
        client = self._client()
        clean_key = storage_path.lstrip("/")
        client.delete_object(Bucket=self.bucket_name, Key=clean_key)
        logger.info("S3 DELETE: bucket=%s key=%r", self.bucket_name, clean_key)

    async def delete_file(self, *, storage_path: str) -> None:
        if not self.is_configured:
            return
        import asyncio
        await asyncio.to_thread(self._delete_sync, storage_path=storage_path)
