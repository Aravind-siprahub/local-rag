"""Production-Grade Supabase Storage Service.

Communicates with Supabase Storage REST API directly via `httpx.AsyncClient`
using `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, and `SUPABASE_BUCKET`.

Never reads or writes files from local disk in production. Streams downloads
memory-efficiently during background ingestion runs.
"""
from __future__ import annotations

import hashlib
import logging
from typing import AsyncGenerator
from urllib.parse import quote
import httpx

from app.core.config import get_settings
from app.storage.base import SavedFile

logger = logging.getLogger(__name__)


def _encode_storage_path(storage_path: str) -> str:
    """URL-encode each segment of a storage path, preserving '/' separators.

    Supabase Storage REST API expects the object key to be percent-encoded
    when it contains spaces or special characters (e.g. 'My File (1).pdf').
    Encoding only the segments (not the slashes) keeps the path hierarchy intact.
    """
    clean = storage_path.strip().lstrip("/")
    return "/".join(quote(seg, safe="") for seg in clean.split("/"))


class SupabaseStorageError(Exception):
    """Base exception for Supabase Storage failures."""


class SupabaseStorageService:
    """Async service interacting with Supabase Storage REST API."""

    def __init__(
        self,
        supabase_url: str | None = None,
        service_role_key: str | None = None,
        bucket_name: str | None = None,
    ) -> None:
        settings = get_settings()
        self.supabase_url = (supabase_url or settings.SUPABASE_URL or "").rstrip("/")
        self.service_role_key = service_role_key or settings.SUPABASE_SERVICE_ROLE_KEY or ""
        self.bucket_name = bucket_name or settings.SUPABASE_BUCKET or "documents"

    @property
    def is_configured(self) -> bool:
        """Check whether valid Supabase credentials are configured."""
        return bool(self.supabase_url and self.service_role_key)

    def _headers(self, content_type: str | None = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.service_role_key}",
            "apiKey": self.service_role_key,
        }
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    async def ensure_bucket_exists(self) -> None:
        """Ensure the specified bucket exists in Supabase Storage."""
        if not self.is_configured:
            return

        endpoint = f"{self.supabase_url}/storage/v1/bucket"
        headers = self._headers("application/json")
        payload = {"id": self.bucket_name, "name": self.bucket_name, "public": True}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(endpoint, headers=headers, json=payload)
                if res.status_code in (200, 201, 409):
                    logger.info("Supabase storage bucket '%s' ready.", self.bucket_name)
        except Exception as exc:
            logger.warning("Failed to check/create Supabase storage bucket '%s': %s", self.bucket_name, exc)

    async def exists_file(self, *, storage_path: str) -> bool:
        """Check whether an object exists in Supabase Storage bucket.

        Uses only HEAD (not GET) to avoid false positives from Supabase
        returning 200 with bucket metadata when a GET is made against
        a path that resolves to a listing rather than an object.
        """
        if not self.is_configured:
            return False

        clean_path = _encode_storage_path(storage_path)
        endpoints = [
            f"{self.supabase_url}/storage/v1/object/authenticated/{self.bucket_name}/{clean_path}",
            f"{self.supabase_url}/storage/v1/object/{self.bucket_name}/{clean_path}",
            f"{self.supabase_url}/storage/v1/object/public/{self.bucket_name}/{clean_path}",
        ]
        headers = self._headers()

        async with httpx.AsyncClient(timeout=10.0) as client:
            for endpoint in endpoints:
                try:
                    res = await client.head(endpoint, headers=headers)
                    logger.debug("exists_file HEAD %s -> %d", endpoint, res.status_code)
                    if res.status_code == 200:
                        return True
                    if res.status_code == 404:
                        # Definitively not found at this endpoint, try next variant
                        continue
                    # For other status codes (401, 403, etc.), also try next variant
                except Exception as exc:
                    logger.debug("exists_file HEAD %s -> exception: %s", endpoint, exc)

        return False

    async def upload_file(
        self,
        *,
        content: bytes,
        storage_path: str,
        mime_type: str = "application/octet-stream",
    ) -> SavedFile:
        """Upload raw file content to Supabase Storage bucket."""
        if not self.is_configured:
            logger.warning(
                "[STORAGE UPLOAD] SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not configured. "
                "Falling back to mock storage key for local offline testing."
            )
            checksum = hashlib.sha256(content).hexdigest()
            return SavedFile(
                storage_key=storage_path,
                size_bytes=len(content),
                checksum_sha256=checksum,
            )

        await self.ensure_bucket_exists()

        # raw_path is stored in DB; encoded_path is used in HTTP requests
        raw_path = storage_path.strip().lstrip("/")
        encoded_path = _encode_storage_path(storage_path)
        base_endpoint = f"{self.supabase_url}/storage/v1/object/{self.bucket_name}/{encoded_path}"
        headers = self._headers(content_type=mime_type)
        headers["x-upsert"] = "true"  # Allow overwrite/upsert

        logger.info(
            "UPLOAD: bucket=%s path=%r (encoded=%r) size=%d mime_type=%s",
            self.bucket_name,
            raw_path,
            encoded_path,
            len(content),
            mime_type,
        )

        response = None
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Try POST first (new object), fall back to PUT (update existing)
            for method in ("POST", "PUT"):
                response = await client.request(method, base_endpoint, headers=headers, content=content)
                logger.info(
                    "UPLOAD %s: bucket=%s path=%r status=%d body=%s",
                    method, self.bucket_name, encoded_path, response.status_code, response.text[:300],
                )
                if response.status_code in (200, 201):
                    break
                if method == "POST" and response.status_code in (400, 409, 422):
                    # Supabase may reject POST for an existing key; retry with PUT
                    logger.warning(
                        "UPLOAD POST returned %d for %r, retrying with PUT",
                        response.status_code, encoded_path,
                    )
                    continue
                # Any other error on POST — break and report
                break

        if response is None or response.status_code not in (200, 201):
            status = response.status_code if response is not None else 0
            body = response.text if response is not None else "no response"
            logger.error(
                "UPLOAD ERROR: bucket=%s path=%r encoded=%r status=%d detail=%s",
                self.bucket_name, raw_path, encoded_path, status, body,
            )
            raise SupabaseStorageError(
                f"Supabase Storage upload failed ({status}): {body}"
            )

        # Verify the object is immediately downloadable after upload
        exists = await self.exists_file(storage_path=raw_path)
        if not exists:
            logger.error(
                "UPLOAD VERIFY FAILED: bucket=%s path=%r — object not found right after upload",
                self.bucket_name, raw_path,
            )
            raise SupabaseStorageError(
                f"Supabase Storage upload verification failed: Object not found at {raw_path!r} "
                f"immediately after upload. Check bucket permissions and RLS policies."
            )

        checksum = hashlib.sha256(content).hexdigest()
        logger.info(
            "UPLOAD COMPLETE: bucket=%s path=%r checksum=%s",
            self.bucket_name, raw_path, checksum,
        )

        return SavedFile(
            storage_key=raw_path,
            size_bytes=len(content),
            checksum_sha256=checksum,
        )

    async def download_file(self, *, storage_path: str) -> bytes:
        """Download raw file bytes from Supabase Storage bucket."""
        if not self.is_configured:
            raise SupabaseStorageError("Supabase Storage credentials not configured.")

        encoded_path = _encode_storage_path(storage_path)
        # Try all known Supabase download endpoint variants in order.
        # - /object/authenticated requires the service role Bearer token.
        # - /object/{bucket}/{path} is the direct private-bucket endpoint.
        # - /object/public is only for public buckets.
        endpoints = [
            f"{self.supabase_url}/storage/v1/object/authenticated/{self.bucket_name}/{encoded_path}",
            f"{self.supabase_url}/storage/v1/object/{self.bucket_name}/{encoded_path}",
            f"{self.supabase_url}/storage/v1/object/public/{self.bucket_name}/{encoded_path}",
        ]
        headers = self._headers()

        logger.info("DOWNLOAD: bucket=%s path=%r encoded=%r", self.bucket_name, storage_path, encoded_path)

        last_error = ""
        async with httpx.AsyncClient(timeout=120.0) as client:
            for endpoint in endpoints:
                response = await client.get(endpoint, headers=headers)
                logger.info(
                    "DOWNLOAD attempt: %s -> status=%d",
                    endpoint, response.status_code,
                )
                if response.status_code == 200:
                    logger.info(
                        "DOWNLOAD COMPLETE: bucket=%s path=%r bytes=%d via %s",
                        self.bucket_name, encoded_path, len(response.content), endpoint,
                    )
                    return response.content
                last_error = f"({response.status_code}): {response.text[:200]}"
                logger.warning("DOWNLOAD fail on %s: %s", endpoint, last_error)

        logger.error(
            "DOWNLOAD ERROR: bucket=%s path=%r encoded=%r all_endpoints_failed last_error=%s",
            self.bucket_name, storage_path, encoded_path, last_error,

        )
        raise SupabaseStorageError(
            f"Supabase Storage download failed {last_error}"
        )

    async def stream_download_file(
        self, *, storage_path: str, chunk_size: int = 65536
    ) -> AsyncGenerator[bytes, None]:
        """Stream download file bytes directly from Supabase Storage without loading full file into RAM."""
        if not self.is_configured:
            raise SupabaseStorageError("Supabase Storage credentials not configured.")

        clean_path = storage_path.lstrip("/")
        endpoints = [
            f"{self.supabase_url}/storage/v1/object/authenticated/{self.bucket_name}/{clean_path}",
            f"{self.supabase_url}/storage/v1/object/{self.bucket_name}/{clean_path}",
            f"{self.supabase_url}/storage/v1/object/public/{self.bucket_name}/{clean_path}",
        ]
        headers = self._headers()

        logger.info(
            "[STORAGE STREAM DOWNLOAD] start: bucket=%s, storage_path=%s",
            self.bucket_name,
            storage_path,
        )

        async with httpx.AsyncClient(timeout=300.0) as client:
            for endpoint in endpoints:
                async with client.stream("GET", endpoint, headers=headers) as response:
                    if response.status_code == 200:
                        async for chunk in response.aiter_bytes(chunk_size):
                            yield chunk
                        logger.info(
                            "[STORAGE STREAM DOWNLOAD] complete: bucket=%s, storage_path=%s via %s",
                            self.bucket_name,
                            storage_path,
                            endpoint,
                        )
                        return
            
            raise SupabaseStorageError(f"Supabase Storage streaming download failed for {storage_path}")

    async def delete_file(self, *, storage_path: str) -> bool:
        """Delete an object from Supabase Storage bucket."""
        if not self.is_configured:
            logger.warning("[STORAGE DELETE] Supabase credentials not set, skipping remote delete for %s", storage_path)
            return True

        endpoint = f"{self.supabase_url}/storage/v1/object/{self.bucket_name}"
        headers = self._headers("application/json")
        payload = {"prefixes": [storage_path]}

        logger.info(
            "[STORAGE DELETE] start: bucket=%s, storage_path=%s",
            self.bucket_name,
            storage_path,
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request("DELETE", endpoint, headers=headers, json=payload)

        if response.status_code not in (200, 204):
            logger.error(
                "[STORAGE DELETE] error: bucket=%s, storage_path=%s, status=%d, detail=%s",
                self.bucket_name,
                storage_path,
                response.status_code,
                response.text,
            )
            return False

        logger.info(
            "[STORAGE DELETE] complete: bucket=%s, storage_path=%s",
            self.bucket_name,
            storage_path,
        )
        return True

    async def generate_signed_url(self, *, storage_path: str, expires_in: int = 3600) -> str:
        """Generate a short-lived signed URL for secure document access."""
        if not self.is_configured:
            raise SupabaseStorageError("Supabase Storage credentials not configured.")

        endpoint = f"{self.supabase_url}/storage/v1/object/sign/{self.bucket_name}/{storage_path}"
        headers = self._headers("application/json")
        payload = {"expiresIn": expires_in}

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(endpoint, headers=headers, json=payload)

        if response.status_code != 200:
            raise SupabaseStorageError(
                f"Failed to generate signed URL ({response.status_code}): {response.text}"
            )

        data = response.json()
        signed_path = data.get("signedURL")
        if not signed_path:
            raise SupabaseStorageError("Signed URL response missing signedURL field.")

        return f"{self.supabase_url}/storage/v1{signed_path}"
