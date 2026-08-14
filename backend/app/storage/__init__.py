from app.core.config import get_settings
from app.storage.base import FileStorage, SavedFile
from app.storage.local_file_storage import LocalFileStorage
from app.storage.s3_storage_service import S3StorageService
from app.storage.supabase_storage_service import SupabaseStorageService


def get_storage_service(bucket_name: str | None = None):
    """Return configured S3StorageService if S3 is configured, else SupabaseStorageService."""
    settings = get_settings()
    if settings.s3_is_configured:
        return S3StorageService(bucket_name=bucket_name)
    return SupabaseStorageService(bucket_name=bucket_name)


__all__ = [
    "FileStorage",
    "SavedFile",
    "LocalFileStorage",
    "S3StorageService",
    "SupabaseStorageService",
    "get_storage_service",
]
