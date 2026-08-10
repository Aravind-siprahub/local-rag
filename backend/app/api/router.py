"""Top-level API router — aggregates all endpoint routers.

New endpoint modules are added here, not registered individually in
`app.main`, so `main.py` stays a thin composition root.
"""
from fastapi import APIRouter

from app.api.endpoints import (
    admin,
    chat,
    chat_messages,
    chat_sessions,
    debug,
    document_versions,
    documents,
    health,
    metrics,
    processing_jobs,
    system_settings,
    users,
)

from app.schemas.upload import DocumentUploadResponse

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(metrics.router)
api_router.include_router(admin.router)
api_router.include_router(debug.router)
api_router.include_router(users.router)
api_router.include_router(documents.router)
api_router.add_api_route(
    "/upload",
    endpoint=documents.upload_document,
    methods=["POST"],
    response_model=DocumentUploadResponse,
    status_code=201,
    summary="Upload a document",
    tags=["Documents"],
)
api_router.include_router(document_versions.router)
api_router.include_router(chat_sessions.router)
api_router.include_router(chat_messages.router)
api_router.include_router(chat.router)
api_router.include_router(processing_jobs.router)
api_router.include_router(system_settings.router)
