"""Top-level API router — aggregates all endpoint routers.

New endpoint modules are added here, not registered individually in
`app.main`, so `main.py` stays a thin composition root.
"""
from fastapi import APIRouter

from app.api.endpoints import (
    chat,
    chat_messages,
    chat_sessions,
    document_versions,
    documents,
    health,
    processing_jobs,
    system_settings,
    users,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(users.router)
api_router.include_router(documents.router)
api_router.include_router(document_versions.router)
api_router.include_router(chat_sessions.router)
api_router.include_router(chat_messages.router)
api_router.include_router(chat.router)
api_router.include_router(processing_jobs.router)
api_router.include_router(system_settings.router)
