"""Resolve Swagger placeholder chat session ids to a real session."""
import logging
import uuid

from app.core.swagger_constants import OPENAPI_PLACEHOLDER_UUID
from app.repositories.chat_session_repository import ChatSessionRepository
from app.repositories.user_repository import UserRepository
from app.services.chat_session_service import ChatSessionService
from app.services.exceptions import NotFoundError, ValidationError

logger = logging.getLogger(__name__)

SWAGGER_DEMO_SESSION_TITLE = "Swagger demo chat"


async def get_or_create_swagger_demo_session(
    *,
    users: UserRepository,
    sessions: ChatSessionRepository,
    session_service: ChatSessionService,
    user_id: uuid.UUID | None = None,
) -> uuid.UUID:
    """Return an existing active session for the user, or create a demo one."""
    if user_id is None:
        from sqlalchemy import func, select
        from app.models.document import Document

        stmt = (
            select(Document.user_id)
            .where(Document.deleted_at.is_(None))
            .group_by(Document.user_id)
            .order_by(func.count(Document.id).desc())
            .limit(1)
        )
        owner_result = (await users.session.execute(stmt)).first()
        if owner_result:
            user_id = owner_result[0]
        else:
            active_users = await users.list_active(limit=100)
            if not active_users:
                raise ValidationError("No users exist yet. Create one via POST /users first.")
            user_id = active_users[0].id

    existing = await sessions.list_by_user(user_id, limit=1)
    if existing:
        return existing[0].id

    created = await session_service.create_session(
        user_id=user_id,
        title=SWAGGER_DEMO_SESSION_TITLE,
    )
    logger.info("Created Swagger demo chat session %s for user %s", created.id, user_id)
    return created.id


async def resolve_chat_session_id(
    session_id: uuid.UUID,
    *,
    users: UserRepository,
    sessions: ChatSessionRepository,
    session_service: ChatSessionService,
) -> uuid.UUID:
    """Return `session_id` if it exists; map Swagger's placeholder UUID to a demo session."""
    existing = await sessions.get(session_id)
    if existing is not None and existing.deleted_at is None:
        return session_id

    if session_id == OPENAPI_PLACEHOLDER_UUID:
        resolved = await get_or_create_swagger_demo_session(
            users=users,
            sessions=sessions,
            session_service=session_service,
        )
        logger.info(
            "Resolved OpenAPI placeholder session_id to demo session %s",
            resolved,
        )
        return resolved

    raise NotFoundError(
        f"ChatSession with id={session_id!r} was not found. "
        "Create one via POST /chat-sessions and use the returned id "
        "(Swagger Try-it-out should pre-fill a real session id after refreshing /docs)."
    )
