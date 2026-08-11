"""Resolve Swagger placeholder owner ids to a real active user."""
import logging
import uuid

from app.core.swagger_constants import OPENAPI_PLACEHOLDER_UUID
from app.repositories.user_repository import UserRepository
from app.services.exceptions import ValidationError

logger = logging.getLogger(__name__)


async def resolve_owner_user_id(user_id: uuid.UUID | None, users: UserRepository) -> uuid.UUID:
    """Return `user_id` unchanged, or the first active user when None or Swagger's
    placeholder UUID was submitted.
    """
    if user_id is not None and user_id != OPENAPI_PLACEHOLDER_UUID:
        return user_id

    active_users = await users.list_active(limit=1)
    if not active_users:
        raise ValidationError("No users exist yet. Create one via POST /users first.")

    resolved_id = active_users[0].id
    logger.info(
        "Resolved OpenAPI placeholder/None user_id to first active user %s",
        resolved_id,
    )
    return resolved_id
