"""Schemas for `app.models.user.User`."""
import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.enums import UserRole
from app.schemas.common import ORMModel, PaginatedResponse, TimestampSchema
from app.schemas.openapi_examples import (
    EMAIL_EXAMPLE,
    FULL_NAME_EXAMPLE,
    USER_CREATE_OPENAPI_EXAMPLE,
)
from app.schemas.validators.password import PasswordField


class UserBase(BaseModel):
    """Fields common to creating and reading a user."""

    email: Annotated[
        EmailStr,
        Field(examples=[EMAIL_EXAMPLE], json_schema_extra={"example": EMAIL_EXAMPLE}),
    ]
    full_name: Annotated[
        str | None,
        Field(
            default=None,
            max_length=255,
            examples=[FULL_NAME_EXAMPLE],
            json_schema_extra={"example": FULL_NAME_EXAMPLE},
        ),
    ]


class UserCreate(UserBase):
    password: PasswordField
    role: UserRole = UserRole.MEMBER

    model_config = ConfigDict(json_schema_extra={"example": USER_CREATE_OPENAPI_EXAMPLE})


class UserUpdate(BaseModel):
    """All fields optional — partial update. Password changes are
    deliberately excluded here; that belongs behind a dedicated
    auth/security flow, not a generic field update.
    """

    email: EmailStr | None = None
    full_name: Annotated[str | None, Field(default=None, max_length=255)]
    role: UserRole | None = None
    is_active: bool | None = None
    is_verified: bool | None = None


class UserResponse(UserBase, TimestampSchema, ORMModel):
    id: uuid.UUID
    role: UserRole
    is_active: bool
    is_verified: bool
    last_login_at: datetime | None = None


UserListResponse = PaginatedResponse[UserResponse]
