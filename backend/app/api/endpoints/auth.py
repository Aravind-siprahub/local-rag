"""Authentication endpoints for issuing and verifying JWT tokens."""
from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from app.api.dependencies import get_current_user
from app.api.security import create_access_token, verify_password, hash_password
from app.db.session import get_db
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserResponse
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("app.api.auth")

router = APIRouter(prefix="/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    email: str | None = None
    user_id: uuid.UUID | None = None
    password: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Issue JWT access token",
    description="Authenticates user credentials and returns a cryptographically signed HS256 JWT access token.",
)
@router.post(
    "/login",
    response_model=TokenResponse,
    summary="User login",
    description="Authenticates user with email and password, returning access token upon successful verification.",
)
async def login(
    payload: LoginRequest,
    session: AsyncSession = Depends(get_db),
) -> TokenResponse:
    if not payload.password or not payload.password.strip():
        logger.warning("[AUTH LOGIN FAILED] Missing password.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed: Password is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_repo = UserRepository(session)
    user: User | None = None

    if payload.user_id:
        user = await user_repo.get(payload.user_id)
    elif payload.email:
        clean_email = payload.email.strip()
        user = await user_repo.get_by_email(clean_email)

    if not user:
        logger.warning(f"[AUTH LOGIN FAILED] No active user account found for email='{payload.email}'.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed: Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active or user.deleted_at is not None:
        logger.warning(f"[AUTH LOGIN FAILED] Account is inactive or soft-deleted for email='{payload.email}'.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed: Account is inactive.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(payload.password, user.hashed_password):
        logger.warning(f"[AUTH LOGIN FAILED] Password hash mismatch for user id='{user.id}', email='{user.email}'.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed: Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Auto-upgrade legacy password hash to PBKDF2 upon successful verification
    if user.hashed_password.startswith("INSECURE-sha256$"):
        user.hashed_password = hash_password(payload.password)
        session.add(user)
        await session.commit()

    token = create_access_token(user.id)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse.model_validate(user),
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get authenticated current user profile",
)
async def get_me(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    return UserResponse.model_validate(current_user)
