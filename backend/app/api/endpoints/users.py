"""User endpoints."""
import hashlib
import uuid

from fastapi import APIRouter, Depends, status

from app.api.dependencies import PaginationParams, get_current_user, get_user_service
from app.models.user import User
from app.schemas.user import UserCreate, UserListResponse, UserResponse, UserUpdate
from app.services.user_service import UserService

router = APIRouter(prefix="/users", tags=["Users"])


from app.api.security import hash_password


@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a user",
    description="Creates a new user account with secure PBKDF2-HMAC-SHA256 salted password hashing.",
)
async def create_user(payload: UserCreate, service: UserService = Depends(get_user_service)) -> UserResponse:
    user = await service.create_user(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        role=payload.role,
    )
    return UserResponse.model_validate(user)


@router.get("", response_model=UserListResponse, summary="List users")
async def list_users(
    pagination: PaginationParams = Depends(),
    current_user: User = Depends(get_current_user),
    service: UserService = Depends(get_user_service),
) -> UserListResponse:
    users = await service.list(limit=pagination.limit, offset=pagination.offset)
    total = await service.count()
    return UserListResponse(
        items=[UserResponse.model_validate(u) for u in users],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/{user_id}", response_model=UserResponse, summary="Get a user by id")
async def get_user(
    user_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: UserService = Depends(get_user_service),
) -> UserResponse:
    user = await service.get(user_id)
    return UserResponse.model_validate(user)


@router.patch("/{user_id}", response_model=UserResponse, summary="Update a user")
async def update_user(
    user_id: uuid.UUID,
    payload: UserUpdate,
    current_user: User = Depends(get_current_user),
    service: UserService = Depends(get_user_service),
) -> UserResponse:
    from app.api.security import verify_ownership
    verify_ownership(user_id, current_user, "user profile")
    updates = payload.model_dump(exclude_unset=True)
    user = await service.update(user_id, **updates)
    return UserResponse.model_validate(user)


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Soft-delete a user",
    description="Sets deleted_at rather than removing the row; see UserService.delete.",
)
async def delete_user(
    user_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: UserService = Depends(get_user_service),
) -> None:
    from app.api.security import verify_ownership
    verify_ownership(user_id, current_user, "user profile")
    await service.delete(user_id)
