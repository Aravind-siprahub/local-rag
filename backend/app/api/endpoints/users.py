"""User endpoints."""
import hashlib
import uuid

from fastapi import APIRouter, Depends, status

from app.api.dependencies import PaginationParams, get_user_service
from app.schemas.user import UserCreate, UserListResponse, UserResponse, UserUpdate
from app.services.user_service import UserService

router = APIRouter(prefix="/users", tags=["Users"])


def _INSECURE_placeholder_password_hash(password: str) -> str:
    """NOT a real password hash — authentication is explicitly out of scope
    for this step (see task constraints). This exists only so `POST /users`
    can be exercised end-to-end without a hashing library. It has no salt
    and uses a fast general-purpose digest, which is unsuitable for
    passwords. Replace with passlib/bcrypt or argon2 before this endpoint
    is ever used against real user data.
    """
    return "INSECURE-sha256$" + hashlib.sha256(password.encode("utf-8")).hexdigest()


@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a user",
    description="Creates a user account. See a loud warning in this router's "
    "source about the password-hashing placeholder — real authentication is a later step.",
)
async def create_user(payload: UserCreate, service: UserService = Depends(get_user_service)) -> UserResponse:
    user = await service.create_user(
        email=payload.email,
        hashed_password=_INSECURE_placeholder_password_hash(payload.password),
        full_name=payload.full_name,
        role=payload.role,
    )
    return UserResponse.model_validate(user)


@router.get("", response_model=UserListResponse, summary="List users")
async def list_users(
    pagination: PaginationParams = Depends(), service: UserService = Depends(get_user_service)
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
async def get_user(user_id: uuid.UUID, service: UserService = Depends(get_user_service)) -> UserResponse:
    user = await service.get(user_id)
    return UserResponse.model_validate(user)


@router.patch("/{user_id}", response_model=UserResponse, summary="Update a user")
async def update_user(
    user_id: uuid.UUID, payload: UserUpdate, service: UserService = Depends(get_user_service)
) -> UserResponse:
    updates = payload.model_dump(exclude_unset=True)
    user = await service.update(user_id, **updates)
    return UserResponse.model_validate(user)


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Soft-delete a user",
    description="Sets deleted_at rather than removing the row; see UserService.delete.",
)
async def delete_user(user_id: uuid.UUID, service: UserService = Depends(get_user_service)) -> None:
    await service.delete(user_id)
