"""Chat session endpoints."""
import uuid

from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import PaginationParams, get_chat_session_service
from app.schemas.chat_session import ChatSessionCreate, ChatSessionListResponse, ChatSessionResponse, ChatSessionUpdate
from app.services.chat_session_service import ChatSessionService

router = APIRouter(prefix="/chat-sessions", tags=["Chat Sessions"])


@router.post(
    "", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED, summary="Create a chat session"
)
async def create_chat_session(
    payload: ChatSessionCreate, service: ChatSessionService = Depends(get_chat_session_service)
) -> ChatSessionResponse:
    chat_session = await service.create_session(user_id=payload.user_id, title=payload.title)
    return ChatSessionResponse.model_validate(chat_session)


@router.get("", response_model=ChatSessionListResponse, summary="List a user's chat sessions")
async def list_chat_sessions(
    user_id: uuid.UUID | None = Query(
        default=None,
        description=(
            "Owner of the chat sessions. Omit to list sessions for the first "
            "active user (Swagger-friendly). Get ids from GET /users."
        ),
        examples=["dffcb114-052e-4a3c-ad02-de94753f875d"],
    ),
    include_archived: bool = Query(default=False),
    pagination: PaginationParams = Depends(),
    service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatSessionListResponse:
    if user_id is None:
        from app.repositories.user_repository import UserRepository
        from app.services.exceptions import ValidationError

        users = await UserRepository(service.session).list_active(limit=1)
        if not users:
            raise ValidationError("No users exist yet. Create one via POST /users first.")
        user_id = users[0].id

    sessions = await service.list_by_user(
        user_id, include_archived=include_archived, limit=pagination.limit, offset=pagination.offset
    )
    return ChatSessionListResponse(
        items=[ChatSessionResponse.model_validate(s) for s in sessions],
        total=len(sessions),  # filtered count unavailable without modifying the repository layer
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/{session_id}", response_model=ChatSessionResponse, summary="Get a chat session by id")
async def get_chat_session(
    session_id: uuid.UUID, service: ChatSessionService = Depends(get_chat_session_service)
) -> ChatSessionResponse:
    chat_session = await service.get(session_id)
    return ChatSessionResponse.model_validate(chat_session)


@router.patch("/{session_id}", response_model=ChatSessionResponse, summary="Update a chat session")
async def update_chat_session(
    session_id: uuid.UUID,
    payload: ChatSessionUpdate,
    service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatSessionResponse:
    updates = payload.model_dump(exclude_unset=True)
    chat_session = await service.update(session_id, **updates)
    return ChatSessionResponse.model_validate(chat_session)


@router.post("/{session_id}/archive", response_model=ChatSessionResponse, summary="Archive a chat session")
async def archive_chat_session(
    session_id: uuid.UUID, service: ChatSessionService = Depends(get_chat_session_service)
) -> ChatSessionResponse:
    chat_session = await service.archive(session_id)
    return ChatSessionResponse.model_validate(chat_session)


@router.post("/{session_id}/unarchive", response_model=ChatSessionResponse, summary="Unarchive a chat session")
async def unarchive_chat_session(
    session_id: uuid.UUID, service: ChatSessionService = Depends(get_chat_session_service)
) -> ChatSessionResponse:
    chat_session = await service.unarchive(session_id)
    return ChatSessionResponse.model_validate(chat_session)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Soft-delete a chat session",
    description="Sets deleted_at; the transcript (messages/citations) is preserved.",
)
async def delete_chat_session(
    session_id: uuid.UUID, service: ChatSessionService = Depends(get_chat_session_service)
) -> None:
    await service.delete(session_id)
