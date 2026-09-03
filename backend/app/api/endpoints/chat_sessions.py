"""Chat session endpoints."""
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import PaginationParams, get_current_user, get_chat_session_service
from app.api.security import verify_ownership
from app.models.user import User
from app.schemas.chat_session import ChatSessionCreate, ChatSessionListResponse, ChatSessionResponse, ChatSessionUpdate
from app.services.chat_session_service import ChatSessionService

router = APIRouter(prefix="/chat-sessions", tags=["Chat Sessions"])


@router.post(
    "", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED, summary="Create a chat session"
)
async def create_chat_session(
    payload: ChatSessionCreate,
    current_user: User = Depends(get_current_user),
    service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatSessionResponse:
    chat_session = await service.create_session(user_id=current_user.id, title=payload.title)
    return ChatSessionResponse.model_validate(chat_session)


@router.get("", response_model=ChatSessionListResponse, summary="List a user's chat sessions")
async def list_chat_sessions(
    user_id: str | None = Query(
        default=None,
        description="Owner of the chat sessions.",
    ),
    include_archived: bool = Query(default=False),
    pagination: PaginationParams = Depends(),
    current_user: User = Depends(get_current_user),
    service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatSessionListResponse:
    if user_id and user_id.strip() and user_id.strip().lower() not in ("undefined", "null", "none"):
        try:
            parsed_uuid = uuid.UUID(user_id.strip())
            if str(parsed_uuid) != str(current_user.id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: Cannot access chat sessions belonging to another user.",
                )
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid user_id UUID format: {user_id!r}",
            )

    sessions = await service.list_by_user(
        current_user.id, include_archived=include_archived, limit=pagination.limit, offset=pagination.offset
    )
    return ChatSessionListResponse(
        items=[ChatSessionResponse.model_validate(s) for s in sessions],
        total=len(sessions),
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/{session_id}", response_model=ChatSessionResponse, summary="Get a chat session by id")
async def get_chat_session(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatSessionResponse:
    chat_session = await service.get(session_id)
    if not chat_session:
        raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found.")
    verify_ownership(chat_session.user_id, current_user, "chat session")
    return ChatSessionResponse.model_validate(chat_session)


@router.patch("/{session_id}", response_model=ChatSessionResponse, summary="Update a chat session")
async def update_chat_session(
    session_id: uuid.UUID,
    payload: ChatSessionUpdate,
    current_user: User = Depends(get_current_user),
    service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatSessionResponse:
    chat_session = await service.get(session_id)
    if not chat_session:
        raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found.")
    verify_ownership(chat_session.user_id, current_user, "chat session")

    updates = payload.model_dump(exclude_unset=True)
    updated_session = await service.update(session_id, **updates)
    return ChatSessionResponse.model_validate(updated_session)


@router.post("/{session_id}/archive", response_model=ChatSessionResponse, summary="Archive a chat session")
async def archive_chat_session(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatSessionResponse:
    chat_session = await service.get(session_id)
    if not chat_session:
        raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found.")
    verify_ownership(chat_session.user_id, current_user, "chat session")

    updated_session = await service.archive(session_id)
    return ChatSessionResponse.model_validate(updated_session)


@router.post("/{session_id}/unarchive", response_model=ChatSessionResponse, summary="Unarchive a chat session")
async def unarchive_chat_session(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatSessionResponse:
    chat_session = await service.get(session_id)
    if not chat_session:
        raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found.")
    verify_ownership(chat_session.user_id, current_user, "chat session")

    updated_session = await service.unarchive(session_id)
    return ChatSessionResponse.model_validate(updated_session)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Soft-delete a chat session",
    description="Sets deleted_at; the transcript (messages/citations) is preserved.",
)
async def delete_chat_session(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: ChatSessionService = Depends(get_chat_session_service),
) -> None:
    chat_session = await service.get_optional(session_id)
    if not chat_session or chat_session.deleted_at is not None:
        return
    verify_ownership(chat_session.user_id, current_user, "chat session")

    await service.delete(session_id)
