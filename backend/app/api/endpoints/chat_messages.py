"""Chat message endpoints.

No PATCH: messages are append-only (see `ChatMessageService`'s module
docstring) — correcting a message means posting a new one.
"""
import uuid

from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import PaginationParams, get_chat_message_service
from app.schemas.chat_message import ChatMessageCreate, ChatMessageListResponse, ChatMessageResponse
from app.services.chat_message_service import ChatMessageService

router = APIRouter(prefix="/chat-messages", tags=["Chat Messages"])


@router.post(
    "",
    response_model=ChatMessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Post a message to a chat session",
    description="Also bumps the parent session's last_message_at in the same transaction.",
)
async def create_chat_message(
    payload: ChatMessageCreate, service: ChatMessageService = Depends(get_chat_message_service)
) -> ChatMessageResponse:
    message = await service.create_message(
        session_id=payload.session_id,
        role=payload.role,
        content=payload.content,
        model_used=payload.model_used,
        prompt_tokens=payload.prompt_tokens,
        completion_tokens=payload.completion_tokens,
        latency_ms=payload.latency_ms,
        generation_time_ms=payload.generation_time_ms,
    )
    return ChatMessageResponse.model_validate(message)


@router.get("", response_model=ChatMessageListResponse, summary="List a chat session's messages")
async def list_chat_messages(
    session_id: uuid.UUID = Query(..., description="Parent chat session id."),
    pagination: PaginationParams = Depends(),
    service: ChatMessageService = Depends(get_chat_message_service),
) -> ChatMessageListResponse:
    messages = await service.list_by_session(session_id, limit=pagination.limit, offset=pagination.offset)
    return ChatMessageListResponse(
        items=[ChatMessageResponse.model_validate(m) for m in messages],
        total=len(messages),  # filtered count unavailable without modifying the repository layer
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/{message_id}", response_model=ChatMessageResponse, summary="Get a chat message by id")
async def get_chat_message(
    message_id: uuid.UUID, service: ChatMessageService = Depends(get_chat_message_service)
) -> ChatMessageResponse:
    message = await service.get(message_id)
    return ChatMessageResponse.model_validate(message)


@router.delete(
    "/{message_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a chat message",
    description="Hard delete — use sparingly; messages are designed to be append-only.",
)
async def delete_chat_message(
    message_id: uuid.UUID, service: ChatMessageService = Depends(get_chat_message_service)
) -> None:
    await service.delete(message_id)
