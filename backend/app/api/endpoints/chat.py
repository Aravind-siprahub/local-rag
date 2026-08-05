"""Unified chat API — RAG Q&A and session transcript access."""
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import PaginationParams, get_chat_message_service, get_chat_session_service, get_rag_service
from app.core.swagger_constants import OPENAPI_PLACEHOLDER_UUID
from app.rag.response import RAGResponse
from app.rag.service import RAGError, RAGService
from app.repositories.chat_session_repository import ChatSessionRepository
from app.repositories.user_repository import UserRepository
from app.retrieval.search import SearchFilters
from app.schemas.chat import (
    ChatCitationResponse,
    ChatRequest,
    ChatResponse,
    ChatTokenUsageResponse,
)
from app.schemas.chat_message import ChatMessageListResponse, ChatMessageResponse
from app.schemas.chat_session import ChatSessionResponse
from app.services.chat_message_service import ChatMessageService
from app.services.chat_session_service import ChatSessionService
from app.services.session_resolution import get_or_create_swagger_demo_session

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    operation_id="chat_ask",
    summary="Ask a question using RAG",
    description=(
        "Runs retrieval → prompt building → LLM generation, persists the user "
        "and assistant messages, and stores citations for retrieved chunks.\n\n"
        "**Session id:** create one first via **POST /chat-sessions** (tag: "
        "Chat Sessions), then paste the returned `id` here. "
        "Or omit `session_id` to auto-use a demo session."
    ),
)
async def ask_chat(
    payload: ChatRequest,
    rag: RAGService = Depends(get_rag_service),
    session_service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatResponse:
    filters = SearchFilters(
        document_id=payload.document_id,
        document_version_id=payload.document_version_id,
    )

    session_id = payload.session_id
    if session_id is None or session_id == OPENAPI_PLACEHOLDER_UUID:
        session_id = await get_or_create_swagger_demo_session(
            users=UserRepository(session_service.session),
            sessions=ChatSessionRepository(session_service.session),
            session_service=session_service,
        )

    try:
        result = await rag.ask(
            session_id,
            payload.question,
            filters=filters,
            top_k=payload.top_k,
            similarity_threshold=payload.similarity_threshold,
        )
    except RAGError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    return _to_chat_response(result)


@router.get(
    "/sessions/{session_id}",
    response_model=ChatSessionResponse,
    operation_id="chat_get_session",
    summary="Get a chat session by id",
)
async def get_chat_session(
    session_id: uuid.UUID,
    service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatSessionResponse:
    chat_session = await service.get(session_id)
    return ChatSessionResponse.model_validate(chat_session)


@router.get(
    "/sessions/{session_id}/messages",
    response_model=ChatMessageListResponse,
    operation_id="chat_list_session_messages",
    summary="List messages in a chat session",
)
async def list_chat_session_messages(
    session_id: uuid.UUID,
    pagination: PaginationParams = Depends(),
    session_service: ChatSessionService = Depends(get_chat_session_service),
    message_service: ChatMessageService = Depends(get_chat_message_service),
) -> ChatMessageListResponse:
    await session_service.get(session_id)

    messages = await message_service.list_by_session(
        session_id, limit=pagination.limit, offset=pagination.offset
    )
    return ChatMessageListResponse(
        items=[ChatMessageResponse.model_validate(message) for message in messages],
        total=len(messages),
        limit=pagination.limit,
        offset=pagination.offset,
    )


def _to_chat_response(result: RAGResponse) -> ChatResponse:
    token_usage = None
    if result.token_usage is not None:
        token_usage = ChatTokenUsageResponse(
            prompt_tokens=result.token_usage.prompt_tokens,
            completion_tokens=result.token_usage.completion_tokens,
            total_tokens=result.token_usage.total_tokens,
        )

    return ChatResponse(
        answer=result.answer,
        citations=[
            ChatCitationResponse(
                chunk_id=source.chunk_id,
                chunk_text=source.chunk_text,
                document_id=source.document_id,
                document_version_id=source.document_version_id,
                similarity_score=source.similarity_score,
                rank=source.rank,
            )
            for source in result.sources
        ],
        token_usage=token_usage,
        model=result.model,
        processing_time_ms=result.processing_time_ms,
        user_message_id=result.user_message_id,
        assistant_message_id=result.assistant_message_id,
    )
