import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status, Request

logger = logging.getLogger(__name__)

from app.api.dependencies import PaginationParams, get_chat_message_service, get_chat_session_service, get_current_user, get_rag_service
from app.api.security import verify_ownership
from app.core.swagger_constants import OPENAPI_PLACEHOLDER_UUID, is_demo_placeholder
from app.llm.client import LLMClientError, LLMTimeoutError, LLMUnavailableError
from app.models.user import User
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
    summary="Ask a question using RAG",
    description=(
        "Runs retrieval -> prompt building -> LLM generation, persists the user "
        "and assistant messages, and stores citations for retrieved chunks.\n\n"
        "**Session id:** create one first via **POST /chat-sessions** (tag: "
        "Chat Sessions), then paste the returned `id` here. "
        "Or omit `session_id` to auto-use a demo session."
    ),
)
async def ask_chat(
    request: Request,
    response: Response,
    x_request_id: str | None = Header(None, alias="X-Request-ID"),
    current_user: User = Depends(get_current_user),
    rag: RAGService = Depends(get_rag_service),
    session_service: ChatSessionService = Depends(get_chat_session_service),
) -> ChatResponse:
    import logging
    import sys
    import time
    from app.tools.web_search import DuckDuckGoWebSearchProvider
    logger = logging.getLogger(__name__)

    request_id = x_request_id or str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id

    start_time = time.monotonic()
    provider_module = sys.modules.get(DuckDuckGoWebSearchProvider.__module__)
    module_file = getattr(provider_module, "__file__", "unknown")

    # Determine dynamic content parameters
    content_type = request.headers.get("content-type", "")
    question = ""
    session_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    document_version_id: uuid.UUID | None = None
    top_k: int | None = None
    similarity_threshold: float | None = None
    image_bytes: bytes | None = None
    image_name: str | None = None
    image_mime: str | None = None
    image_size: int | None = None
    image_storage_path: str | None = None
    attachments_meta: list[dict[str, Any]] | None = None

    logger.info(
        '[CHAT REQUEST] request_id=%s content_type=%r content_length=%s is_multipart=%s',
        request_id,
        content_type[:80],
        request.headers.get('content-length', 'unknown'),
        'multipart/form-data' in content_type,
    )

    from pydantic import ValidationError
    from fastapi.exceptions import RequestValidationError
    from fastapi import UploadFile

    req_provider: str | None = None
    req_model: str | None = None

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            form_keys = list(form.keys())
            logger.info('[CHAT REQUEST] form_keys=%s file_present=%s', form_keys, 'file' in form_keys)
            question = str(form.get("question", "")).strip()
            req_provider = str(form.get("provider", "")).strip() or None
            req_model = str(form.get("model", "")).strip() or None
            
            session_id_str = form.get("session_id")
            if session_id_str and isinstance(session_id_str, str):
                session_id = uuid.UUID(session_id_str)
                
            doc_id_str = form.get("document_id")
            if doc_id_str and isinstance(doc_id_str, str):
                document_id = uuid.UUID(doc_id_str)
                
            doc_ver_str = form.get("document_version_id")
            if doc_ver_str and isinstance(doc_ver_str, str):
                document_version_id = uuid.UUID(doc_ver_str)
                
            top_k_str = form.get("top_k")
            if top_k_str is not None and isinstance(top_k_str, str):
                top_k = int(top_k_str)
                
            sim_threshold_str = form.get("similarity_threshold")
            if sim_threshold_str is not None and isinstance(sim_threshold_str, str):
                similarity_threshold = float(sim_threshold_str)
                
            file_val = form.get("file")
            has_file = hasattr(file_val, "filename") and bool(getattr(file_val, "filename", None))
            
            # Validate input using ChatRequest schema rules
            validate_question = question if (question or has_file) else ""
            
            params = {
                "question": validate_question,
                "session_id": session_id_str if isinstance(session_id_str, str) else None,
                "document_id": doc_id_str if isinstance(doc_id_str, str) else None,
                "document_version_id": doc_ver_str if isinstance(doc_ver_str, str) else None,
                "top_k": top_k_str if isinstance(top_k_str, str) else None,
                "similarity_threshold": sim_threshold_str if isinstance(sim_threshold_str, str) else None,
                "provider": req_provider,
                "model": req_model,
            }
            params = {k: v for k, v in params.items() if v is not None}
            ChatRequest.model_validate(params)

            if has_file and file_val is not None:
                from app.api.file_utils import validate_image_bytes, resize_image
                from app.services.exceptions import ValidationError as ServiceValidationError
                
                upload_file_obj: Any = file_val
                image_bytes = await upload_file_obj.read()
                filename_str = str(getattr(upload_file_obj, "filename", ""))
                content_type_str = str(getattr(upload_file_obj, "content_type", ""))
                logger.info(
                    '[IMAGE] image_received request_id=%s filename=%s mime=%s size=%d',
                    request_id, filename_str, content_type_str, len(image_bytes)
                )
                if len(image_bytes) == 0:
                    raise HTTPException(status_code=400, detail="Image upload was not received by the server.")
                if len(image_bytes) > 10 * 1024 * 1024:
                    raise HTTPException(status_code=400, detail="Image exceeds the maximum allowed size of 10 MB.")
                
                ext = ""
                if filename_str:
                    ext = filename_str.split(".")[-1].lower()
                if ext not in ["png", "jpg", "jpeg", "webp"]:
                    raise HTTPException(status_code=400, detail="Unsupported file extension. Only PNG, JPEG, and WEBP are supported.")
                    
                try:
                    image_mime = validate_image_bytes(image_bytes)
                except ServiceValidationError as ve:
                    raise HTTPException(status_code=400, detail=str(ve))

                logger.info('[IMAGE] image_validated request_id=%s mime=%s', request_id, image_mime)
                image_bytes = resize_image(image_bytes)
                image_name = filename_str
                image_size = len(image_bytes)
            else:
                logger.debug('[IMAGE] no file found in multipart form request_id=%s form_keys=%s', request_id, list(form.keys()))
        else:
            body = await request.json()
            payload = ChatRequest.model_validate(body)
            question = payload.question
            session_id = payload.session_id
            document_id = payload.document_id
            document_version_id = payload.document_version_id
            top_k = payload.top_k
            similarity_threshold = payload.similarity_threshold
            attachments_meta = [att.model_dump(mode="json") for att in payload.attachments] if payload.attachments else None
            req_provider = payload.provider
            req_model = payload.model
    except ValidationError as err:
        raise RequestValidationError(err.errors())

    logger.info('[CHAT START] request_id=%s query="%s" module_file="%s"', request_id, question, module_file)

    if is_demo_placeholder(session_id):
        session_id = await get_or_create_swagger_demo_session(
            users=UserRepository(session_service.session),
            sessions=ChatSessionRepository(session_service.session),
            session_service=session_service,
            user_id=current_user.id,
        )

    if session_id is None:
        raise HTTPException(status_code=400, detail="Session ID is required.")

    chat_session = await session_service.get(session_id)
    if not chat_session:
        raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found.")
    verify_ownership(chat_session.user_id, current_user, "chat session")

    if image_bytes is not None:
        from app.storage import get_storage_service
        from app.core.config import get_settings
        
        storage = get_storage_service(bucket_name=get_settings().SUPABASE_STORAGE_BUCKET)
        unique_name = f"{uuid.uuid4()}_{image_name or 'upload.png'}"
        storage_path = f"{current_user.id}/{session_id}/{unique_name}"
        
        logger.info(
            '[IMAGE] supabase_upload_started request_id=%s bucket=%s path=%s size=%d mime=%s',
            request_id, get_settings().SUPABASE_STORAGE_BUCKET, storage_path, image_size or 0, image_mime
        )
        try:
            await storage.upload_file(
                content=image_bytes,
                storage_path=storage_path,
                mime_type=image_mime or "application/octet-stream"
            )
            image_storage_path = storage_path
            logger.info(
                '[IMAGE] supabase_upload_success request_id=%s path=%s',
                request_id, image_storage_path
            )
        except Exception as e:
            logger.error("[IMAGE] supabase_upload_failed request_id=%s error=%s", request_id, e)
            raise HTTPException(status_code=500, detail="Image upload failed. Please try again.")

    if document_id is None and attachments_meta:
        for att in attachments_meta:
            doc_id_val = att.get("document_id")
            if doc_id_val:
                try:
                    document_id = uuid.UUID(str(doc_id_val))
                    logger.info("[CHAT API] Extracted document_id=%s from attachments_meta", document_id)
                    break
                except ValueError:
                    pass

    filters = SearchFilters(
        user_id=current_user.id,
        document_id=document_id,
        document_version_id=document_version_id,
    )

    try:
        result = await rag.ask(
            session_id,
            question,
            filters=filters,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            request_id=request_id,
            image=image_bytes,
            image_storage_path=image_storage_path,
            image_name=image_name,
            image_mime=image_mime,
            image_size=image_size,
            attachments=attachments_meta,
            provider=req_provider,
            model=req_model,
        )

        total_ms = int((time.monotonic() - start_time) * 1000)
        logger.info('[CHAT END] request_id=%s status=200 total_ms=%d', request_id, total_ms)
    except RAGError as exc:
        total_ms = int((time.monotonic() - start_time) * 1000)
        logger.error("[CHAT END] request_id=%s status=400 total_ms=%d error=%s", request_id, total_ms, exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LLMClientError as exc:
        total_ms = int((time.monotonic() - start_time) * 1000)
        logger.error("[CHAT END] request_id=%s total_ms=%d error=%s", request_id, total_ms, exc)
        raise exc

    return _to_chat_response(result)


@router.post(
    "/stream",
    summary="Ask a question using RAG (Server-Sent Events streaming)",
    description="Streams RAG tokens and citations in real time via Server-Sent Events (SSE).",
)
async def ask_chat_stream(
    request: Request,
    rag: RAGService = Depends(get_rag_service),
    session_service: ChatSessionService = Depends(get_chat_session_service),
):
    from fastapi.responses import StreamingResponse

    # Determine dynamic content parameters
    content_type = request.headers.get("content-type", "")
    request_id = request.headers.get("X-Request-ID") or f"req-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    question = ""
    session_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    document_version_id: uuid.UUID | None = None
    top_k: int | None = None
    similarity_threshold: float | None = None
    image_bytes: bytes | None = None
    image_name: str | None = None
    image_mime: str | None = None
    image_size: int | None = None
    image_storage_path: str | None = None
    attachments_meta: list[dict[str, Any]] | None = None
    req_provider: str | None = None
    req_model: str | None = None

    from pydantic import ValidationError
    from fastapi.exceptions import RequestValidationError
    from fastapi import UploadFile

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            question = str(form.get("question", "")).strip()
            
            session_id_str = form.get("session_id")
            if session_id_str and isinstance(session_id_str, str):
                session_id = uuid.UUID(session_id_str)
                
            doc_id_str = form.get("document_id")
            if doc_id_str and isinstance(doc_id_str, str):
                document_id = uuid.UUID(doc_id_str)
                
            doc_ver_str = form.get("document_version_id")
            if doc_ver_str and isinstance(doc_ver_str, str):
                document_version_id = uuid.UUID(doc_ver_str)
                
            top_k_str = form.get("top_k")
            if top_k_str is not None and isinstance(top_k_str, str):
                top_k = int(top_k_str)
                
            sim_threshold_str = form.get("similarity_threshold")
            if sim_threshold_str is not None and isinstance(sim_threshold_str, str):
                similarity_threshold = float(sim_threshold_str)

            provider_str = form.get("provider")
            if provider_str and isinstance(provider_str, str):
                req_provider = provider_str

            model_str = form.get("model")
            if model_str and isinstance(model_str, str):
                req_model = model_str

            file_val = form.get("file")
            has_file = hasattr(file_val, "filename") and bool(getattr(file_val, "filename", None))
            
            # Validate input using ChatRequest schema rules
            validate_question = question if (question or has_file) else ""
            
            params = {
                "question": validate_question,
                "session_id": session_id_str if isinstance(session_id_str, str) else None,
                "document_id": doc_id_str if isinstance(doc_id_str, str) else None,
                "document_version_id": doc_ver_str if isinstance(doc_ver_str, str) else None,
                "top_k": top_k_str if isinstance(top_k_str, str) else None,
                "similarity_threshold": sim_threshold_str if isinstance(sim_threshold_str, str) else None,
            }
            params = {k: v for k, v in params.items() if v is not None}
            ChatRequest.model_validate(params)

            if has_file and file_val is not None:
                from app.api.file_utils import validate_image_bytes, resize_image
                from app.services.exceptions import ValidationError as ServiceValidationError
                import logging as _logging
                _logger = _logging.getLogger(__name__)
                
                upload_file_obj: Any = file_val
                image_bytes = await upload_file_obj.read()
                filename_str = str(getattr(upload_file_obj, "filename", ""))
                content_type_str = str(getattr(upload_file_obj, "content_type", ""))
                _logger.info(
                    '[IMAGE] image_received (stream) filename=%s mime=%s size=%d',
                    filename_str, content_type_str, len(image_bytes)
                )
                if len(image_bytes) == 0:
                    raise HTTPException(status_code=400, detail="Image upload was not received by the server.")
                if len(image_bytes) > 10 * 1024 * 1024:
                    raise HTTPException(status_code=400, detail="Image exceeds the maximum allowed size of 10 MB.")
                
                ext = ""
                if filename_str:
                    ext = filename_str.split(".")[-1].lower()
                if ext not in ["png", "jpg", "jpeg", "webp"]:
                    raise HTTPException(status_code=400, detail="Unsupported file extension. Only PNG, JPEG, and WEBP are supported.")
                    
                try:
                    image_mime = validate_image_bytes(image_bytes)
                except ServiceValidationError as ve:
                    raise HTTPException(status_code=400, detail=str(ve))

                _logger.info('[IMAGE] image_validated (stream) mime=%s', image_mime)
                image_bytes = resize_image(image_bytes)
                image_name = filename_str
                image_size = len(image_bytes)
            else:
                import logging as _logging
                _logging.getLogger(__name__).debug('[IMAGE] no file in stream multipart form form_keys=%s', list(form.keys()))
        else:
            body = await request.json()
            payload = ChatRequest.model_validate(body)
            question = payload.question
            session_id = payload.session_id
            document_id = payload.document_id
            document_version_id = payload.document_version_id
            top_k = payload.top_k
            similarity_threshold = payload.similarity_threshold
            attachments_meta = [att.model_dump(mode="json") for att in payload.attachments] if payload.attachments else None
            req_provider = payload.provider
            req_model = payload.model
    except ValidationError as err:
        raise RequestValidationError(err.errors())

    if is_demo_placeholder(session_id):
        session_id = await get_or_create_swagger_demo_session(
            users=UserRepository(session_service.session),
            sessions=ChatSessionRepository(session_service.session),
            session_service=session_service,
        )

    if session_id is None:
        raise HTTPException(status_code=400, detail="Session ID is required.")

    chat_session = await session_service.get(session_id)
    if not chat_session:
        raise HTTPException(status_code=404, detail=f"Chat session {session_id} not found.")
    
    # In stream we might not have current_user if it's SSE without auth header easily,
    # but verify_ownership isn't used here currently. We will just use chat_session.user_id for path.
    if image_bytes is not None:
        import logging as _logging
        from app.storage import get_storage_service
        from app.core.config import get_settings
        
        _stream_logger = _logging.getLogger(__name__)
        storage = get_storage_service(bucket_name=get_settings().SUPABASE_STORAGE_BUCKET)
        unique_name = f"{uuid.uuid4()}_{image_name or 'upload.png'}"
        storage_path = f"{chat_session.user_id}/{session_id}/{unique_name}"
        
        _stream_logger.info(
            '[IMAGE] supabase_upload_started (stream) bucket=%s path=%s size=%d mime=%s',
            get_settings().SUPABASE_STORAGE_BUCKET, storage_path, image_size or 0, image_mime
        )
        try:
            await storage.upload_file(
                content=image_bytes,
                storage_path=storage_path,
                mime_type=image_mime or "application/octet-stream"
            )
            image_storage_path = storage_path
            _stream_logger.info('[IMAGE] supabase_upload_success (stream) path=%s', image_storage_path)
        except Exception as e:
            _stream_logger.error("[IMAGE] supabase_upload_failed (stream) error=%s", e)
            raise HTTPException(status_code=500, detail="Image upload failed. Please try again.")

    if not image_storage_path and attachments_meta:
        for att in attachments_meta:
            sp = att.get("storage_path")
            mime = str(att.get("mime_type") or "").lower()
            fname = str(att.get("filename") or "").lower()
            is_img = mime.startswith("image/") or any(fname.endswith(f".{ext}") for ext in ("png", "jpg", "jpeg", "webp"))
            if sp and is_img:
                image_storage_path = sp
                image_name = image_name or att.get("filename")
                image_mime = image_mime or (mime if mime.startswith("image/") else "image/png")
                image_size = image_size or att.get("size")
                logger.info("[IMAGE] extracted image_storage_path=%s from attachments_meta", image_storage_path)
                break

    if document_id is None and attachments_meta:
        for att in attachments_meta:
            doc_id_val = att.get("document_id")
            if doc_id_val:
                try:
                    document_id = uuid.UUID(str(doc_id_val))
                    logger.info("[CHAT API stream] Extracted document_id=%s from attachments_meta", document_id)
                    break
                except ValueError:
                    pass

    filters = SearchFilters(
        document_id=document_id,
        document_version_id=document_version_id,
    )

    logger.info(
        "stage=rag_request_received request_id=%s session_id=%s question=%r provider=%s model=%s stream=true",
        request_id, session_id, question[:100], req_provider, req_model
    )

    generator = rag.ask_stream(
        session_id,
        question,
        filters=filters,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        request_id=request_id,
        image=image_bytes,
        image_storage_path=image_storage_path,
        image_name=image_name,
        image_mime=image_mime,
        image_size=image_size,
        attachments=attachments_meta,
        provider=req_provider,
        model=req_model,
    )

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/sessions/{session_id}",
    response_model=ChatSessionResponse,
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
                document_title=source.document_title,
                section_title=source.section_title,
                page_number=source.page_number,
                similarity_score=source.similarity_score,
                rank=source.rank,
                url=getattr(source, "url", None),
                domain=getattr(source, "domain", None),
                source_type=getattr(source, "source_type", "local"),
            )
            for source in result.sources
        ],
        token_usage=token_usage,
        model=result.model,
        processing_time_ms=result.processing_time_ms,
        user_message_id=result.user_message_id,
        assistant_message_id=result.assistant_message_id,
        retrieval_mode=getattr(result, "retrieval_mode", "local") or "local",
    )

