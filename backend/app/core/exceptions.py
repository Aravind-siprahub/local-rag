"""Application-level exceptions and their FastAPI handlers.

Concerns living here, deliberately kept together:
1. `AppError` and subclasses — typed errors the application layer can raise
   without importing FastAPI (keeps business/domain code framework-agnostic,
   per Clean Architecture's dependency rule).
2. `register_exception_handlers` — the *only* place that translates any
   exception into an HTTP response, so every error path returns the same
   JSON shape instead of each router inventing its own. This is also where
   `app.services.exceptions.*` (raised by the service layer, which is
   deliberately FastAPI-free) get mapped to HTTP status codes — endpoints
   never catch these themselves, which is what keeps them thin.
"""
import logging

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.llm.client import LLMUnavailableError
from app.services.exceptions import ConflictError, NotFoundError, ValidationError

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for all application-raised errors.

    Subclass this for domain/service-layer failures instead of raising
    `HTTPException` outside the API layer — keeps the service layer free of
    a FastAPI import, so it stays testable and reusable outside a web
    context.
    """

    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class DatabaseUnavailableError(AppError):
    """Raised when the database cannot be reached."""

    def __init__(self, message: str = "The database is currently unavailable.") -> None:
        super().__init__(message, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach global exception handlers to the FastAPI app.

    Ordered most-specific to least-specific; FastAPI dispatches on the most
    specific matching type regardless of registration order, but keeping
    that order here makes the fallback intent explicit for readers.
    """

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.info(
            "Request validation failed on %s %s: %s",
            request.method,
            request.url.path,
            exc.errors(),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": jsonable_encoder(exc.errors())},
        )

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        logger.error("Application error on %s %s: %s", request.method, request.url.path, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"status": "error", "message": exc.message},
        )

    @app.exception_handler(LLMUnavailableError)
    async def llm_unavailable_error_handler(
        request: Request, exc: LLMUnavailableError
    ) -> JSONResponse:
        logger.error(
            "LLM unavailable on %s %s: reason=%s details=%s",
            request.method,
            request.url.path,
            exc.reason,
            exc.details,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=exc.to_response_body(),
        )

    @app.exception_handler(NotFoundError)
    async def not_found_error_handler(request: Request, exc: NotFoundError) -> JSONResponse:
        logger.info("Not found on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

    @app.exception_handler(ConflictError)
    async def conflict_error_handler(request: Request, exc: ConflictError) -> JSONResponse:
        logger.info("Conflict on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": str(exc)})

    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
        logger.info("Business validation failed on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": str(exc)})

    @app.exception_handler(SQLAlchemyError)
    async def sqlalchemy_error_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        logger.exception("Unhandled database error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": "A database error occurred."},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": "An unexpected error occurred."},
        )
