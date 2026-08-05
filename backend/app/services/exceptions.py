"""Service-layer exceptions — plain Python, no FastAPI dependency.

`app/core/exceptions.py` already defines `AppError` with an HTTP status code
baked in, but it imports `fastapi.status` to do it. Services must stay
importable and testable without FastAPI installed or running, so they raise
these instead. Translating a `ServiceError` into an HTTP response (404 for
`NotFoundError`, 409 for `ConflictError`, 422 for `ValidationError`) is the
future API layer's job, via its own exception handler — not this module's.
"""


class ServiceError(Exception):
    """Base class for all service-layer errors."""


class NotFoundError(ServiceError):
    """The requested entity does not exist (or is soft-deleted, where that
    matters for the operation)."""


class ConflictError(ServiceError):
    """A business/uniqueness rule would be violated by this operation."""


class ValidationError(ServiceError):
    """Input fails a business rule that isn't already enforced by the
    Pydantic schema layer or a database constraint — e.g. a state-machine
    transition, or a cross-entity consistency check."""
