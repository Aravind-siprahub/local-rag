"""Central OpenAPI example payloads — single source of truth for Swagger Try-it-out.

Runtime-specific values (dynamic user ids, unique emails) are patched in
`app.core.openapi.patch_openapi_with_examples` at startup; static field
examples here prevent Swagger from autogenerating invalid `"string"` values.
"""
from typing import Any

from app.schemas.validators.password import PASSWORD_EXAMPLE

EMAIL_EXAMPLE = "john.doe@example.com"
FULL_NAME_EXAMPLE = "John Doe"

USER_CREATE_OPENAPI_EXAMPLE: dict[str, Any] = {
    "email": EMAIL_EXAMPLE,
    "full_name": FULL_NAME_EXAMPLE,
    "password": PASSWORD_EXAMPLE,
    "role": "member",
}

DOCUMENT_CREATE_OPENAPI_EXAMPLE: dict[str, Any] = {
    "title": "My first document",
    "description": "Optional description",
    "tags": ["demo"],
}

CHAT_SESSION_CREATE_OPENAPI_EXAMPLE: dict[str, Any] = {
    "title": "Research session",
}

CHAT_REQUEST_OPENAPI_EXAMPLE: dict[str, Any] = {
    "question": "What are the key findings in the uploaded report?",
    "top_k": 5,
    "similarity_threshold": 0.7,
    "document_id": None,
    "document_version_id": None,
    # session_id is injected at startup from a real chat session
}
