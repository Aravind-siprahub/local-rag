"""OpenAPI / Swagger UI example patching.

Swagger UI (OpenAPI 3.0) prefers singular property ``example`` values and
named media-type ``examples`` on request bodies. Pydantic v2 emits JSON
Schema ``examples`` arrays, which Swagger often ignores — filling Try-it-out
with invalid defaults like ``"string"``. This module normalizes the generated
schema so Try-it-out always ships valid payloads.

Kept free of ``app.services`` imports to avoid circular imports with
``DocumentService`` / other service modules.
"""
from typing import Any

from app.schemas.openapi_examples import (
    CHAT_REQUEST_OPENAPI_EXAMPLE,
    CHAT_SESSION_CREATE_OPENAPI_EXAMPLE,
    DOCUMENT_CREATE_OPENAPI_EXAMPLE,
    USER_CREATE_OPENAPI_EXAMPLE,
)
from app.schemas.validators.password import PASSWORD_EXAMPLE


def _promote_examples_to_example(node: Any) -> None:
    """Recursively copy JSON Schema ``examples[0]`` onto OpenAPI ``example``."""
    if isinstance(node, dict):
        examples = node.get("examples")
        if (
            "example" not in node
            and isinstance(examples, list)
            and examples
            and not isinstance(examples[0], dict)
        ):
            node["example"] = examples[0]
        for value in node.values():
            _promote_examples_to_example(value)
    elif isinstance(node, list):
        for item in node:
            _promote_examples_to_example(item)


def _set_schema_example(schema: dict, schema_name: str, example: dict[str, Any]) -> None:
    target = schema.get("components", {}).get("schemas", {}).get(schema_name)
    if target is None:
        return
    target["example"] = example
    properties = target.get("properties", {})
    for key, value in example.items():
        prop = properties.get(key)
        if prop is not None:
            prop["example"] = value
            prop["examples"] = [value]


def _set_request_body_example(
    schema: dict, path: str, method: str, content_type: str, example: dict[str, Any]
) -> None:
    content = (
        schema.get("paths", {})
        .get(path, {})
        .get(method, {})
        .get("requestBody", {})
        .get("content", {})
        .get(content_type)
    )
    if content is None:
        return
    content["example"] = example
    content["examples"] = {
        "default": {
            "summary": "Valid request",
            "value": example,
        }
    }


def patch_openapi_with_examples(
    schema: dict,
    example_user_id: str | None = None,
    example_user_email: str | None = None,
    example_session_id: str | None = None,
) -> dict:
    """Make Swagger Try-it-out use valid payloads across the API."""
    _promote_examples_to_example(schema)

    user_example = {
        **USER_CREATE_OPENAPI_EXAMPLE,
        "email": example_user_email or USER_CREATE_OPENAPI_EXAMPLE["email"],
        "password": PASSWORD_EXAMPLE,
    }
    _set_schema_example(schema, "UserCreate", user_example)
    _set_request_body_example(schema, "/users", "post", "application/json", user_example)

    if example_user_id:
        document_example = {
            "user_id": example_user_id,
            **DOCUMENT_CREATE_OPENAPI_EXAMPLE,
        }
        _set_schema_example(schema, "DocumentCreate", document_example)
        _set_request_body_example(
            schema, "/documents", "post", "application/json", document_example
        )

        upload_multipart = (
            schema.get("paths", {})
            .get("/documents/upload", {})
            .get("post", {})
            .get("requestBody", {})
            .get("content", {})
            .get("multipart/form-data")
        )
        if upload_multipart is not None:
            upload_example = {
                "user_id": example_user_id,
                "title": DOCUMENT_CREATE_OPENAPI_EXAMPLE["title"],
            }
            upload_multipart["example"] = upload_example
            upload_multipart["examples"] = {
                "default": {"summary": "Valid upload", "value": upload_example}
            }
            upload_props = upload_multipart.get("schema", {}).get("properties", {})
            user_id_field = upload_props.get("user_id")
            if user_id_field is not None:
                user_id_field["example"] = example_user_id

        chat_session_example = {
            "user_id": example_user_id,
            **CHAT_SESSION_CREATE_OPENAPI_EXAMPLE,
        }
        _set_schema_example(schema, "ChatSessionCreate", chat_session_example)
        _set_request_body_example(
            schema, "/chat-sessions", "post", "application/json", chat_session_example
        )

        # GET /chat-sessions requires (or accepts) user_id as a query param —
        # without an example Swagger Try-it-out omits it and returns 422.
        get_sessions = schema.get("paths", {}).get("/chat-sessions", {}).get("get", {})
        for param in get_sessions.get("parameters", []):
            if param.get("name") == "user_id" and param.get("in") == "query":
                param["example"] = example_user_id
                param.setdefault("schema", {})["example"] = example_user_id
                param["required"] = False
                break

    chat_example = {**CHAT_REQUEST_OPENAPI_EXAMPLE}
    if example_session_id:
        chat_example["session_id"] = example_session_id
    _set_schema_example(schema, "ChatRequest", chat_example)
    _set_request_body_example(schema, "/chat", "post", "application/json", chat_example)

    return schema
