"""Constants shared by Swagger/OpenAPI helpers (no service-layer imports)."""
import uuid

# FastAPI/Swagger UI auto-fills this UUID for `format: uuid` fields when no
# per-field example is present. It is not a real database row.
OPENAPI_PLACEHOLDER_UUID = uuid.UUID("3fa85f64-5717-4562-b3fc-2c963f66afa6")
ZERO_PLACEHOLDER_UUID = uuid.UUID("00000000-0000-0000-0000-000000000000")
DEMO_PLACEHOLDER_UUIDS = {OPENAPI_PLACEHOLDER_UUID, ZERO_PLACEHOLDER_UUID}


def is_demo_placeholder(session_id: uuid.UUID | None) -> bool:
    """Return True if session_id is None or matches a known demo placeholder UUID."""
    if session_id is None:
        return True
    return session_id in DEMO_PLACEHOLDER_UUIDS
