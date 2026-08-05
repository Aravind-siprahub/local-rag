"""Constants shared by Swagger/OpenAPI helpers (no service-layer imports)."""
import uuid

# FastAPI/Swagger UI auto-fills this UUID for `format: uuid` fields when no
# per-field example is present. It is not a real database row.
OPENAPI_PLACEHOLDER_UUID = uuid.UUID("3fa85f64-5717-4562-b3fc-2c963f66afa6")
