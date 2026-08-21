"""Response schemas for the health-check endpoint."""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    database: str
    pgvector: str
    ollama: str
    models: list[str] | None = None


class HealthErrorResponse(BaseModel):
    status: str
    database: str
    pgvector: str
    ollama: str
    message: str
    error: str
