"""Response schemas for the health-check endpoint."""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    database: str


class HealthErrorResponse(BaseModel):
    status: str
    database: str
    message: str
    error: str
