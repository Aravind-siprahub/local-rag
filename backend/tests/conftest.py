"""Shared pytest configuration for backend tests."""
import os

# Parser/cleaner/chunker tests do not need a database, but importing the
# processor or services pulls in SQLAlchemy settings validation.
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres",
)
