"""Python enums mirroring the PostgreSQL enum types created by
`db/sql/002_enums.sql`.

`create_type=False` on every mapping below is load-bearing: these PostgreSQL
types already exist in the database. Without it, SQLAlchemy would try to
`CREATE TYPE` them again the first time it issues DDL for these columns
(e.g. in an offline Alembic SQL dump), colliding with the existing type.
"""
import enum
from collections.abc import Callable

from sqlalchemy import Enum as SqlEnum


def pg_enum(python_enum: type[enum.Enum], name: str) -> SqlEnum:
    """Build a SQLAlchemy Enum bound to an existing PostgreSQL enum type.

    `values_callable` makes SQLAlchemy send the enum *values* (e.g.
    `"member"`) rather than member *names* (`"MEMBER"`) — required since the
    Postgres type's labels are lowercase.
    """
    values_callable: Callable[[type[enum.Enum]], list[str]] = lambda e: [m.value for m in e]
    return SqlEnum(
        python_enum,
        name=name,
        create_type=False,
        validate_strings=True,
        values_callable=values_callable,
    )


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    MEMBER = "member"


class DocumentStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    ARCHIVED = "archived"


class DocumentVersionStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    PARSING = "parsing"
    PARSED = "parsed"
    CHUNKING = "chunking"
    CHUNKED = "chunked"
    EMBEDDING = "embedding"
    EMBEDDED = "embedded"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingJobType(str, enum.Enum):
    UPLOAD = "upload"
    PARSE = "parse"
    CHUNK = "chunk"
    EMBED = "embed"
    INDEX = "index"


class ProcessingJobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageRole(str, enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class VectorMetric(str, enum.Enum):
    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "inner_product"
