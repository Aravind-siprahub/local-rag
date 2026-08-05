"""Shared SQLAlchemy declarative base.

All ORM models (see `app/models/`) inherit from this and only this, so
`Base.metadata` — the source Alembic's `env.py` targets — sees every table.
"""
from datetime import datetime

from sqlalchemy import DateTime, MetaData, Text
from sqlalchemy.orm import DeclarativeBase

# Mirrors PostgreSQL's own default auto-naming for constraints that were
# created without an explicit name in db/sql/003_tables.sql (plain
# `REFERENCES`, inline `PRIMARY KEY`). Without this, any unnamed
# constraint/index SQLAlchemy generates for comparison gets an
# SQLAlchemy-internal default name that doesn't match what Postgres actually
# named it when the raw SQL ran — which would make `alembic revision
# --autogenerate` report spurious rename/drop/create diffs against a schema
# that hasn't actually changed. All CHECK/UNIQUE constraints in this project
# are explicitly named in db/sql/003_tables.sql and are named to match
# explicitly in each model instead of relying on this convention.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "%(table_name)s_%(column_0_name)s_key",
    "fk": "%(table_name)s_%(column_0_name)s_fkey",
    "pk": "%(table_name)s_pkey",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    # Overrides SQLAlchemy's built-in annotation -> column type defaults,
    # which would otherwise silently produce the wrong DDL against this
    # already-deployed schema:
    #   - bare `str`      -> String/VARCHAR, but every text column here is TEXT
    #   - bare `datetime`  -> TIMESTAMP WITHOUT TIME ZONE, but every timestamp
    #                          column here is TIMESTAMPTZ
    # Columns that are genuinely something else (CITEXT, CHAR(n)) pass an
    # explicit type to `mapped_column()`, which always takes precedence over
    # this map.
    type_annotation_map = {
        str: Text,
        datetime: DateTime(timezone=True),
    }
