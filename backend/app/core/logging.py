"""Logging configuration.

Called once at process startup (from `app.main`) before anything else logs.
"""
import logging
import sys

from app.core.config import get_settings


def setup_logging() -> None:
    settings = get_settings()

    logging.basicConfig(
        level=logging.DEBUG if settings.DEBUG else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,  # override any handlers a dependency may have configured on import
    )

    # SQLAlchemy's engine logger is separately controlled by DB_ECHO — avoid
    # duplicating that toggle by mirroring it here rather than letting
    # basicConfig's root level silence/verbose it inconsistently.
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if settings.DB_ECHO else logging.WARNING
    )
