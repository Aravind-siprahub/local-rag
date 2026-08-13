"""Unit tests verifying Current Document Version filtering in search layer."""
import uuid
import pytest
from app.retrieval.search import SearchFilters


def test_search_filters_default_version() -> None:
    filters = SearchFilters()
    assert filters.document_version_id is None


def test_search_filters_explicit_version() -> None:
    version_id = uuid.uuid4()
    filters = SearchFilters(document_version_id=version_id)
    assert filters.document_version_id == version_id
