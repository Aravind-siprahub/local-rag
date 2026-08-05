"""Unit tests for `app.retrieval.search`."""
import uuid

import pytest

from app.retrieval.search import SearchFilters, search_similar


class TestSearchSimilarValidation:
    @pytest.mark.asyncio
    async def test_rejects_wrong_embedding_dimensions(self) -> None:
        with pytest.raises(ValueError, match="768"):
            await search_similar(None, [0.1, 0.2], model_name="test", top_k=5)


class TestSearchFilters:
    def test_defaults_are_none(self) -> None:
        filters = SearchFilters()
        assert filters.user_id is None
        assert filters.document_id is None
        assert filters.document_version_id is None

    def test_accepts_scope_ids(self) -> None:
        user_id = uuid.uuid4()
        document_id = uuid.uuid4()
        version_id = uuid.uuid4()
        filters = SearchFilters(user_id=user_id, document_id=document_id, document_version_id=version_id)
        assert filters.user_id == user_id
        assert filters.document_id == document_id
        assert filters.document_version_id == version_id
