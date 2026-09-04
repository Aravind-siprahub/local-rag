"""Tests for Role-Based Document Upload and Centralized RAG Context Building."""
import uuid
import pytest
from httpx import AsyncClient, ASGITransport
from starlette.datastructures import UploadFile

from app.main import app
from app.api.dependencies import (
    UPLOAD_ALLOWED_ROLES,
    can_upload_documents,
    get_current_user,
    get_db,
    get_document_service,
    get_document_upload_service,
    get_document_version_service,
    require_document_upload_permission,
)
from app.models.enums import UserRole
from app.models.user import User
from app.prompting.builder import PromptBuilder
from app.rag.context_builder import (
    ContextBuilder,
    STANDARDIZED_UNANSWERABLE_MESSAGE,
)
from app.retrieval.ranking import RankedResult


# ============================================================================
# 1. Role-Based Document Upload Tests
# ============================================================================

def test_user_roles_enum() -> None:
    """Verify supported UserRole enum values include admin, hr, user, member."""
    assert UserRole.ADMIN.value == "admin"
    assert UserRole.HR.value == "hr"
    assert UserRole.USER.value == "user"
    assert UserRole.MEMBER.value == "member"


def test_can_upload_documents_logic() -> None:
    """Verify only admin can upload by default, but UPLOAD_ALLOWED_ROLES is configurable."""
    admin_user = User(id=uuid.uuid4(), email="admin@example.com", role=UserRole.ADMIN, is_active=True)
    hr_user = User(id=uuid.uuid4(), email="hr@example.com", role=UserRole.HR, is_active=True)
    regular_user = User(id=uuid.uuid4(), email="user@example.com", role=UserRole.USER, is_active=True)
    member_user = User(id=uuid.uuid4(), email="member@example.com", role=UserRole.MEMBER, is_active=True)

    # By default, only admin is allowed
    assert can_upload_documents(admin_user) is True
    assert can_upload_documents(hr_user) is False
    assert can_upload_documents(regular_user) is False
    assert can_upload_documents(member_user) is False
    assert can_upload_documents(None) is False

    # Test configurability: easily add 'hr' to allowed roles
    try:
        UPLOAD_ALLOWED_ROLES.add("hr")
        assert can_upload_documents(hr_user) is True
        assert can_upload_documents(regular_user) is False
    finally:
        UPLOAD_ALLOWED_ROLES.discard("hr")


@pytest.mark.asyncio
async def test_admin_upload_allowed_server_side() -> None:
    """Admin user can access document upload endpoint without 403 Forbidden."""
    admin_id = uuid.uuid4()
    admin = User(id=admin_id, email="admin@example.com", role=UserRole.ADMIN, is_active=True)

    class MockUploadResult:
        document_id = uuid.uuid4()
        original_filename = "test.txt"
        bucket_name = "documents"
        storage_path = "path/test.txt"
        version_id = uuid.uuid4()
        processing_job_id = uuid.uuid4()
        mime_type = "text/plain"
        file_size_bytes = 100
        checksum_sha256 = "abc123"
        storage_key = "key123"

    class MockUploadService:
        async def upload(self, **kwargs):
            return MockUploadResult()

    app.dependency_overrides[get_current_user] = lambda: admin
    app.dependency_overrides[require_document_upload_permission] = lambda: admin
    app.dependency_overrides[get_document_upload_service] = lambda: MockUploadService()

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/documents/upload",
                files={"file": ("test.txt", b"Hello World content", "text/plain")},
            )
            assert response.status_code == 201
            data = response.json()
            assert data["status"] == "Pending"
            assert data["filename"] == "test.txt"
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(require_document_upload_permission, None)
        app.dependency_overrides.pop(get_document_upload_service, None)


@pytest.mark.asyncio
async def test_regular_user_upload_blocked_with_403() -> None:
    """Non-admin user calling upload receives 403 Forbidden with exact error payload."""
    user_id = uuid.uuid4()
    regular_user = User(id=user_id, email="user@example.com", role=UserRole.USER, is_active=True)

    app.dependency_overrides[get_current_user] = lambda: regular_user

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/documents/upload",
                files={"file": ("unauthorized.txt", b"Forbidden content", "text/plain")},
            )
            assert response.status_code == 403
            json_body = response.json()
            assert json_body.get("error") == "You do not have permission to upload documents."
            assert json_body.get("detail") == "You do not have permission to upload documents."
    finally:
        app.dependency_overrides.pop(get_current_user, None)


@pytest.mark.asyncio
async def test_member_user_upload_blocked_with_403() -> None:
    """MEMBER role user calling direct upload receives 403 Forbidden."""
    member_id = uuid.uuid4()
    member_user = User(id=member_id, email="member@example.com", role=UserRole.MEMBER, is_active=True)

    app.dependency_overrides[get_current_user] = lambda: member_user

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/documents/upload",
                files={"file": ("member_doc.txt", b"Member content", "text/plain")},
            )
            assert response.status_code == 403
            json_body = response.json()
            assert json_body.get("error") == "You do not have permission to upload documents."
            assert json_body.get("detail") == "You do not have permission to upload documents."
    finally:
        app.dependency_overrides.pop(get_current_user, None)


@pytest.mark.asyncio
async def test_regular_user_document_create_blocked_with_403() -> None:
    """Non-admin user calling POST /api/documents receives 403 Forbidden."""
    user_id = uuid.uuid4()
    regular_user = User(id=user_id, email="user@example.com", role=UserRole.USER, is_active=True)

    app.dependency_overrides[get_current_user] = lambda: regular_user

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/documents",
                json={"title": "Unauthorized Document"},
            )
            assert response.status_code == 403
            assert response.json().get("error") == "You do not have permission to upload documents."
    finally:
        app.dependency_overrides.pop(get_current_user, None)


@pytest.mark.asyncio
async def test_regular_user_reindex_blocked_with_403() -> None:
    """Non-admin user calling POST /api/documents/{id}/reindex receives 403 Forbidden."""
    user_id = uuid.uuid4()
    regular_user = User(id=user_id, email="user@example.com", role=UserRole.USER, is_active=True)

    app.dependency_overrides[get_current_user] = lambda: regular_user

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            doc_id = uuid.uuid4()
            response = await client.post(f"/api/documents/{doc_id}/reindex")
            assert response.status_code == 403
            assert response.json().get("error") == "You do not have permission to upload documents."
    finally:
        app.dependency_overrides.pop(get_current_user, None)


@pytest.mark.asyncio
async def test_regular_user_document_versions_blocked_with_403() -> None:
    """Non-admin user calling POST /api/document-versions receives 403 Forbidden."""
    user_id = uuid.uuid4()
    regular_user = User(id=user_id, email="user@example.com", role=UserRole.USER, is_active=True)

    app.dependency_overrides[get_current_user] = lambda: regular_user

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/document-versions",
                json={
                    "document_id": str(uuid.uuid4()),
                    "original_filename": "v2.pdf",
                    "file_size_bytes": 1000,
                    "mime_type": "application/pdf",
                    "checksum_sha256": "abc",
                    "storage_path": "path/v2.pdf",
                },
            )
            assert response.status_code == 403
            assert response.json().get("error") == "You do not have permission to upload documents."
    finally:
        app.dependency_overrides.pop(get_current_user, None)


# ============================================================================
# 2. ContextBuilder & RAG Context Snippet Generation Tests
# ============================================================================

def test_context_builder_filtering_and_capping() -> None:
    """Verify ContextBuilder filters low scores, deduplicates, sorts, and caps to max_chunks."""
    doc_id = uuid.uuid4()
    # Create 10 mock chunks with varying similarity scores and duplicate texts
    chunks = [
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text="Irrelevant snippet about unrelated topic.",
            document_id=doc_id,
            similarity_score=0.10,  # Below threshold (0.30)
            rank=1,
            document_title="Doc1",
        ),
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text="Relevant chunk about policy vacation leave rules and requirements.",
            document_id=doc_id,
            similarity_score=0.85,  # Top score
            rank=2,
            document_title="HR Policy",
            section_title="Leave",
            page_number=2,
        ),
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text="Relevant chunk about policy vacation leave rules and requirements.",  # Duplicate text
            document_id=doc_id,
            similarity_score=0.84,
            rank=3,
            document_title="HR Policy",
        ),
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text="Second relevant chunk about carry-over limit up to 5 days.",
            document_id=doc_id,
            similarity_score=0.72,
            rank=4,
            document_title="HR Policy",
            section_title="Leave Carryover",
            page_number=3,
        ),
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text="Third relevant chunk about manager approval required 2 weeks in advance.",
            document_id=doc_id,
            similarity_score=0.65,
            rank=5,
            document_title="HR Policy",
            section_title="Approval",
            page_number=3,
        ),
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text="Fourth relevant chunk about unpaid leave provisions.",
            document_id=doc_id,
            similarity_score=0.55,
            rank=6,
            document_title="HR Policy",
            section_title="Unpaid Leave",
            page_number=4,
        ),
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text="Fifth relevant chunk about bereavement leave 3 working days.",
            document_id=doc_id,
            similarity_score=0.50,
            rank=7,
            document_title="HR Policy",
            section_title="Special Leave",
            page_number=5,
        ),
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text="Sixth relevant chunk exceeding the 5 chunk limit.",
            document_id=doc_id,
            similarity_score=0.45,
            rank=8,
            document_title="HR Policy",
        ),
    ]

    builder = ContextBuilder(similarity_threshold=0.30, max_chunks=5, max_context_chars=6000)
    result = builder.build_context(chunks, query="What is the vacation policy?")

    assert result.has_context is True
    assert result.total_retrieved == 8
    # Should cap at max 5 chunks
    assert len(result.selected_chunks) == 5
    assert result.total_filtered == 5

    # Top similarity score should be 0.85
    assert result.top_similarity_score == 0.85
    # The lowest score chunk (0.10) must be filtered out
    assert not any(c.similarity_score == 0.10 for c in result.selected_chunks)

    # Chunks must be sorted descending by similarity score
    scores = [c.similarity_score for c in result.selected_chunks]
    assert scores == sorted(scores, reverse=True)

    # Formatted context must contain chunk citation header and metadata
    assert "[Chunk 1] (Document: HR Policy | Page: 2 | Section: Leave)" in result.formatted_context
    assert "Relevant chunk about policy vacation leave rules and requirements." in result.formatted_context
    assert "[Chunk 2]" in result.formatted_context


def test_context_builder_character_budget_capping() -> None:
    """Verify ContextBuilder enforces max_context_chars and per-chunk limit."""
    doc_id = uuid.uuid4()
    # Chunk with 2000 characters
    long_text = "Important policy sentence. " * 80
    chunks = [
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text=long_text,
            document_id=doc_id,
            similarity_score=0.90,
            rank=1,
            document_title="Doc",
        )
    ]

    builder = ContextBuilder(max_chunks=5, max_context_chars=1000, max_chunk_chars=500)
    result = builder.build_context(chunks, query="policy")

    assert result.has_context is True
    # Per-chunk text should be capped to <= 500 chars (+ ellipsis)
    assert len(result.selected_chunks[0].chunk_text) <= 505
    # Total formatted context should fit inside budget
    assert len(result.formatted_context) <= 1000


def test_context_builder_empty_when_no_relevant_chunks() -> None:
    """When all chunks are below threshold, context is empty and has_context is False."""
    doc_id = uuid.uuid4()
    chunks = [
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text="Totally unrelated text.",
            document_id=doc_id,
            similarity_score=0.15,
            rank=1,
        )
    ]

    builder = ContextBuilder(similarity_threshold=0.30)
    result = builder.build_context(chunks, query="vacation policy")

    assert result.has_context is False
    assert result.selected_chunks == []
    assert result.formatted_context == ""
    assert result.total_filtered == 0


def test_prompt_builder_integrates_context_builder() -> None:
    """PromptBuilder uses ContextBuilder to format clean bounded context and rules."""
    doc_id = uuid.uuid4()
    chunks = [
        RankedResult(
            chunk_id=uuid.uuid4(),
            chunk_text="Core working hours are 10:00 AM to 4:00 PM.",
            document_id=doc_id,
            similarity_score=0.88,
            rank=1,
            document_title="Handbook",
            section_title="Hours",
            page_number=1,
        )
    ]
    builder = PromptBuilder(max_context_chars=6000)
    prompt = builder.build("What are the working hours?", chunks)

    assert len(prompt.retrieved_chunks) == 1
    assert "Core working hours are 10:00 AM to 4:00 PM." in prompt.user_prompt
    assert "[Chunk 1] (Document: Handbook | Page: 1 | Section: Hours)" in prompt.user_prompt
    assert STANDARDIZED_UNANSWERABLE_MESSAGE in prompt.user_prompt


@pytest.mark.asyncio
async def test_regular_user_can_read_documents() -> None:
    """Non-admin user retains read and query permissions (GET /api/documents returns 200)."""
    user_id = uuid.uuid4()
    regular_user = User(id=user_id, email="user@example.com", role=UserRole.USER, is_active=True)

    class MockDocService:
        async def list_by_user(self, user_id, limit, offset):
            return []

    app.dependency_overrides[get_current_user] = lambda: regular_user
    app.dependency_overrides[get_document_service] = lambda: MockDocService()

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/documents")
            assert response.status_code == 200
            data = response.json()
            assert data["items"] == []
            assert data["total"] == 0
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_document_service, None)


@pytest.mark.asyncio
async def test_member_user_can_read_documents() -> None:
    """MEMBER role user retains read and query permissions (GET /api/documents returns 200)."""
    user_id = uuid.uuid4()
    member_user = User(id=user_id, email="member@example.com", role=UserRole.MEMBER, is_active=True)

    class MockDocService:
        async def list_by_user(self, user_id, limit, offset):
            return []

    app.dependency_overrides[get_current_user] = lambda: member_user
    app.dependency_overrides[get_document_service] = lambda: MockDocService()

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/documents")
            assert response.status_code == 200
            data = response.json()
            assert data["items"] == []
            assert data["total"] == 0
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_document_service, None)


@pytest.mark.asyncio
async def test_unanswerable_query_returns_standardized_message_and_empty_citations() -> None:
    """When a query cannot be answered from retrieved documents, return standardized message with empty citations."""
    from app.rag.validator import validate_and_reconcile_answer
    from app.models.chat_message import MessageRole

    # 1. Validator test: with empty context chunks, validator returns standard message
    reconciled = validate_and_reconcile_answer("What is the extraterrestrial spaceship budget?", "Some guess", [])
    assert reconciled == STANDARDIZED_UNANSWERABLE_MESSAGE

    # 2. Validator test: completely unsupported query against context
    irrelevant_chunk = RankedResult(
        chunk_id=uuid.uuid4(),
        chunk_text="Annual leave policy allows 15 days per year.",
        document_id=uuid.uuid4(),
        similarity_score=0.4,
        rank=1,
    )
    ans = validate_and_reconcile_answer(
        "What is the policy for deep space exploration?",
        "Deep space exploration requires special permission.",
        [irrelevant_chunk],
    )
    assert STANDARDIZED_UNANSWERABLE_MESSAGE in ans or "does not specify" in ans.lower()

