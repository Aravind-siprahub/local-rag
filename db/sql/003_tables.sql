-- =============================================================================
-- 003_tables.sql
-- Local RAG ("Talk to My Data") — core schema
-- Requires: 001_extensions.sql, 002_enums.sql
-- =============================================================================

-- -----------------------------------------------------------------------------
-- users
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email               CITEXT NOT NULL,
    hashed_password     TEXT NOT NULL,
    full_name           TEXT,
    role                user_role NOT NULL DEFAULT 'member',
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,
    is_verified         BOOLEAN NOT NULL DEFAULT FALSE,
    last_login_at       TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at          TIMESTAMPTZ,

    -- No table-level UNIQUE(email) here on purpose: uniqueness is enforced
    -- by the *partial* unique index users_email_active_uidx in
    -- 004_indexes.sql (WHERE deleted_at IS NULL) so a soft-deleted user's
    -- email can be reused by a new account. A plain UNIQUE constraint would
    -- block that regardless of deleted_at.
    CONSTRAINT users_email_format_chk CHECK (email ~ '^[^@\s]+@[^@\s]+\.[^@\s]+$')
);

COMMENT ON TABLE users IS 'Application accounts. Soft-deleted via deleted_at to preserve FK history in documents/chats.';

-- -----------------------------------------------------------------------------
-- documents  (logical document owned by a user; wraps 1..N versions)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title               TEXT NOT NULL,
    description         TEXT,
    status              document_status NOT NULL DEFAULT 'uploaded',
    current_version_id  UUID,  -- FK added after document_versions exists (circular ref)
    tags                TEXT[] NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at          TIMESTAMPTZ,

    CONSTRAINT documents_title_not_blank_chk CHECK (btrim(title) <> '')
);

COMMENT ON TABLE documents IS 'Logical document. Re-uploads of the same file create a new document_versions row rather than a new document.';

-- -----------------------------------------------------------------------------
-- document_versions  (one immutable file per version; reprocessing lives here)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_versions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id         UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    version_number      INTEGER NOT NULL,
    storage_key         TEXT NOT NULL,          -- path/key in object storage (S3, local disk, etc.)
    original_filename   TEXT NOT NULL,
    mime_type           TEXT NOT NULL,
    file_size_bytes     BIGINT NOT NULL,
    checksum_sha256     CHAR(64) NOT NULL,
    page_count          INTEGER,
    status              document_version_status NOT NULL DEFAULT 'uploaded',
    error_message       TEXT,
    uploaded_by         UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    parsed_at           TIMESTAMPTZ,
    chunked_at          TIMESTAMPTZ,
    embedded_at         TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT document_versions_version_positive_chk CHECK (version_number > 0),
    CONSTRAINT document_versions_size_positive_chk CHECK (file_size_bytes > 0),
    CONSTRAINT document_versions_doc_version_unique UNIQUE (document_id, version_number)
);

COMMENT ON TABLE document_versions IS 'One row per uploaded file. Immutable once uploaded; re-processing updates status, not content.';

DO $$ BEGIN
    ALTER TABLE documents
        ADD CONSTRAINT documents_current_version_fk
        FOREIGN KEY (current_version_id) REFERENCES document_versions(id) ON DELETE SET NULL;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- -----------------------------------------------------------------------------
-- document_chunks  (text spans produced by the chunking stage)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_chunks (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_version_id  UUID NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
    chunk_index           INTEGER NOT NULL,
    content                TEXT NOT NULL,
    content_tokens         INTEGER,
    page_number            INTEGER,
    section_title          TEXT,
    char_start              INTEGER,
    char_end                INTEGER,
    metadata                JSONB NOT NULL DEFAULT '{}'::JSONB,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT document_chunks_index_unique UNIQUE (document_version_id, chunk_index),
    CONSTRAINT document_chunks_content_not_blank_chk CHECK (btrim(content) <> ''),
    CONSTRAINT document_chunks_span_chk CHECK (char_end IS NULL OR char_start IS NULL OR char_end >= char_start)
);

COMMENT ON TABLE document_chunks IS 'Immutable text spans. One document_version fans out into many chunks; chunk_index preserves document order.';

-- -----------------------------------------------------------------------------
-- embeddings  (vector representation of a chunk; supports multiple models/chunk)
-- -----------------------------------------------------------------------------
-- NOTE ON DIMENSIONALITY: pgvector requires a fixed dimension per column.
-- 768 matches nomic-embed-text (the local embedding model via Ollama).
-- See docs/DATABASE_DESIGN.md "Future Extensibility" for how to add a second
-- embedding table (e.g. embeddings_1536) if a higher-dimension model is added
-- later without migrating existing vectors.
CREATE TABLE IF NOT EXISTS embeddings (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id            UUID NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    model_name          TEXT NOT NULL,
    model_version       TEXT,
    dimensions           INTEGER NOT NULL,
    metric               vector_metric NOT NULL DEFAULT 'cosine',
    embedding             VECTOR(768) NOT NULL,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT embeddings_chunk_model_unique UNIQUE (chunk_id, model_name),
    CONSTRAINT embeddings_dimensions_chk CHECK (dimensions = 768)
);

COMMENT ON TABLE embeddings IS 'Vector embeddings per chunk per model. Unique per (chunk, model) so re-embedding with a new model coexists with the old vectors during migration.';

-- -----------------------------------------------------------------------------
-- chat_sessions
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chat_sessions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title               TEXT NOT NULL DEFAULT 'New chat',
    is_archived         BOOLEAN NOT NULL DEFAULT FALSE,
    last_message_at     TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at          TIMESTAMPTZ
);

COMMENT ON TABLE chat_sessions IS 'A conversation thread belonging to exactly one user.';

-- -----------------------------------------------------------------------------
-- chat_messages
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chat_messages (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id          UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role                message_role NOT NULL,
    content             TEXT NOT NULL,
    model_used          TEXT,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    total_tokens        INTEGER GENERATED ALWAYS AS (COALESCE(prompt_tokens, 0) + COALESCE(completion_tokens, 0)) STORED,
    latency_ms          INTEGER,
    generation_time_ms  INTEGER,
    error_message       TEXT,
    attachments         JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT chat_messages_content_not_blank_chk CHECK (btrim(content) <> '')
);

COMMENT ON TABLE chat_messages IS 'One row per turn. Assistant turns carry model/token/latency telemetry; retrieval provenance lives in citations.';

-- -----------------------------------------------------------------------------
-- citations  (links an assistant message to the chunks used to answer it)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS citations (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id          UUID NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    chunk_id            UUID NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    similarity_score     DOUBLE PRECISION,
    rank                  INTEGER NOT NULL,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT citations_message_chunk_unique UNIQUE (message_id, chunk_id),
    CONSTRAINT citations_rank_positive_chk CHECK (rank > 0)
);

COMMENT ON TABLE citations IS 'Retrieval provenance: which chunks were shown to the model for a given assistant message, in retrieval-rank order.';

-- -----------------------------------------------------------------------------
-- processing_jobs  (audit trail / retry queue for the ingestion pipeline)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS processing_jobs (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_version_id  UUID NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
    job_type               processing_job_type NOT NULL,
    status                  processing_job_status NOT NULL DEFAULT 'pending',
    error_message           TEXT,
    retry_count              INTEGER NOT NULL DEFAULT 0,
    started_at                TIMESTAMPTZ,
    completed_at               TIMESTAMPTZ,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                    TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT processing_jobs_retry_nonneg_chk CHECK (retry_count >= 0)
);

COMMENT ON TABLE processing_jobs IS 'One row per attempt of a pipeline stage (parse/chunk/embed/index). Drives worker retries and surfaces failure reasons to the UI.';

-- -----------------------------------------------------------------------------
-- system_settings  (optional key/value config, admin-editable)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS system_settings (
    key                 TEXT PRIMARY KEY,
    value               JSONB NOT NULL,
    description         TEXT,
    updated_by          UUID REFERENCES users(id) ON DELETE SET NULL,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE system_settings IS 'Runtime-configurable settings (default chunk size, active embedding model, retrieval top_k, etc.) without a redeploy.';

-- -----------------------------------------------------------------------------
-- updated_at trigger (shared across tables that track it)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- CREATE OR REPLACE TRIGGER (PostgreSQL 14+) makes this block idempotent —
-- safe to re-run if a deployment is retried after a partial failure.
CREATE OR REPLACE TRIGGER trg_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE OR REPLACE TRIGGER trg_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE OR REPLACE TRIGGER trg_document_versions_updated_at BEFORE UPDATE ON document_versions
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE OR REPLACE TRIGGER trg_chat_sessions_updated_at BEFORE UPDATE ON chat_sessions
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE OR REPLACE TRIGGER trg_processing_jobs_updated_at BEFORE UPDATE ON processing_jobs
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
