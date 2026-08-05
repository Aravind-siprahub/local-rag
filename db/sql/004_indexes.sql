-- =============================================================================
-- 004_indexes.sql
-- Performance indexes for Local RAG
-- Requires: 003_tables.sql
-- =============================================================================

-- -----------------------------------------------------------------------------
-- users
-- -----------------------------------------------------------------------------
-- This is the ONLY uniqueness enforcement on email (003_tables.sql
-- deliberately has no table-level UNIQUE(email)). Scoping it to
-- WHERE deleted_at IS NULL means a soft-deleted user's email can be
-- reused by a new signup, while still preventing two active users from
-- sharing an email. Also serves as the login-lookup index.
CREATE UNIQUE INDEX IF NOT EXISTS users_email_active_uidx ON users (email) WHERE deleted_at IS NULL;

-- -----------------------------------------------------------------------------
-- documents — "my documents" list screen, filtered by owner and status
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS documents_user_id_idx ON documents (user_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS documents_user_status_idx ON documents (user_id, status) WHERE deleted_at IS NULL;
-- Tag filtering (e.g. "show contracts") via GIN on the array column.
CREATE INDEX IF NOT EXISTS documents_tags_gin_idx ON documents USING GIN (tags);
-- Trigram index powers ILIKE '%term%' title search in the document picker.
CREATE INDEX IF NOT EXISTS documents_title_trgm_idx ON documents USING GIN (title gin_trgm_ops);

-- -----------------------------------------------------------------------------
-- document_versions — pipeline dashboards & "get latest version for doc"
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS document_versions_document_id_idx ON document_versions (document_id);
-- Worker queue: "find versions stuck in a given stage" / status dashboards.
CREATE INDEX IF NOT EXISTS document_versions_status_idx ON document_versions (status);
CREATE INDEX IF NOT EXISTS document_versions_checksum_idx ON document_versions (checksum_sha256);

-- -----------------------------------------------------------------------------
-- document_chunks — retrieval joins back to source document, and reading a
-- document's chunks in order for display/debugging
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS document_chunks_version_id_idx ON document_chunks (document_version_id);
CREATE INDEX IF NOT EXISTS document_chunks_version_order_idx ON document_chunks (document_version_id, chunk_index);
-- Full-text/keyword fallback search (hybrid search alongside vector search).
CREATE INDEX IF NOT EXISTS document_chunks_content_trgm_idx ON document_chunks USING GIN (content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS document_chunks_metadata_gin_idx ON document_chunks USING GIN (metadata);

-- -----------------------------------------------------------------------------
-- embeddings — the hot path: ANN vector search + scoping by chunk/model
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS embeddings_chunk_id_idx ON embeddings (chunk_id);
CREATE INDEX IF NOT EXISTS embeddings_model_name_idx ON embeddings (model_name);

-- HNSW index for cosine-distance ANN search (pgvector >= 0.5).
-- HNSW is chosen over IVFFlat because it doesn't need a training/build pass
-- proportional to data size and gives better recall/latency at query time,
-- which matters for an interactive chat UI. Rebuild/tune m & ef_construction
-- as row counts grow (see docs/DATABASE_DESIGN.md "Scaling").
CREATE INDEX IF NOT EXISTS embeddings_embedding_hnsw_cosine_idx
    ON embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- -----------------------------------------------------------------------------
-- chat_sessions — "my chats" list, most-recent-first
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS chat_sessions_user_id_idx ON chat_sessions (user_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS chat_sessions_user_recent_idx ON chat_sessions (user_id, last_message_at DESC) WHERE deleted_at IS NULL;

-- -----------------------------------------------------------------------------
-- chat_messages — loading a session's transcript in order
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS chat_messages_session_id_idx ON chat_messages (session_id);
CREATE INDEX IF NOT EXISTS chat_messages_session_created_idx ON chat_messages (session_id, created_at);

-- -----------------------------------------------------------------------------
-- citations — "show sources" for a message, and reverse lookup
-- ("which conversations cited this chunk")
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS citations_message_id_idx ON citations (message_id);
CREATE INDEX IF NOT EXISTS citations_chunk_id_idx ON citations (chunk_id);

-- -----------------------------------------------------------------------------
-- processing_jobs — worker polling queue and per-document status rollups
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS processing_jobs_version_id_idx ON processing_jobs (document_version_id);
-- Partial index: workers only ever poll for pending/running jobs, so indexing
-- the full status column wastes space once most jobs are 'completed'.
CREATE INDEX IF NOT EXISTS processing_jobs_active_status_idx ON processing_jobs (status, created_at)
    WHERE status IN ('pending', 'running');
