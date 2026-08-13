-- =============================================================================
-- 008_rag_traces.sql
-- Local RAG ("Talk to My Data") — durable RAG request trace persistence
-- =============================================================================

CREATE TABLE IF NOT EXISTS rag_traces (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id              TEXT NOT NULL,
    user_id                 UUID REFERENCES users(id) ON DELETE SET NULL,
    session_id              UUID REFERENCES chat_sessions(id) ON DELETE SET NULL,
    original_query          TEXT NOT NULL,
    normalized_query        TEXT,
    detected_intent         TEXT,
    selected_route          TEXT,
    retrieval_start         TIMESTAMPTZ,
    retrieval_end           TIMESTAMPTZ,
    retrieval_duration_ms   INT NOT NULL DEFAULT 0,
    retrieved_chunk_ids     JSONB NOT NULL DEFAULT '[]'::jsonb,
    retrieved_document_ids  JSONB NOT NULL DEFAULT '[]'::jsonb,
    document_version_ids    JSONB NOT NULL DEFAULT '[]'::jsonb,
    similarity_scores       JSONB NOT NULL DEFAULT '[]'::jsonb,
    document_metadata       JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding_duration_ms   INT NOT NULL DEFAULT 0,
    llm_duration_ms         INT NOT NULL DEFAULT 0,
    total_duration_ms       INT NOT NULL DEFAULT 0,
    token_usage             JSONB,
    fallback_info           TEXT,
    error_type              TEXT,
    status                  TEXT NOT NULL DEFAULT 'SUCCESS',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_rag_traces_request_id ON rag_traces(request_id);
CREATE INDEX IF NOT EXISTS ix_rag_traces_session_id ON rag_traces(session_id);
CREATE INDEX IF NOT EXISTS ix_rag_traces_user_id ON rag_traces(user_id);
CREATE INDEX IF NOT EXISTS ix_rag_traces_created_at ON rag_traces(created_at DESC);

COMMENT ON TABLE rag_traces IS 'Durable request execution traces for RAG pipeline auditing, debugging, and evaluation.';
