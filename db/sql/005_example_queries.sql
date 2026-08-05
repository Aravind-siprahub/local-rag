-- =============================================================================
-- 005_example_queries.sql
-- Representative queries the FastAPI app issues against this schema
--
-- *** REFERENCE ONLY — DO NOT RUN THIS FILE AGAINST PRODUCTION ***
-- Every statement below uses :named placeholders (no bind values), is a
-- read-only SELECT, and depends on rows existing (a real user, real
-- documents/chunks/embeddings). It is not part of the schema deployment —
-- deploy only 001 through 004. Use these as copy/paste starting points from
-- the FastAPI service layer, or to sanity-check the schema in a scratch
-- session after seeding some test data.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Vector similarity search (cosine) scoped to one user's ready documents,
--    top-8 chunks — the core retrieval step of the RAG pipeline.
-- -----------------------------------------------------------------------------
-- :query_embedding is a 768-dim vector literal, e.g. '[0.012,-0.034,...]'
SET LOCAL hnsw.ef_search = 100; -- trade recall for latency per-query

SELECT
    dc.id            AS chunk_id,
    dc.content,
    dc.page_number,
    d.id             AS document_id,
    d.title          AS document_title,
    1 - (e.embedding <=> :query_embedding) AS similarity   -- <=> is cosine distance
FROM embeddings e
JOIN document_chunks dc      ON dc.id = e.chunk_id
JOIN document_versions dv    ON dv.id = dc.document_version_id
JOIN documents d              ON d.id = dv.document_id
WHERE d.user_id = :user_id
  AND d.deleted_at IS NULL
  AND d.status = 'ready'
  AND dv.id = d.current_version_id      -- only search the current version
  AND e.model_name = :active_model_name
ORDER BY e.embedding <=> :query_embedding
LIMIT 8;

-- -----------------------------------------------------------------------------
-- 2. Hybrid search: combine vector similarity with trigram keyword match
--    (useful when the query contains exact identifiers vector search misses).
-- -----------------------------------------------------------------------------
SELECT dc.id, dc.content,
       1 - (e.embedding <=> :query_embedding) AS vector_similarity,
       similarity(dc.content, :query_text)    AS text_similarity
FROM embeddings e
JOIN document_chunks dc ON dc.id = e.chunk_id
JOIN document_versions dv ON dv.id = dc.document_version_id
JOIN documents d ON d.id = dv.document_id
WHERE d.user_id = :user_id
  AND d.deleted_at IS NULL
  AND (dc.content % :query_text OR (e.embedding <=> :query_embedding) < 0.5)
ORDER BY (0.7 * (1 - (e.embedding <=> :query_embedding))
          + 0.3 * similarity(dc.content, :query_text)) DESC
LIMIT 8;

-- -----------------------------------------------------------------------------
-- 3. Load a chat session's full transcript with citations, in order.
-- -----------------------------------------------------------------------------
SELECT
    cm.id, cm.role, cm.content, cm.model_used,
    cm.prompt_tokens, cm.completion_tokens, cm.latency_ms, cm.created_at,
    COALESCE(
        jsonb_agg(
            jsonb_build_object(
                'chunk_id', c.chunk_id,
                'document_title', d.title,
                'page_number', dc.page_number,
                'similarity_score', c.similarity_score,
                'rank', c.rank
            ) ORDER BY c.rank
        ) FILTER (WHERE c.id IS NOT NULL),
        '[]'
    ) AS citations
FROM chat_messages cm
LEFT JOIN citations c        ON c.message_id = cm.id
LEFT JOIN document_chunks dc ON dc.id = c.chunk_id
LEFT JOIN document_versions dv ON dv.id = dc.document_version_id
LEFT JOIN documents d ON d.id = dv.document_id
WHERE cm.session_id = :session_id
GROUP BY cm.id
ORDER BY cm.created_at ASC;

-- -----------------------------------------------------------------------------
-- 4. User's document library with pipeline status (dashboard view).
-- -----------------------------------------------------------------------------
SELECT d.id, d.title, d.status, d.tags, d.created_at,
       dv.version_number, dv.page_count, dv.file_size_bytes,
       (SELECT count(*) FROM document_chunks dc WHERE dc.document_version_id = dv.id) AS chunk_count
FROM documents d
JOIN document_versions dv ON dv.id = d.current_version_id
WHERE d.user_id = :user_id AND d.deleted_at IS NULL
ORDER BY d.created_at DESC;

-- -----------------------------------------------------------------------------
-- 5. Pipeline stage failures needing attention (ops/admin view).
-- -----------------------------------------------------------------------------
SELECT pj.id, pj.job_type, pj.status, pj.error_message, pj.retry_count,
       dv.id AS document_version_id, d.title, u.email AS owner_email
FROM processing_jobs pj
JOIN document_versions dv ON dv.id = pj.document_version_id
JOIN documents d ON d.id = dv.document_id
JOIN users u ON u.id = d.user_id
WHERE pj.status = 'failed'
ORDER BY pj.updated_at DESC
LIMIT 50;

-- -----------------------------------------------------------------------------
-- 6. Recent chats for a user, most-recently-active first.
-- -----------------------------------------------------------------------------
SELECT id, title, last_message_at, created_at
FROM chat_sessions
WHERE user_id = :user_id AND deleted_at IS NULL AND is_archived = FALSE
ORDER BY last_message_at DESC NULLS LAST
LIMIT 20;
