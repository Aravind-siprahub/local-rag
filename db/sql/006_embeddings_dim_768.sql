-- =============================================================================
-- 006_embeddings_dim_768.sql
-- Align embeddings.embedding with nomic-embed-text (768-d).
-- Safe when embeddings is empty; truncate first if old 1024-d rows exist.
-- Requires: 003_tables.sql, 004_indexes.sql
-- =============================================================================

DROP INDEX IF EXISTS embeddings_embedding_hnsw_cosine_idx;

ALTER TABLE embeddings DROP CONSTRAINT IF EXISTS embeddings_dimensions_chk;

ALTER TABLE embeddings
    ALTER COLUMN embedding TYPE vector(768);

ALTER TABLE embeddings
    ADD CONSTRAINT embeddings_dimensions_chk CHECK (dimensions = 768);

CREATE INDEX embeddings_embedding_hnsw_cosine_idx
    ON embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
