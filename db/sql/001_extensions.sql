-- =============================================================================
-- 001_extensions.sql
-- Local RAG ("Talk to My Data") — PostgreSQL 17 extension setup
-- =============================================================================

-- pgvector: vector similarity search (cosine, L2, inner product)
CREATE EXTENSION IF NOT EXISTS vector;

-- gen_random_uuid() — built into pgcrypto (or core in PG13+, kept explicit for portability)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- citext: case-insensitive text, used for email uniqueness
CREATE EXTENSION IF NOT EXISTS citext;

-- pg_trgm: trigram indexes for fuzzy/ILIKE search over chunk text and titles
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- btree_gin: allows GIN indexes to cover scalar columns combined with jsonb/array columns
CREATE EXTENSION IF NOT EXISTS btree_gin;
