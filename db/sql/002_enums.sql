-- =============================================================================
-- 002_enums.sql
-- PostgreSQL enum types for Local RAG
--
-- Every CREATE TYPE is wrapped in a DO block that swallows "already exists"
-- (duplicate_object) so this file can be re-run safely if a deployment is
-- retried after a partial failure — CREATE TYPE has no native IF NOT EXISTS.
-- =============================================================================

-- Application-level role for authorization
DO $$ BEGIN
    CREATE TYPE user_role AS ENUM ('admin', 'member');
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Lifecycle of the logical "document" record (rolls up its versions)
DO $$ BEGIN
    CREATE TYPE document_status AS ENUM (
        'uploaded',      -- file stored, no processing started
        'processing',    -- at least one pipeline stage running
        'ready',         -- latest version fully indexed and queryable
        'failed',        -- latest version failed processing
        'archived'       -- user archived; excluded from retrieval by default
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Fine-grained pipeline status for a specific document_version
DO $$ BEGIN
    CREATE TYPE document_version_status AS ENUM (
        'uploaded',
        'parsing',
        'parsed',
        'chunking',
        'chunked',
        'embedding',
        'embedded',
        'indexing',
        'completed',
        'failed'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- processing_jobs.job_type — one row per pipeline stage attempt
DO $$ BEGIN
    CREATE TYPE processing_job_type AS ENUM (
        'upload',
        'parse',
        'chunk',
        'embed',
        'index'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- processing_jobs.status — generic job state machine
DO $$ BEGIN
    CREATE TYPE processing_job_status AS ENUM (
        'pending',
        'running',
        'completed',
        'failed',
        'cancelled'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- chat_messages.role — mirrors OpenAI/Ollama chat roles
DO $$ BEGIN
    CREATE TYPE message_role AS ENUM ('system', 'user', 'assistant');
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Distance metric an embedding column/index was built for (future-proofing
-- multi-metric support; cosine is the default for normalized embeddings)
DO $$ BEGIN
    CREATE TYPE vector_metric AS ENUM ('cosine', 'l2', 'inner_product');
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;
