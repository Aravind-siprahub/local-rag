-- Migration 007: Add Supabase Storage metadata columns to documents and document_versions

ALTER TABLE documents
ADD COLUMN IF NOT EXISTS storage_provider VARCHAR(50) DEFAULT 'supabase',
ADD COLUMN IF NOT EXISTS bucket_name VARCHAR(100) DEFAULT 'documents',
ADD COLUMN IF NOT EXISTS storage_path VARCHAR(500),
ADD COLUMN IF NOT EXISTS last_error TEXT;

ALTER TABLE document_versions
ADD COLUMN IF NOT EXISTS storage_provider VARCHAR(50) DEFAULT 'supabase',
ADD COLUMN IF NOT EXISTS bucket_name VARCHAR(100) DEFAULT 'documents',
ADD COLUMN IF NOT EXISTS storage_path VARCHAR(500);

-- Backfill defaults for pre-existing rows
UPDATE documents
SET storage_provider = 'supabase', bucket_name = 'documents'
WHERE storage_provider IS NULL;

UPDATE document_versions
SET storage_provider = 'supabase', bucket_name = 'documents'
WHERE storage_provider IS NULL;
