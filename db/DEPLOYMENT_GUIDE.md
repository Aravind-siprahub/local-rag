# Local RAG — Production Deployment Guide (Supabase PostgreSQL)

This is the single authoritative guide for deploying the schema in
[`db/sql/`](sql) to your Supabase PostgreSQL project. It supersedes ad-hoc
notes — follow this document top to bottom.

---

## 0. Pre-deployment review — what was checked and fixed

Every file in `db/sql/` was reviewed for execution order, cross-table
dependencies, foreign key targets, pgvector setup, index validity, and
PostgreSQL 17 compatibility. Two real problems were found and fixed:

| # | File | Problem | Fix |
|---|---|---|---|
| 1 | `003_tables.sql` | `users` had a table-level `UNIQUE (email)` **and** `004_indexes.sql` had a partial unique index `WHERE deleted_at IS NULL` with a comment claiming soft-deleted emails could be reused. The full constraint made that false — a deleted user's email was permanently blocked. | Removed the table-level `UNIQUE (email)`. The partial index in `004_indexes.sql` is now the *only* uniqueness enforcement on email, correctly scoped to active (non-deleted) users. |
| 2 | `002_enums.sql`, `003_tables.sql`, `004_indexes.sql` | `CREATE TYPE` and plain `CREATE TABLE`/`CREATE TRIGGER` are not idempotent — re-running the scripts after a partial failure (e.g. the Supabase SQL Editor times out mid-script) would fail with "already exists" errors instead of completing. | Enums wrapped in `DO $$ ... EXCEPTION WHEN duplicate_object ... $$` blocks; all `CREATE TABLE` → `CREATE TABLE IF NOT EXISTS`; all `CREATE INDEX` → `CREATE INDEX IF NOT EXISTS`; triggers use `CREATE OR REPLACE TRIGGER` (PostgreSQL 14+, safe on Supabase's PG17). The deferred `documents_current_version_fk` constraint is wrapped the same way. |

Everything else checked out:

- **Execution order / dependencies:** extensions → enums → tables → indexes is correct. Every FK target table is created before the table that references it (`documents` → `document_versions` → `document_chunks` → `embeddings`; `chat_sessions` → `chat_messages` → `citations`). The one circular reference (`documents.current_version_id` ↔ `document_versions.id`) is correctly deferred to a post-creation `ALTER TABLE`.
- **pgvector:** `CREATE EXTENSION IF NOT EXISTS vector;` runs before any `VECTOR(768)` column or `vector_cosine_ops`/HNSW index — correct order. Supabase ships pgvector pre-approved for `CREATE EXTENSION`.
- **Indexes:** every index targets a column that exists on a table created earlier in the sequence; the HNSW index (`embeddings_embedding_hnsw_cosine_idx`) uses valid pgvector syntax (`USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)`), requires pgvector ≥ 0.5.0 (verified in step 4 below).
- **PostgreSQL 17 compatibility:** generated stored columns (`total_tokens`), `CITEXT`, `JSONB` defaults, `gen_random_uuid()`, `CREATE OR REPLACE TRIGGER`, and all constraint syntax are supported on PG17/Supabase without changes.
- **Production hygiene:** `001`–`004` contain no seed data, test rows, or sample queries. `005_example_queries.sql` is `SELECT`-only reference material for the application layer and is explicitly excluded from deployment (now marked with a banner in the file itself).

---

## 1. Prerequisites

- A Supabase project already created (you have this).
- Access to **Supabase Dashboard → SQL Editor** for your project.
- The four files, in this exact order, from `db/sql/`:
  1. `001_extensions.sql`
  2. `002_enums.sql`
  3. `003_tables.sql`
  4. `004_indexes.sql`
- **Do not run** `005_example_queries.sql` — it's reference material, not a deployment step, and now says so at the top of the file.

---

## 2. Execution order (must be sequential, not parallel)

Each file depends on objects created by the one before it — running them out
of order will fail:

```
001_extensions.sql   →  installs vector, pgcrypto, citext, pg_trgm, btree_gin
002_enums.sql         →  creates enum types used as column types in 003
003_tables.sql         →  creates all 10 tables + FKs + triggers (needs 001 + 002)
004_indexes.sql         →  creates all indexes (needs 003's tables/columns to exist)
```

---

## 3. Step-by-step: running in the Supabase SQL Editor

1. Open your project at **supabase.com/dashboard** → select the project →
   left sidebar → **SQL Editor**.
2. Click **"New query"**.
3. Open [`db/sql/001_extensions.sql`](sql/001_extensions.sql) locally, copy
   its **entire contents**, paste into the SQL Editor.
4. Click **Run** (or `Ctrl+Enter` / `Cmd+Enter`).
   - Expected result: `Success. No rows returned.`
5. Click **"New query"** again (fresh tab keeps errors from one file from
   being confused with another). Repeat steps 3–4 for:
   - [`db/sql/002_enums.sql`](sql/002_enums.sql)
   - [`db/sql/003_tables.sql`](sql/003_tables.sql)
   - [`db/sql/004_indexes.sql`](sql/004_indexes.sql)
6. Do **not** run `005_example_queries.sql`.

If any statement in a file fails partway through, Supabase's SQL Editor runs
each query in its own implicit transaction *per statement* in most cases —
because every statement in `002`–`004` is now idempotent (`IF NOT EXISTS` /
`DO $$ ... EXCEPTION ...`), you can safely re-run the entire file from the
top after fixing the failing statement, rather than hand-tracking which
objects already exist.

**Alternative — Supabase CLI (if you have it installed locally):**
```bash
supabase db execute --file db/sql/001_extensions.sql
supabase db execute --file db/sql/002_enums.sql
supabase db execute --file db/sql/003_tables.sql
supabase db execute --file db/sql/004_indexes.sql
```
(Do not `supabase db execute --file db/sql/005_example_queries.sql`.)

---

## 4. Post-deployment verification

Run each of these in a fresh SQL Editor query after all four files complete.

**4.1 — Extensions installed:**
```sql
SELECT extname, extversion
FROM pg_extension
WHERE extname IN ('vector', 'pgcrypto', 'citext', 'pg_trgm', 'btree_gin')
ORDER BY extname;
```
Expect 5 rows. For `vector`, confirm `extversion >= 0.5.0` (required for HNSW
indexes) — Supabase typically ships a current version, but check if
`004_indexes.sql` failed on the HNSW `CREATE INDEX` statement.

**4.2 — Enum types created:**
```sql
SELECT t.typname AS enum_name, string_agg(e.enumlabel, ', ' ORDER BY e.enumsortorder) AS values
FROM pg_type t
JOIN pg_enum e ON e.enumtypid = t.oid
WHERE t.typname IN (
    'user_role', 'document_status', 'document_version_status',
    'processing_job_type', 'processing_job_status', 'message_role', 'vector_metric'
)
GROUP BY t.typname
ORDER BY t.typname;
```
Expect 7 rows, each with its full value list.

**4.3 — All tables created:**
```sql
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN (
      'users', 'documents', 'document_versions', 'document_chunks',
      'embeddings', 'chat_sessions', 'chat_messages', 'citations',
      'processing_jobs', 'system_settings'
  )
ORDER BY table_name;
```
Expect exactly 10 rows.

**4.4 — Foreign keys wired correctly (including the deferred one):**
```sql
SELECT
    tc.table_name, kcu.column_name,
    ccu.table_name AS references_table, ccu.column_name AS references_column
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu ON kcu.constraint_name = tc.constraint_name
JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public'
ORDER BY tc.table_name, kcu.column_name;
```
Confirm `documents.current_version_id → document_versions.id` appears — this
is the one added via the deferred `ALTER TABLE` and is the easiest to miss if
`003_tables.sql` was interrupted partway through.

**4.5 — All indexes created, including the HNSW vector index:**
```sql
SELECT indexname, tablename, indexdef
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;
```
Specifically confirm the vector index exists and used `hnsw`:
```sql
SELECT indexname, indexdef
FROM pg_indexes
WHERE indexname = 'embeddings_embedding_hnsw_cosine_idx';
```
`indexdef` should contain `USING hnsw (embedding vector_cosine_ops)`.

**4.6 — Email uniqueness is scoped correctly (regression check for the fix in §0):**
```sql
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'users' AND indexname = 'users_email_active_uidx';
```
`indexdef` should show `WHERE (deleted_at IS NULL)` — confirming there is
**no** unscoped unique constraint on `email` fighting it.

**4.7 — Triggers active:**
```sql
SELECT event_object_table, trigger_name
FROM information_schema.triggers
WHERE trigger_schema = 'public'
ORDER BY event_object_table;
```
Expect 5 rows (`users`, `documents`, `document_versions`, `chat_sessions`,
`processing_jobs`), each named `trg_<table>_updated_at`.

**4.8 — End-to-end smoke test (optional, safe to run and easy to clean up):**
```sql
BEGIN;
INSERT INTO users (email, hashed_password) VALUES ('smoketest@example.com', 'x')
RETURNING id, email, role, created_at;
ROLLBACK;  -- discards the test row, verifies INSERT/DEFAULT/trigger wiring without leaving data behind
```
If this returns one row with `role = 'member'` and a populated `created_at`,
the enum default, `gen_random_uuid()`, and column defaults are all working.

---

## 5. If something fails

- **`type "..." already exists` / `relation "..." already exists`:** you're
  re-running a file that partially succeeded before this guide's idempotency
  fixes were applied to your local copy — re-copy the current version of the
  file and re-run; it's now safe to run twice.
- **`extension "vector" is not available`:** contact Supabase support or
  check Dashboard → Database → Extensions — pgvector should be pre-approved
  on all Supabase projects, but a very old project may need it enabled from
  the dashboard UI once before `CREATE EXTENSION` succeeds.
- **HNSW `CREATE INDEX` fails with a syntax/operator class error:** your
  project's pgvector version predates 0.5.0 (check with the query in §4.1);
  upgrade the extension via Dashboard → Database → Extensions, or fall back
  to `USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)` as a
  temporary substitute (has different scaling/recall tradeoffs — see
  `docs/DATABASE_DESIGN.md` §6).
- **Any FK violation during the smoke test:** stop and re-check §4.3/§4.4 —
  it means a table or constraint from `003_tables.sql` didn't fully apply.

## 6. Next step

Once verification passes, point your FastAPI backend's `DATABASE_URL` (see
`backend/.env`) at this same Supabase project and hit `GET /health` — a
`{"status": "ok", "database": "connected"}` response confirms the app can
reach the schema you just deployed.
