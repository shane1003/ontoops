-- Migration 001 — source_cq_id → source_id (source-neutral provenance, NOT NULL).
--
-- WHY: provenance is generalized from CQ-only to a
-- source spine. abox_*.source_cq_id (FK → cq_repository) is replaced by source_id
-- (FK → source), which may denote a document CQ *or* an RDB ref. NOT NULL is retained.
-- abox_schema.sql already declares the converged shape for FRESH databases; this file is the
-- ALTER path for a database created under the OLD (source_cq_id) schema.
--
-- APPLIED AUTOMATICALLY by `app.aboxmapper.core.schema.apply_pending_migrations` (name order, once,
-- recorded in the aboxmapper_schema_migrations ledger). A fresh DB has this stamped as
-- already-applied by ensure_schema, so this runs ONLY on a pre-existing source_cq_id schema.
-- The runner owns the transaction — do NOT add BEGIN/COMMIT here.
--
-- DEPENDENCY: the tboxcraft schema (creates `source`, and sync_document_sources() derives its
-- rows from cq_repository via the same deterministic key) must be applied before
-- aboxmapper-migrate. The precondition below fails loud if `source` is missing.
--
-- Convergent-schema rule (ADD only, no RENAME): NOT NULL is 3-step
--   1. ADD source_id nullable  →  2. backfill  →  3. SET NOT NULL, then retire source_cq_id.

-- ── Precondition: the source registry (tboxcraft) must be present ───────────────────────
-- Resolved via search_path (unqualified) so the check follows wherever `source` lives.
DO $$
BEGIN
    IF to_regclass('source') IS NULL THEN
        RAISE EXCEPTION 'source registry missing: apply the tboxcraft `source` schema in this '
            'database before this migration (FKs cannot cross databases).';
    END IF;
END $$;

-- ── Step 1 — ADD source_id (nullable for now), FK → source ──────────────────────────────
ALTER TABLE abox_individual
    ADD COLUMN IF NOT EXISTS source_id varchar REFERENCES source(source_id) ON DELETE RESTRICT;
ALTER TABLE abox_data_assertion
    ADD COLUMN IF NOT EXISTS source_id varchar REFERENCES source(source_id) ON DELETE RESTRICT;

-- ── Step 2 — BACKFILL (policy DECIDED) ──────────────────────────────────────────────────
-- Existing rows carry the legacy source_cq_id (a document CQ). The deterministic key
-- 'doc:cq:' || cq_id is shared by adapter and core; tboxcraft's sync_document_sources()
-- already derives the matching `source` rows from cq_repository (ON CONFLICT DO NOTHING), so
-- aboxmapper only points source_id at that key — it does NOT INSERT into `source` (that would
-- only risk duplicates). The compose order guarantees those source rows exist by now.
UPDATE abox_individual
    SET source_id = 'doc:cq:' || source_cq_id
    WHERE source_cq_id IS NOT NULL AND source_id IS NULL;
UPDATE abox_data_assertion
    SET source_id = 'doc:cq:' || source_cq_id
    WHERE source_cq_id IS NOT NULL AND source_id IS NULL;

-- ── Guard — refuse to proceed while any source_id is still NULL ──────────────────────────
-- Only rows whose source_cq_id was already NULL remain (un-backfillable). The conceptualizer
-- ships source_cq_id on every individual, so this is expected to be empty; if not, fail
-- loud rather than coerce — decide the un-backfillable-row policy (back-trace / delete) first.
DO $$
DECLARE
    n_ind bigint;
    n_val bigint;
BEGIN
    SELECT count(*) INTO n_ind FROM abox_individual     WHERE source_id IS NULL;
    SELECT count(*) INTO n_val FROM abox_data_assertion WHERE source_id IS NULL;
    IF n_ind > 0 OR n_val > 0 THEN
        RAISE EXCEPTION
            'Un-backfillable rows: % abox_individual and % abox_data_assertion have no '
            'source_cq_id to derive source_id from. Decide the policy before SET NOT NULL.',
            n_ind, n_val;
    END IF;
END $$;

-- ── Step 3a — re-point identity_key at source_id, drop the now-trivial CHECK ─────────────
-- identity_key is a STORED generated column referencing source_cq_id, and the unique key and
-- the loader's ON CONFLICT depend on it — so drop/redefine in order.
ALTER TABLE abox_individual DROP CONSTRAINT IF EXISTS uq_abox_individual_identity;
ALTER TABLE abox_individual DROP CONSTRAINT IF EXISTS chk_abox_individual_identifiable;
ALTER TABLE abox_individual DROP COLUMN IF EXISTS identity_key;
ALTER TABLE abox_individual ADD COLUMN identity_key text
    GENERATED ALWAYS AS (COALESCE(canonical_name, 'src:' || source_id)) STORED;
ALTER TABLE abox_individual ADD CONSTRAINT uq_abox_individual_identity
    UNIQUE (class_id, identity_key);

-- ── Step 3b — SET NOT NULL ──────────────────────────────────────────────────────────────
ALTER TABLE abox_individual     ALTER COLUMN source_id SET NOT NULL;
ALTER TABLE abox_data_assertion ALTER COLUMN source_id SET NOT NULL;

-- ── Step 4 — retire the legacy column (end of coexistence) ──────────────────────────────
ALTER TABLE abox_individual     DROP COLUMN IF EXISTS source_cq_id;
ALTER TABLE abox_data_assertion DROP COLUMN IF EXISTS source_cq_id;
