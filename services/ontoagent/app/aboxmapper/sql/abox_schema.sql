-- ABox mapping layer — owned by this module (aboxmapper).
--
-- The TBox side (tbox_term, surface_form, tbox_hierarchy, tbox_domain_range, the `source`
-- registry, ...) is owned by the TBox pipeline and lives in the SAME database (Postgres FKs
-- can't cross databases); this module only reads it. Everything below is written by the loader.
--
-- Scope (deliberate):
--   * class typing + data-property values only — object properties are NOT stored here
--     (they live in the TBox TTL; no abox_object_assertion table by design)
--   * no embeddings — this module does not compute them
--   * no evidence/confidence fields — provenance is a single mandatory FK to `source`.
--     Provenance is source-neutral: source_id -> source(source_id),
--     which may denote a document CQ or an RDB ref. "nothing is produced without a source" —
--     source_id is NOT NULL on both tables, so every individual/assertion is traceable.
--
-- Apply (idempotent — safe to re-run): `aboxmapper-migrate`, or by hand:
--   psql "$TBOXCRAFT_DATABASE_URL" -f app/aboxmapper/sql/abox_schema.sql

-- ---------------------------------------------------------------- individuals

-- `id` is an opaque IRI (<ontology base>/individual/<uuid>): stable across renames and
-- reclassification, per standard practice (Cool URIs don't change).
--
-- Identity for dedup is NOT the IRI but `identity_key`:
--   * named individual   -> canonical_name        (same entity merges across sources)
--   * unnamed individual -> 'src:' || source_id   (one per class per source)
-- The upstream extractor is expected to fill canonical_name; the src: fallback keeps
-- value-bearing individuals loadable when it does not (source_id is always present).
CREATE TABLE IF NOT EXISTS abox_individual (
    id             text PRIMARY KEY,
    class_id       text NOT NULL REFERENCES tbox_term(id) ON DELETE RESTRICT,
    canonical_name text,
    source_id      varchar NOT NULL REFERENCES source(source_id) ON DELETE RESTRICT,
    status         varchar(20) NOT NULL DEFAULT 'ACTIVE'
                     CHECK (status IN ('ACTIVE', 'MERGED', 'RETIRED')),
    merged_into_id text REFERENCES abox_individual(id) ON DELETE SET NULL,
    identity_key   text GENERATED ALWAYS AS
                     (COALESCE(canonical_name, 'src:' || source_id)) STORED,
    created_at     timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT uq_abox_individual_identity UNIQUE (class_id, identity_key),
    -- (no chk_abox_individual_identifiable: source_id is NOT NULL, so identity_key's
    --  'src:' || source_id fallback is always available — the old CHECK is trivially true.)
    CONSTRAINT chk_abox_individual_merged
        CHECK ((status = 'MERGED') = (merged_into_id IS NOT NULL)),
    CONSTRAINT chk_abox_individual_no_self_merge
        CHECK (merged_into_id IS NULL OR merged_into_id <> id)
);

CREATE INDEX IF NOT EXISTS idx_abox_individual_class ON abox_individual (class_id);
CREATE INDEX IF NOT EXISTS idx_abox_individual_merged ON abox_individual (merged_into_id)
    WHERE merged_into_id IS NOT NULL;

-- ---------------------------------------------------------------- data assertions

-- One row per (individual, data property, value). The UNIQUE key is what makes the loader
-- idempotent: re-running the same input is ON CONFLICT DO NOTHING.
CREATE TABLE IF NOT EXISTS abox_data_assertion (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    subject_id   text NOT NULL REFERENCES abox_individual(id) ON DELETE CASCADE,
    data_prop_id text NOT NULL REFERENCES tbox_term(id) ON DELETE RESTRICT,
    value        text NOT NULL,
    xsd_datatype varchar(50) NOT NULL DEFAULT 'xsd:string',  -- from tbox_domain_range RANGE
    source_id    varchar NOT NULL REFERENCES source(source_id) ON DELETE RESTRICT,
    status       varchar(20) NOT NULL DEFAULT 'ACTIVE'
                   CHECK (status IN ('ACTIVE', 'RETIRED')),
    created_at   timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT uq_abox_data_assertion UNIQUE (subject_id, data_prop_id, value)
);

CREATE INDEX IF NOT EXISTS idx_abox_assertion_subject ON abox_data_assertion (subject_id);
CREATE INDEX IF NOT EXISTS idx_abox_assertion_prop ON abox_data_assertion (data_prop_id);

-- ---------------------------------------------------------------- term-kind guards

-- FKs only prove the term EXISTS; they cannot prove it is the right kind. Same gap the
-- TBox side covers for tbox_hierarchy, so mirror the trigger pattern here.
CREATE OR REPLACE FUNCTION check_abox_class_kind() RETURNS TRIGGER AS $$
DECLARE k varchar(20);
BEGIN
    SELECT term_type INTO k FROM tbox_term WHERE id = NEW.class_id;
    IF k IS DISTINCT FROM 'CLASS' THEN
        RAISE EXCEPTION 'abox_individual.class_id must reference a CLASS term (% is %)',
            NEW.class_id, COALESCE(k, 'missing');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_abox_individual_class_kind ON abox_individual;
CREATE TRIGGER trg_abox_individual_class_kind
BEFORE INSERT OR UPDATE OF class_id ON abox_individual
FOR EACH ROW EXECUTE FUNCTION check_abox_class_kind();

CREATE OR REPLACE FUNCTION check_abox_data_prop_kind() RETURNS TRIGGER AS $$
DECLARE k varchar(20);
BEGIN
    SELECT term_type INTO k FROM tbox_term WHERE id = NEW.data_prop_id;
    IF k IS DISTINCT FROM 'DATA_PROP' THEN
        RAISE EXCEPTION 'abox_data_assertion.data_prop_id must reference a DATA_PROP term (% is %)',
            NEW.data_prop_id, COALESCE(k, 'missing');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_abox_assertion_prop_kind ON abox_data_assertion;
CREATE TRIGGER trg_abox_assertion_prop_kind
BEFORE INSERT OR UPDATE OF data_prop_id ON abox_data_assertion
FOR EACH ROW EXECUTE FUNCTION check_abox_data_prop_kind();
