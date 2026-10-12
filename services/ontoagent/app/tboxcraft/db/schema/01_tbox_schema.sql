-- =====================================================================
-- tboxcraft relational store — CONVERGING schema.
--
-- Applied to a LIVE database and safe to re-run: it brings whatever is there to this shape.
--
--   present & correct -> no-op       absent -> created       column added later -> ALTER'd in
--
-- This is deliberately NOT a /docker-entrypoint-initdb.d script: those fire only on an empty
-- volume, so once the database has booted a schema edit would never reach it. The DB stays up;
-- this file is re-applied against it.
--
-- How to keep it converging when you edit it:
--   * new table   -> CREATE TABLE IF NOT EXISTS
--   * new column  -> add it to the CREATE *and* to the "column convergence" block.
--                    CREATE TABLE IF NOT EXISTS does NOT alter an existing table, so a column
--                    added only to the CREATE is silently missing on databases created earlier.
--   * constraint / index / trigger / view -> DROP IF EXISTS then CREATE (done throughout).
--   * type change / rename -> NOT expressible here; write an explicit migration.
--
-- PRECONDITION — upstream applies FIRST: document -> paragraph -> cq -> (this file).
-- Those three tables are owned by the upstream ingestion model, and the consumer views in
-- section 6 join them. Applying this against a database where upstream has not run fails at
-- CREATE VIEW under ON_ERROR_STOP=1, and that hard failure is the intended behaviour: it names
-- the missing table at apply time. The alternative — guarding the views so they are skipped —
-- moves the error to a consumer receiving `relation "Classes" does not exist` later, with
-- nothing connecting it to the apply that silently omitted them.
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- =====================================================================
-- 1. Tables
-- =====================================================================

CREATE TABLE IF NOT EXISTS embedding_model (
    id             VARCHAR(50) PRIMARY KEY,
    model_name     VARCHAR(100) NOT NULL,
    revision       VARCHAR(50)  NOT NULL,
    dimensions     INT          NOT NULL CHECK (dimensions = 1024),
    -- Query-centering mean (center_query). Immutable: a changed mean silently invalidates every
    -- vector stored against this model, so a new mean means a NEW model row (trigger enforces).
    whitening_mean vector(1024) NOT NULL,
    created_at     TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ontology_snapshot (
    id                 VARCHAR(100) PRIMARY KEY,
    parent_snapshot_id VARCHAR(100) REFERENCES ontology_snapshot(id),
    serialized_turtle  TEXT         NOT NULL,
    file_hash          VARCHAR(64)  NOT NULL,
    reasoner_status    VARCHAR(20)  NOT NULL
        CHECK (reasoner_status IN ('VALIDATED', 'FAILED', 'ROLLBACKED')),
    note               TEXT,        -- which operation produced it ('generate', 'review:relabel', ...)
    -- The IRI namespace this snapshot's TBox belongs to. One chain per namespace: a document is
    -- one KG, so "the latest ontology" is only meaningful within a namespace. Without it every
    -- /update started from whatever the *store* last wrote and serialized the whole graph back,
    -- so a per-document TTL was a point-in-time snapshot wearing a document's name.
    -- Nullable for rows written before the column existed; section 3c backfills them.
    base_iri           VARCHAR(500),
    created_at         TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- cq is NOT defined here. It belongs to the upstream ingestion model, together with document and
-- paragraph — a CQ is a parsing artefact, and the side that parses owns its shape. tboxcraft reads
-- it (gloss text) and cites it (source.cq_id, term_cq_map.cq_id) but never defines or alters it,
-- so nothing below may assume its columns beyond what upstream publishes:
--
--   cq(cq_id PK, document_id FK, paragraph_id FK, run_id, question, answer)   -- all NOT NULL
--   document(document_id PK, document_name NULLABLE, location, content_hash, processed_at)
--
-- document_name being nullable is why the view below still guards on IS NOT NULL: a document may
-- be registered before its name is known, and array_agg would otherwise fold a NULL into "RefDocuments".
--
-- The name is upstream's, not a consumer-side alias. This file and the runtime used to read
-- `cq_repository(id, competency_question, expected_answer, ref_document, ...)`, a shape only the
-- upstream orchestrator produced by mapping columns as it copied. Any path that skipped it saw an
-- empty table, derived zero sources, and left every term unattributed while still returning 200.
-- Reading the producer's name deletes the mapping layer instead of duplicating it per path.
--
-- Consequences, all deliberate:
--   * no FK to it anywhere — the same rule paragraph already follows, because a reference to an
--     absent table hard-fails the whole apply (ON_ERROR_STOP=1) rather than degrading;
--   * the consumer views in section 6 DO join it, and `document` with it, for "RefDocuments" — the
--     document name lives on `document.document_name`, since `cq` carries no ref_document column.
--     They are why the apply order in the header is a precondition and not a preference. A view
--     is a contract: one whose columns depend on what happened to exist at apply time is worse
--     than an apply that stops and says which table is missing;
--   * the source backfill in section 3b keeps its to_regclass('cq') guard even though the order
--     above makes it unreachable in a correct apply. It is no longer a defence — it is the apply
--     order written down where the next editor of that block will see it, and it costs nothing.

-- =====================================================================
-- rdb kind leaves — the RDB counterpart of document -> paragraph -> cq.
--
-- Schema is owned here; the values are supplied by a relational-database producer, which reads
-- a catalogue offline and ships them with its output. tboxcraft never connects to those databases:
-- dsn_label is an identifier, never a credential.
-- =====================================================================

CREATE TABLE IF NOT EXISTS rdb_database (
    rdb_database_id VARCHAR(100) PRIMARY KEY,
    dsn_label       VARCHAR(255) NOT NULL,   -- connection identity, secrets excluded
    schema_name     VARCHAR(255) NOT NULL,
    catalog_hash    VARCHAR(100),            -- schema fingerprint, so a catalogue change is visible
    introspected_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rdb_table (
    rdb_table_id    VARCHAR(150) PRIMARY KEY,
    rdb_database_id VARCHAR(100) NOT NULL REFERENCES rdb_database(rdb_database_id) ON DELETE CASCADE,
    table_name      VARCHAR(255) NOT NULL,
    temporality     VARCHAR(20),             -- static | volatile
    UNIQUE (rdb_database_id, table_name)
);

CREATE TABLE IF NOT EXISTS rdb_ref (
    rdb_ref_id   VARCHAR(255) PRIMARY KEY,
    rdb_table_id VARCHAR(150) NOT NULL REFERENCES rdb_table(rdb_table_id) ON DELETE CASCADE,
    -- Which grain of the table this reference points at: both NULL = the table itself (a class),
    -- column_name = a column (a data property), pk_value = one row (an individual).
    column_name  VARCHAR(255),
    pk_value     VARCHAR(255),
    -- The RDB counterpart of cq.question/answer: definitions, comments and
    -- sample values the adapter gathered. Without it a gloss can be rebuilt from a document source
    -- but not from an RDB one, and reproducibility becomes asymmetric between the two front ends.
    evidence_text TEXT
);

-- =====================================================================
-- source registry — the provenance spine every produced artefact points at.
--
-- Provenance used to be CQ-only (term_cq_map, source_cq_id). That assumption holds for document
-- input and hard-fails for RDB input, where a term comes from a table row and there is no CQ. The
-- requirement is "trace the origin", not "must be a CQ", so the spine is generalised one level:
-- consumers (terms, edges, ABox) reference source_id alone and never learn what kind of source it
-- was. Depth is unchanged — source -> (cq | rdb_ref) -> ... -> document | rdb_database.
--
-- source_id is a DETERMINISTIC natural key computed by the adapter, not issued here:
--   document: doc:cq:<cq_id>
--   rdb     : rdb:<dsn_label>.<schema>.<table>[#col=<col> | #pk=<pk>]
-- so the adapter stays read-only and offline, and re-running an ingest is idempotent because the
-- upsert of an already-known key is a no-op.
-- =====================================================================

CREATE TABLE IF NOT EXISTS source (
    source_id  VARCHAR(255) PRIMARY KEY,
    kind       VARCHAR(20)  NOT NULL CHECK (kind IN ('document', 'rdb')),
    label      VARCHAR(500),            -- human-readable, e.g. 'row 42 of table unit'
    run_id     VARCHAR(100),            -- tag only; there is deliberately no run table
    -- Exclusive arc: exactly one leaf per kind. A source with neither (or both) filled would be
    -- untraceable in one direction and ambiguous in the other, so the CHECK below rejects it.
    -- No FK: cq is upstream-owned (see the note at the top of the tables section).
    -- A dangling cq_id is therefore possible and is the write path's problem, not the DDL's —
    -- the alternative is an apply that hard-fails whenever upstream has not run yet.
    cq_id      VARCHAR(50),
    rdb_ref_id VARCHAR(255) REFERENCES rdb_ref(rdb_ref_id) ON DELETE CASCADE,
    CHECK (
        (kind = 'document' AND cq_id IS NOT NULL AND rdb_ref_id IS NULL) OR
        (kind = 'rdb'      AND rdb_ref_id IS NOT NULL AND cq_id IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_source_cq      ON source (cq_id);
CREATE INDEX IF NOT EXISTS idx_source_rdb_ref ON source (rdb_ref_id);

CREATE TABLE IF NOT EXISTS tbox_term (
    id                     TEXT PRIMARY KEY,          -- full IRI
    term_type              VARCHAR(20) NOT NULL
        CHECK (term_type IN ('CLASS', 'OBJECT_PROP', 'DATA_PROP')),
    status                 VARCHAR(20) NOT NULL DEFAULT 'ACTIVE'
        CHECK (status IN ('DRAFT', 'ACTIVE', 'DEPRECATED')),
    embedding_model_id     VARCHAR(50)  NOT NULL REFERENCES embedding_model(id),
    embedding_val          vector(1024) NOT NULL,     -- whitened; no term without a vector
    created_in_snapshot_id VARCHAR(100) REFERENCES ontology_snapshot(id),
    -- Advisory domain-term verdict. Never gates; kept for audit/quality review. Provenance
    -- is mandatory when present (chk_verdict_provenance) so a stored opinion is attributable to a
    -- model — verdicts from different verifiers are not comparable without it.
    llm_verdict            JSONB,
    llm_verdict_model      VARCHAR(100),
    llm_verdict_at         TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS surface_form (
    id           SERIAL PRIMARY KEY,
    term_id      TEXT NOT NULL REFERENCES tbox_term(id) ON DELETE CASCADE,
    surface_text TEXT NOT NULL,
    lang         VARCHAR(10) NOT NULL,   -- no default: an unset lang would mislabel the term
    form_type    VARCHAR(20) NOT NULL CHECK (form_type IN ('PREF', 'ALTER', 'HIDDEN')),
    source       VARCHAR(100) NOT NULL
    -- Uniqueness is idx_unique_surface_role below, NOT an inline UNIQUE on
    -- (term_id, surface_text, lang, form_type). form_type in the key meant a term could hold one
    -- surface in two roles at once: an ALTER written into a string already HIDDEN was not a
    -- conflict, it was a second row, and the earlier hide decision was reversed without anything
    -- saying so. Expressed as an index because an existing database cannot have its inline
    -- constraint changed by a convergent apply.
);

-- Superseded by term_source_map below. Kept because it holds data and the convergent schema
-- cannot RENAME: the two coexist, term_source_map is backfilled from this one, and a later
-- explicit migration retires it. New writes go to term_source_map only.
CREATE TABLE IF NOT EXISTS term_cq_map (
    term_id TEXT        REFERENCES tbox_term(id)     ON DELETE CASCADE,
    cq_id   VARCHAR(50),                 -- no FK: upstream-owned table
                                          -- (kept for the ABox loader until it moves to source)
    PRIMARY KEY (term_id, cq_id)
);

-- Term provenance, source-neutral. A term can be claimed by several sources — clustering merges
-- components that came from different CQs, and the same class can be asserted by a document and
-- an RDB table — so this is m:n, not a column on tbox_term.
--
-- INVARIANT (write path, not DDL): >= 1 row per term. A term with no source is an orphan whose
-- origin cannot be traced, which is the whole point of the spine. Postgres cannot express
-- "at least one referencing row", so ingest is what guarantees it.
CREATE TABLE IF NOT EXISTS term_source_map (
    term_id   TEXT         NOT NULL REFERENCES tbox_term(id)   ON DELETE CASCADE,
    source_id VARCHAR(255) NOT NULL REFERENCES source(source_id) ON DELETE CASCADE,
    run_id    VARCHAR(100),
    PRIMARY KEY (term_id, source_id)
);

CREATE INDEX IF NOT EXISTS idx_term_source_source ON term_source_map (source_id);

CREATE TABLE IF NOT EXISTS tbox_hierarchy (
    child_id  TEXT NOT NULL REFERENCES tbox_term(id) ON DELETE CASCADE,
    parent_id TEXT NOT NULL REFERENCES tbox_term(id) ON DELETE CASCADE,
    PRIMARY KEY (child_id, parent_id),
    CHECK (child_id <> parent_id)     -- 1-hop guard; multi-hop cycles are caught by the trigger
);

CREATE TABLE IF NOT EXISTS tbox_domain_range (
    id              SERIAL PRIMARY KEY,
    prop_id         TEXT        NOT NULL REFERENCES tbox_term(id) ON DELETE CASCADE,
    relation_type   VARCHAR(10) NOT NULL CHECK (relation_type IN ('DOMAIN', 'RANGE')),
    target_class_id TEXT        REFERENCES tbox_term(id) ON DELETE CASCADE,
    xsd_datatype    VARCHAR(50),
    -- Exactly one of the two targets. A multi-class (union) range is stored as several RANGE
    -- rows — the relational form has no union node, so the union is implied by row count.
    CHECK (
        (target_class_id IS NOT NULL AND xsd_datatype IS NULL) OR
        (target_class_id IS NULL AND xsd_datatype IS NOT NULL)
    )
);

-- Edge provenance as m:n, not a column on the edge. A subClassOf or domain/range relation can be
-- asserted by several sources (the same claim read from more than one question, or from a document
-- and an RDB foreign key), so it is the same shape as term_source_map, not a single source column.
-- Provenance is asserted-only for now: origin (asserted|inferred) and source_snapshot_id are
-- reserved for a later version, when an inferred producer and a replace_topology redesign land
-- together (there is no inferred producer today, and replace_topology full-replaces every batch, so
-- a stored inferred row would be deleted each run anyway).
--
-- Both tables ON DELETE CASCADE from their edge: replace_topology clears and re-inserts the edge
-- tables every batch, so their maps are rebuilt on re-ingest — deliberately, so the map never
-- outlives the edge it describes.
--
-- These two replace the CQ-only hierarchy_cq_map / domain_range_cq_map added earlier on this same
-- branch. DROP is safe here (and DROP-then-CREATE is what the header prescribes for anything that
-- cannot be ALTER'd into shape) precisely because those two never shipped and were never written
-- to — the commit that added them left population to a follow-up. term_cq_map is the opposite case
-- and is therefore kept and backfilled instead.
DROP TABLE IF EXISTS hierarchy_cq_map;
DROP TABLE IF EXISTS domain_range_cq_map;

CREATE TABLE IF NOT EXISTS hierarchy_source_map (
    child_id  TEXT         NOT NULL,
    parent_id TEXT         NOT NULL,
    source_id VARCHAR(255) NOT NULL REFERENCES source(source_id) ON DELETE CASCADE,
    PRIMARY KEY (child_id, parent_id, source_id),
    FOREIGN KEY (child_id, parent_id)
        REFERENCES tbox_hierarchy(child_id, parent_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS domain_range_source_map (
    domain_range_id INTEGER      NOT NULL REFERENCES tbox_domain_range(id) ON DELETE CASCADE,
    source_id       VARCHAR(255) NOT NULL REFERENCES source(source_id)     ON DELETE CASCADE,
    PRIMARY KEY (domain_range_id, source_id)
);

CREATE TABLE IF NOT EXISTS review_queue (
    id               VARCHAR(100) PRIMARY KEY,
    -- NULL for a candidate that is not a term yet (GRAY / CREATE) — the common case, and the
    -- reason this FK cannot be NOT NULL: gray items are precisely the ones not yet committed.
    term_id          TEXT REFERENCES tbox_term(id) ON DELETE SET NULL,
    candidate_uid    VARCHAR(255) NOT NULL,
    -- Which document's batch raised this. A queue row belongs to the namespace that raised it, and
    -- it cannot be derived from term_id: the items most in need of review (GRAY, KIND_CONFLICT)
    -- are exactly the ones not minted yet, so term_id is NULL for them. Without this column the
    -- queue listing mixes two documents' items with no way to tell them apart.
    -- Nullable for rows written before it existed; section 3d backfills what it can.
    base_iri         VARCHAR(500),
    entry_type       VARCHAR(20)  NOT NULL
        CHECK (entry_type IN ('GRAY', 'RELABEL', 'DELETE', 'CREATE', 'KIND_CONFLICT',
                              'PREF_CONFLICT')),
    -- Frozen at enqueue: re-embedding at decision time could land in a different vector space.
    -- Nullable only for DELETE: a dangling-orphan candidate is an IRI that is referenced but has
    -- no term behind it, so there is nothing to embed (chk_candidate_vector below enforces this).
    candidate_vector vector(1024),
    nearest          JSONB        NOT NULL,
    gloss            TEXT,
    verdict          JSONB,
    status           VARCHAR(20)  NOT NULL DEFAULT 'PENDING'
        CHECK (status IN ('PENDING', 'APPROVED', 'REJECTED')),
    diff_payload     JSONB        NOT NULL,
    -- The authoritative human call (verdict above is only the LLM's advisory opinion).
    decision         JSONB,
    resolved_at      TIMESTAMP WITH TIME ZONE,
    created_at       TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (candidate_uid, entry_type)   -- idempotent enqueue
);

-- =====================================================================
-- 2. Column convergence — brings tables created by an older version of this file up to date.
--    (CREATE TABLE IF NOT EXISTS above skips existing tables entirely.)
-- =====================================================================

ALTER TABLE tbox_term ADD COLUMN IF NOT EXISTS llm_verdict       JSONB;
ALTER TABLE tbox_term ADD COLUMN IF NOT EXISTS llm_verdict_model VARCHAR(100);
ALTER TABLE tbox_term ADD COLUMN IF NOT EXISTS llm_verdict_at    TIMESTAMP WITH TIME ZONE;

-- Which operation produced a snapshot ('generate', 'update', 'review:relabel', ...). Without it
-- the snapshot chain records that the ontology changed but not what changed it.
ALTER TABLE ontology_snapshot ADD COLUMN IF NOT EXISTS note     TEXT;
ALTER TABLE ontology_snapshot ADD COLUMN IF NOT EXISTS base_iri VARCHAR(500);

-- The human decision that resolved a review item, and when. The verdict column holds the LLM's
-- advisory opinion; these hold the authoritative call, which is the part that must be auditable.
ALTER TABLE review_queue ADD COLUMN IF NOT EXISTS decision    JSONB;
ALTER TABLE review_queue ADD COLUMN IF NOT EXISTS base_iri    VARCHAR(500);
ALTER TABLE review_queue ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMP WITH TIME ZONE;

-- =====================================================================
-- 3. Named constraints (drop-then-add so re-running converges)
-- =====================================================================

ALTER TABLE embedding_model DROP CONSTRAINT IF EXISTS uq_model_rev;
ALTER TABLE embedding_model ADD  CONSTRAINT uq_model_rev UNIQUE (model_name, revision);

ALTER TABLE tbox_domain_range DROP CONSTRAINT IF EXISTS chk_domain_no_literal;
ALTER TABLE tbox_domain_range ADD  CONSTRAINT chk_domain_no_literal
    CHECK (relation_type <> 'DOMAIN' OR target_class_id IS NOT NULL) NOT VALID;
ALTER TABLE tbox_domain_range VALIDATE CONSTRAINT chk_domain_no_literal;

ALTER TABLE tbox_domain_range DROP CONSTRAINT IF EXISTS uq_domain_range;
ALTER TABLE tbox_domain_range ADD  CONSTRAINT uq_domain_range
    UNIQUE NULLS NOT DISTINCT (prop_id, relation_type, target_class_id, xsd_datatype);

-- Everything that can become a term must carry its frozen vector; only a DELETE candidate (a
-- dangling orphan IRI with no term behind it) legitimately has none.
ALTER TABLE review_queue ALTER COLUMN candidate_vector DROP NOT NULL;
ALTER TABLE review_queue DROP CONSTRAINT IF EXISTS chk_candidate_vector;
ALTER TABLE review_queue ADD  CONSTRAINT chk_candidate_vector
    CHECK (entry_type = 'DELETE' OR candidate_vector IS NOT NULL) NOT VALID;
ALTER TABLE review_queue VALIDATE CONSTRAINT chk_candidate_vector;

-- KIND_CONFLICT: one name judged as two kinds (an object property and a data property both named
-- usesWeapon). Both are minted — they are separate IRIs by design — so the row is not a
-- proposal to remove either; it is the contradiction itself, put where a human sees it. The
-- constraint is replaced rather than extended because a CHECK cannot be added to; an older
-- database rejects the insert outright, so this must run before the first such enqueue.
ALTER TABLE review_queue DROP CONSTRAINT IF EXISTS review_queue_entry_type_check;
ALTER TABLE review_queue ADD  CONSTRAINT review_queue_entry_type_check
    CHECK (entry_type IN ('GRAY', 'RELABEL', 'DELETE', 'CREATE', 'KIND_CONFLICT',
                              'PREF_CONFLICT')) NOT VALID;
ALTER TABLE review_queue VALIDATE CONSTRAINT review_queue_entry_type_check;

ALTER TABLE tbox_term DROP CONSTRAINT IF EXISTS chk_verdict_provenance;
ALTER TABLE tbox_term ADD  CONSTRAINT chk_verdict_provenance CHECK (
    llm_verdict IS NULL
    OR (llm_verdict_model IS NOT NULL AND llm_verdict_at IS NOT NULL)
) NOT VALID;
ALTER TABLE tbox_term VALIDATE CONSTRAINT chk_verdict_provenance;

-- =====================================================================
-- 3b. Data convergence — backfill the source spine from CQ-only provenance.
--
-- Runs on every apply and is idempotent: both statements are ON CONFLICT DO NOTHING over
-- deterministic keys, so once a database has converged they are no-ops. This is what makes the
-- term_cq_map -> term_source_map move safe without a RENAME the convergent schema cannot express.
-- =====================================================================

-- Guarded on cq existing: it is upstream-owned, so this file must still apply against a database
-- where upstream has not run yet. Skipping is safe — with no CQ table there is no CQ-era
-- provenance to carry over, which is exactly the case the guard covers.
DO $$
BEGIN
    IF to_regclass('cq') IS NULL THEN
        RAISE NOTICE 'cq absent (upstream not applied) — skipping source backfill';
        RETURN;
    END IF;

    -- Every CQ is a document-kind source under its deterministic key. Done for all of cq, not only
    -- the rows term_cq_map references, so a CQ registered before any term cites it already has its
    -- source row.
    EXECUTE $q$
        INSERT INTO source (source_id, kind, label, cq_id)
        SELECT 'doc:cq:' || c.cq_id, 'document', c.question, c.cq_id
        FROM cq c
        ON CONFLICT (source_id) DO NOTHING
    $q$;

    -- Carry the existing term->CQ links onto the spine. term_cq_map stays as it is; new writes go
    -- to term_source_map only, so this backfill converges once and then stops finding anything.
    EXECUTE $q$
        INSERT INTO term_source_map (term_id, source_id)
        SELECT m.term_id, 'doc:cq:' || m.cq_id
        FROM term_cq_map m
        ON CONFLICT (term_id, source_id) DO NOTHING
    $q$;
END $$;

-- =====================================================================
-- 3c. Data convergence — adopt pre-namespace snapshots into a chain.
--
-- Snapshots written before base_iri existed carry NULL, and a NULL row belongs to no chain: the
-- scoped lookup would not find it, so the first /update after this column lands would see "no
-- ontology" and demand a cold generate. That is a worse failure than a wrong guess, so the rows
-- are adopted — but only when the guess cannot be wrong.
--
-- The namespace is derived from the terms themselves rather than from an env var, because this
-- file cannot read the app's environment and a value passed in at apply time can disagree with
-- what was actually minted. tbox_term.id IS the record of which namespace produced these
-- snapshots.
--
-- Exactly one namespace -> that is the answer. Two or more means base_iri is already being
-- written and these NULLs predate the split, where picking a chain would be a coin flip: they are
-- left NULL and reported, which surfaces as a cold-start demand rather than a silent misfiling.
-- Idempotent via `WHERE base_iri IS NULL`.
-- =====================================================================

DO $$
DECLARE ns TEXT; n INT;
BEGIN
    SELECT count(*), min(b) INTO n, ns FROM (
        SELECT DISTINCT substring(id from '^(.*)/(class|objectProperty|dataProperty)/') AS b
        FROM tbox_term
    ) s WHERE b IS NOT NULL;

    IF n = 1 THEN
        UPDATE ontology_snapshot SET base_iri = ns WHERE base_iri IS NULL;
        RAISE NOTICE 'ontology_snapshot.base_iri: legacy rows adopted into %', ns;
    ELSIF n > 1 THEN
        RAISE NOTICE 'ontology_snapshot.base_iri: % namespaces present — legacy rows left NULL '
                     '(no single chain to adopt them into)', n;
    END IF;
END $$;

-- =====================================================================
-- 3d. Data convergence — give pre-namespace queue rows a namespace.
--
-- Two sources, in order of certainty. A row whose term_id is set says which namespace it is in,
-- because the IRI is the namespace record; that is derived per row and is never a guess. What is
-- left is the unminted candidates (GRAY, KIND_CONFLICT) — the majority — and those only have an
-- answer when the store holds exactly one namespace, which is the pre-split state this backfill
-- exists for. Beyond that they stay NULL and are reported: a queue item filed under the wrong
-- document is worse than one that shows up in neither listing, because the second is visible.
-- Idempotent via `WHERE base_iri IS NULL`.
-- =====================================================================

DO $$
DECLARE ns TEXT; n INT; left_null INT;
BEGIN
    UPDATE review_queue q SET base_iri = substring(t.id from '^(.*)[/#][^/#]+$')
    FROM tbox_term t WHERE t.id = q.term_id AND q.base_iri IS NULL;

    SELECT count(*), min(b) INTO n, ns FROM (
        SELECT DISTINCT substring(id from '^(.*)/(class|objectProperty|dataProperty)/') AS b
        FROM tbox_term
    ) s WHERE b IS NOT NULL;

    IF n = 1 THEN
        UPDATE review_queue SET base_iri = ns WHERE base_iri IS NULL;
    END IF;

    SELECT count(*) INTO left_null FROM review_queue WHERE base_iri IS NULL;
    IF left_null > 0 THEN
        RAISE NOTICE 'review_queue.base_iri: % row(s) left NULL (no single namespace to adopt '
                     'them into) — they appear in no namespace-scoped listing', left_null;
    END IF;
END $$;

-- Surface-collision review rows raised under a mixed-case key, before the key was casefolded.
-- The two writers report different spellings of one string (the in-batch scan prefers a written
-- surface's spelling, the store scan reports what the store holds), so one collision could sit in
-- the queue twice and n_review_rows counted both.
--
-- PENDING only, and only where the casefolded twin exists — a decided row is history and is never
-- removed, and the payload of a pending one is rebuilt by the next scan, so nothing is lost.
DELETE FROM review_queue q
WHERE q.candidate_uid LIKE 'surface:%'
  AND q.status = 'PENDING'
  AND q.candidate_uid <> lower(q.candidate_uid)
  AND EXISTS (
      SELECT 1 FROM review_queue t
      WHERE t.candidate_uid = lower(q.candidate_uid)
        AND t.entry_type = q.entry_type
  );

-- =====================================================================
-- 4. Triggers
-- =====================================================================

CREATE OR REPLACE FUNCTION prevent_whitening_mean_update()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.whitening_mean IS DISTINCT FROM OLD.whitening_mean THEN
        RAISE EXCEPTION 'Immutable Data Error: whitening_mean cannot be modified. Create a new model revision instead.';
    END IF;
    RETURN NEW;
END; $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_immutable_whitening_mean ON embedding_model;
CREATE TRIGGER trg_immutable_whitening_mean
BEFORE UPDATE ON embedding_model
FOR EACH ROW EXECUTE FUNCTION prevent_whitening_mean_update();

CREATE OR REPLACE FUNCTION check_tbox_hierarchy_integrity()
RETURNS TRIGGER AS $$
DECLARE
    child_type   VARCHAR(20);
    parent_type  VARCHAR(20);
    cycle_exists BOOLEAN;
BEGIN
    SELECT term_type INTO child_type  FROM tbox_term WHERE id = NEW.child_id;
    SELECT term_type INTO parent_type FROM tbox_term WHERE id = NEW.parent_id;

    -- FKs alone would happily connect a CLASS under an OBJECT_PROP.
    IF child_type <> parent_type THEN
        RAISE EXCEPTION 'Hierarchy Type Mismatch: Cannot connect child (%) of type % to parent (%) of type %',
            NEW.child_id, child_type, NEW.parent_id, parent_type;
    END IF;

    -- The table CHECK only stops a 1-hop self-loop; a longer cycle would make the reasoner
    -- infer the classes equivalent (or blow up), so walk the ancestors before inserting.
    WITH RECURSIVE ancestors AS (
        SELECT parent_id FROM tbox_hierarchy WHERE child_id = NEW.parent_id
        UNION
        SELECT h.parent_id FROM tbox_hierarchy h JOIN ancestors a ON h.child_id = a.parent_id
    )
    SELECT EXISTS (SELECT 1 FROM ancestors WHERE parent_id = NEW.child_id) INTO cycle_exists;

    IF cycle_exists THEN
        RAISE EXCEPTION 'HermiT Cycle Error: Multi-hop loop detected between % and %',
            NEW.child_id, NEW.parent_id;
    END IF;

    RETURN NEW;
END; $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_tbox_hierarchy_integrity ON tbox_hierarchy;
CREATE TRIGGER trg_tbox_hierarchy_integrity
BEFORE INSERT OR UPDATE ON tbox_hierarchy
FOR EACH ROW EXECUTE FUNCTION check_tbox_hierarchy_integrity();

-- =====================================================================
-- 5. Indexes
-- =====================================================================

-- Partial HNSW: the kNN path always filters out retired terms, and the views' status predicate
-- implies this one, so lookups through the views still use it.
CREATE INDEX IF NOT EXISTS idx_tbox_vector_hnsw ON tbox_term
USING hnsw (embedding_val vector_cosine_ops) WHERE status != 'DEPRECATED';

CREATE INDEX IF NOT EXISTS idx_review_vector_hnsw ON review_queue
USING hnsw (candidate_vector vector_cosine_ops) WHERE status = 'PENDING';

-- "One PREF per language per term" — the core lexical invariant.
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_pref_per_lang
ON surface_form (term_id, lang) WHERE form_type = 'PREF';

-- "One role per surface per term" — the same pattern as the PREF index above, one level out.
-- Uniqueness *inside* a term was already enforced by key; uniqueness *of a surface's role* was
-- not, because the table's inline UNIQUE carried form_type. So HIDDEN 'AreaOfOperations' and
-- ALTER 'AreaOfOperations' on one term coexisted, and a later batch silently re-advertised what
-- an earlier one had decided to bury.
--
-- Keyed on lower(surface_text) because that is what resolution reads (idx_surface_text_lower,
-- resolve_surface, active_surface_owner) and what insert_surface_forms already compares for its
-- PREF guard. An exact-text key leaves 'MainCP' ALTER beside 'MainCp' HIDDEN — two roles for one
-- resolvable string, which is the same defect spelled differently.
DO $$
DECLARE
    dropped text;
    deduped bigint;
BEGIN
    -- The inline UNIQUE, on databases created before it moved out. Found by its columns rather
    -- than by the generated name, which is not guaranteed.
    SELECT c.conname INTO dropped
    FROM pg_constraint c
    WHERE c.conrelid = 'surface_form'::regclass AND c.contype = 'u'
      AND (SELECT array_agg(a.attname::text ORDER BY a.attname)
           FROM unnest(c.conkey) k JOIN pg_attribute a
             ON a.attrelid = c.conrelid AND a.attnum = k)
          = ARRAY['form_type', 'lang', 'surface_text', 'term_id'];
    IF dropped IS NOT NULL THEN
        EXECUTE format('ALTER TABLE surface_form DROP CONSTRAINT %I', dropped);
        RAISE NOTICE 'dropped superseded unique constraint %', dropped;
    END IF;

    -- Dedup before the index can be built. The surviving row is the one written FIRST, which is
    -- exactly what the index enforces from here on (a later conflicting insert becomes a no-op) —
    -- so the migration and the running behaviour agree instead of the migration preferring a role
    -- the code would not. That means an earlier HIDDEN keeps the surface hidden, which is the
    -- decision that was being reversed.
    WITH loser AS (
        SELECT s.id FROM surface_form s
        WHERE EXISTS (
            SELECT 1 FROM surface_form o
            WHERE o.term_id = s.term_id AND o.lang = s.lang
              AND lower(o.surface_text) = lower(s.surface_text)
              -- PREF outranks age: it is the term's name, and dropping it would leave the concept
              -- unlabelled. In practice nothing overlaps that way (insert_surface_forms refuses an
              -- alias equal to its own term's PREF); this is here so the invariant cannot be
              -- satisfied by deleting the one row that must exist.
              AND (o.form_type = 'PREF' AND s.form_type <> 'PREF'
                   OR (o.form_type = 'PREF') = (s.form_type = 'PREF') AND o.id < s.id)
        )
    )
    DELETE FROM surface_form WHERE id IN (SELECT id FROM loser);
    GET DIAGNOSTICS deduped = ROW_COUNT;
    IF deduped > 0 THEN
        RAISE NOTICE 'surface_form: removed % row(s) holding a surface in a second role', deduped;
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_surface_role
ON surface_form (term_id, lower(surface_text), lang);

-- Surface -> term lookup for /resolve (the UNIQUE above leads with term_id, so it cannot serve it).
CREATE INDEX IF NOT EXISTS idx_surface_text_lower ON surface_form (lower(surface_text));

CREATE INDEX IF NOT EXISTS idx_term_cq_map_reverse ON term_cq_map (cq_id);

-- Audit queries: "which seeds did the verifier doubt", and per-model comparison.
CREATE INDEX IF NOT EXISTS idx_tbox_term_verdict_flag
ON tbox_term ((llm_verdict->>'is_official_term')) WHERE llm_verdict IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tbox_term_verdict_model
ON tbox_term (llm_verdict_model) WHERE llm_verdict_model IS NOT NULL;

-- =====================================================================
-- 6. Consumer views
--    Dropped first: CREATE OR REPLACE VIEW cannot add a column in the middle or drop one, so an
--    edit to the column list fails against an older view.
--    Topology targets are filtered to ACTIVE too — otherwise a Parent/Domain/Range could point at
--    an IRI that appears in no view at all, i.e. a phantom reference for the consumer.
-- =====================================================================

DROP VIEW IF EXISTS "Classes" CASCADE;
DROP VIEW IF EXISTS "ObjectProperties" CASCADE;
DROP VIEW IF EXISTS "DataProperties" CASCADE;

CREATE VIEW "Classes" AS
SELECT
    t.id AS "Class_IRI",
    (SELECT surface_text FROM surface_form WHERE term_id=t.id AND form_type='PREF' AND lang='ko' LIMIT 1) AS "PrefTerm_ko",
    (SELECT surface_text FROM surface_form WHERE term_id=t.id AND form_type='PREF' AND lang='en' LIMIT 1) AS "PrefTerm_en",
    COALESCE((SELECT jsonb_object_agg(lang, terms) FROM (
        SELECT lang, jsonb_agg(surface_text) AS terms FROM surface_form
        WHERE term_id=t.id AND form_type='ALTER' GROUP BY lang) s_alt), '{}'::jsonb) AS "AlterTerms",
    COALESCE((SELECT jsonb_object_agg(lang, terms) FROM (
        SELECT lang, jsonb_agg(surface_text) AS terms FROM surface_form
        WHERE term_id=t.id AND form_type='HIDDEN' GROUP BY lang) s_hid), '{}'::jsonb) AS "HiddenTerms",
    COALESCE((SELECT array_agg(h.parent_id) FROM tbox_hierarchy h
              JOIN tbox_term p ON p.id=h.parent_id AND p.status='ACTIVE'
              WHERE h.child_id=t.id), '{}') AS "Parents",
    COALESCE((SELECT array_agg(DISTINCT d.document_name) FROM term_cq_map m
              JOIN cq c ON m.cq_id=c.cq_id
              JOIN document d ON d.document_id=c.document_id
              WHERE m.term_id=t.id AND d.document_name IS NOT NULL), '{}') AS "RefDocuments",
    COALESCE((SELECT array_agg(m.cq_id) FROM term_cq_map m WHERE m.term_id=t.id), '{}') AS "CQIds",
    t.embedding_val AS "Vector"
FROM tbox_term t
WHERE t.term_type='CLASS' AND t.status='ACTIVE';

CREATE VIEW "ObjectProperties" AS
SELECT
    t.id AS "Property_IRI",
    (SELECT surface_text FROM surface_form WHERE term_id=t.id AND form_type='PREF' AND lang='ko' LIMIT 1) AS "PrefTerm_ko",
    (SELECT surface_text FROM surface_form WHERE term_id=t.id AND form_type='PREF' AND lang='en' LIMIT 1) AS "PrefTerm_en",
    COALESCE((SELECT jsonb_object_agg(lang, terms) FROM (
        SELECT lang, jsonb_agg(surface_text) AS terms FROM surface_form
        WHERE term_id=t.id AND form_type='ALTER' GROUP BY lang) s_alt), '{}'::jsonb) AS "AlterTerms",
    COALESCE((SELECT jsonb_object_agg(lang, terms) FROM (
        SELECT lang, jsonb_agg(surface_text) AS terms FROM surface_form
        WHERE term_id=t.id AND form_type='HIDDEN' GROUP BY lang) s_hid), '{}'::jsonb) AS "HiddenTerms",
    COALESCE((SELECT array_agg(dr.target_class_id) FROM tbox_domain_range dr
              JOIN tbox_term c ON c.id=dr.target_class_id AND c.status='ACTIVE'
              WHERE dr.prop_id=t.id AND dr.relation_type='DOMAIN'), '{}') AS "Domain",
    COALESCE((SELECT jsonb_agg(COALESCE(dr.target_class_id, dr.xsd_datatype)) FROM tbox_domain_range dr
              LEFT JOIN tbox_term c ON c.id=dr.target_class_id
              WHERE dr.prop_id=t.id AND dr.relation_type='RANGE'
                AND (dr.xsd_datatype IS NOT NULL OR c.status='ACTIVE')), '[]'::jsonb) AS "Range",
    COALESCE((SELECT array_agg(DISTINCT d.document_name) FROM term_cq_map m
              JOIN cq c ON m.cq_id=c.cq_id
              JOIN document d ON d.document_id=c.document_id
              WHERE m.term_id=t.id AND d.document_name IS NOT NULL), '{}') AS "RefDocuments",
    COALESCE((SELECT array_agg(m.cq_id) FROM term_cq_map m WHERE m.term_id=t.id), '{}') AS "CQIds",
    t.embedding_val AS "Vector"
FROM tbox_term t
WHERE t.term_type='OBJECT_PROP' AND t.status='ACTIVE';

CREATE VIEW "DataProperties" AS
SELECT
    t.id AS "Property_IRI",
    (SELECT surface_text FROM surface_form WHERE term_id=t.id AND form_type='PREF' AND lang='ko' LIMIT 1) AS "PrefTerm_ko",
    (SELECT surface_text FROM surface_form WHERE term_id=t.id AND form_type='PREF' AND lang='en' LIMIT 1) AS "PrefTerm_en",
    COALESCE((SELECT jsonb_object_agg(lang, terms) FROM (
        SELECT lang, jsonb_agg(surface_text) AS terms FROM surface_form
        WHERE term_id=t.id AND form_type='ALTER' GROUP BY lang) s_alt), '{}'::jsonb) AS "AlterTerms",
    COALESCE((SELECT jsonb_object_agg(lang, terms) FROM (
        SELECT lang, jsonb_agg(surface_text) AS terms FROM surface_form
        WHERE term_id=t.id AND form_type='HIDDEN' GROUP BY lang) s_hid), '{}'::jsonb) AS "HiddenTerms",
    COALESCE((SELECT array_agg(dr.target_class_id) FROM tbox_domain_range dr
              JOIN tbox_term c ON c.id=dr.target_class_id AND c.status='ACTIVE'
              WHERE dr.prop_id=t.id AND dr.relation_type='DOMAIN'), '{}') AS "Domain",
    COALESCE((SELECT jsonb_agg(COALESCE(dr.target_class_id, dr.xsd_datatype)) FROM tbox_domain_range dr
              LEFT JOIN tbox_term c ON c.id=dr.target_class_id
              WHERE dr.prop_id=t.id AND dr.relation_type='RANGE'
                AND (dr.xsd_datatype IS NOT NULL OR c.status='ACTIVE')), '[]'::jsonb) AS "Range",
    COALESCE((SELECT array_agg(DISTINCT d.document_name) FROM term_cq_map m
              JOIN cq c ON m.cq_id=c.cq_id
              JOIN document d ON d.document_id=c.document_id
              WHERE m.term_id=t.id AND d.document_name IS NOT NULL), '{}') AS "RefDocuments",
    COALESCE((SELECT array_agg(m.cq_id) FROM term_cq_map m WHERE m.term_id=t.id), '{}') AS "CQIds",
    t.embedding_val AS "Vector"
FROM tbox_term t
WHERE t.term_type='DATA_PROP' AND t.status='ACTIVE';


-- "PrefTerm_ko" is structurally always NULL, and says so where somebody would go looking.
--
-- The representative label is one per term and English (tboxcraft's PREF_LANG), so the only tags
-- a PREF row can carry are 'en' and — for a term with no English anywhere in its input — 'und'.
-- 'ko' is not reachable. That was already true in practice before the policy was written down:
-- the column has never once held a value.
--
-- It is kept rather than dropped because a NULL column nobody reads costs nothing, while removing
-- one from a published view breaks any consumer that names it — and the consumers of these views
-- are not all in this repository. What it cost instead was a reader seeing NULL and concluding
-- "this class has no Korean name", which is wrong: the Korean name is in "AlterTerms", bucketed
-- by language. These comments are readable through \d+ and information_schema, which is where
-- that reader already is.
DO $$
DECLARE v text;
BEGIN
    FOREACH v IN ARRAY ARRAY['Classes', 'ObjectProperties', 'DataProperties'] LOOP
        EXECUTE format(
            'COMMENT ON COLUMN %I."PrefTerm_ko" IS %L', v,
            'Always NULL: the representative label is one per term and English. A term''s Korean '
            'name is in "AlterTerms" under the "ko" key.');
        EXECUTE format(
            'COMMENT ON COLUMN %I."PrefTerm_en" IS %L', v,
            'The representative label. NULL only when the term had no English anywhere in its '
            'input, in which case the label is under "AlterTerms" with the "und" key.');
        EXECUTE format(
            'COMMENT ON COLUMN %I."AlterTerms" IS %L', v,
            'Alternative surfaces bucketed by language tag, including any curated official term '
            'that could not be the English representative label.');
    END LOOP;
END $$;
