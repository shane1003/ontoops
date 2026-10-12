-- Schema of the load targets.
-- Changing a column requires changing the *Row models in src/schemas.py too: a mismatch
-- breaks only at the moment of inserting into a real DB.
--
-- Changing only a CREATE block does not reach an already created DB. When adding a
-- column, write it in both the CREATE block and the catch-up section at the end of the
-- file, append-only: the former is for new DBs, the latter for existing DBs.
--
-- varchar has no length because in postgres varchar(n) is no faster than text, and a
-- length invites later migrations that only widen it.
CREATE TABLE IF NOT EXISTS document (
    document_id   varchar PRIMARY KEY,
    document_name varchar,
    location      varchar NOT NULL,
    content_hash  varchar NOT NULL UNIQUE,
    processed_at  timestamp NOT NULL
);

CREATE TABLE IF NOT EXISTS paragraph (
    paragraph_id varchar PRIMARY KEY,
    document_id  varchar NOT NULL REFERENCES document (document_id),
    run_id       varchar NOT NULL,
    section_path varchar NOT NULL,
    page_start   int NOT NULL,
    page_end     int NOT NULL,
    seq          int NOT NULL,
    text         text NOT NULL
);

CREATE TABLE IF NOT EXISTS cq (
    cq_id        varchar PRIMARY KEY,
    document_id  varchar NOT NULL REFERENCES document (document_id),
    paragraph_id varchar NOT NULL REFERENCES paragraph (paragraph_id),
    run_id       varchar NOT NULL,
    question     varchar NOT NULL,
    answer       varchar NOT NULL
);

CREATE INDEX IF NOT EXISTS cq_paragraph_id_idx ON cq (paragraph_id);
CREATE INDEX IF NOT EXISTS cq_document_id_idx  ON cq (document_id);

CREATE TABLE IF NOT EXISTS cq_draft (
    draft_id             varchar PRIMARY KEY,
    document_id          varchar NOT NULL REFERENCES document (document_id),
    paragraph_id         varchar NOT NULL REFERENCES paragraph (paragraph_id),
    run_id               varchar NOT NULL,
    question             varchar NOT NULL,
    answer               varchar NOT NULL,
    refine_check_flag    jsonb,
    reviewer_action      varchar,
    reviewer_feedback    varchar,
    regenerated_at       timestamp,
    regeneration_verdict varchar,
    original_question    varchar,
    original_answer      varchar
);

CREATE INDEX IF NOT EXISTS cq_draft_run_id_idx ON cq_draft (run_id);

-- Catch-up for existing DBs (append-only). On a new DB the CREATE above already made
-- these, so all are no-ops.
-- The COMMENT statements below reference columns added here, so this must precede the
-- COMMENT block.
--
-- Add columns to cq_draft that keep the original before regeneration
ALTER TABLE cq_draft ADD COLUMN IF NOT EXISTS original_question varchar;
ALTER TABLE cq_draft ADD COLUMN IF NOT EXISTS original_answer   varchar;

COMMENT ON TABLE  document               IS 'Collected document';
COMMENT ON COLUMN document.document_id   IS 'Document identifier';
COMMENT ON COLUMN document.document_name IS 'Document name';
COMMENT ON COLUMN document.location      IS 'File storage path';
COMMENT ON COLUMN document.content_hash  IS 'Hash of the original bytes. Prevents duplicate collection and detects changes';
COMMENT ON COLUMN document.processed_at  IS 'Time processed by the parsing pipeline';

COMMENT ON TABLE  paragraph              IS 'Parsed paragraph';
COMMENT ON COLUMN paragraph.paragraph_id IS 'Paragraph identifier';
COMMENT ON COLUMN paragraph.document_id  IS 'FK of the owning document';
COMMENT ON COLUMN paragraph.run_id       IS 'Pipeline run ID';
COMMENT ON COLUMN paragraph.section_path IS 'Table of contents path';
COMMENT ON COLUMN paragraph.page_start   IS 'Paragraph start page (in the original PDF)';
COMMENT ON COLUMN paragraph.page_end     IS 'Paragraph end page. Equals page_start if the paragraph does not span pages';
COMMENT ON COLUMN paragraph.seq          IS 'Paragraph sequence number within the document (global order)';
COMMENT ON COLUMN paragraph.text         IS 'Paragraph body. own-node text';

COMMENT ON TABLE  cq              IS 'Competency question (CQ)';
COMMENT ON COLUMN cq.cq_id        IS 'CQ identifier';
COMMENT ON COLUMN cq.document_id  IS 'Denormalized FK. Derivable from paragraph_id but kept for query convenience';
COMMENT ON COLUMN cq.paragraph_id IS 'FK of the source paragraph. Basis for provenance tracking';
COMMENT ON COLUMN cq.run_id       IS 'Pipeline run ID';
COMMENT ON COLUMN cq.question     IS 'Competency question text';
COMMENT ON COLUMN cq.answer       IS 'Paragraph-based answer. Single-fact principle (after decomposition)';

COMMENT ON COLUMN cq_draft.original_question IS 'Question just before regeneration. Kept so reviewers can see what changed';
COMMENT ON COLUMN cq_draft.original_answer   IS 'Answer just before regeneration. Kept so reviewers can see what changed';
