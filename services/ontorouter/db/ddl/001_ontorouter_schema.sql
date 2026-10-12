-- db/ddl/001_ontorouter_schema.sql
-- ontorouter's own schema. This file is the only definition of it; the
-- service applies it at start-up (idempotent, every statement IF NOT EXISTS)
-- and an operator can apply it by hand for a role without CREATE.
-- The service writes only here; the public schema is read-only to it.

CREATE SCHEMA IF NOT EXISTS ontorouter;

-- One row per (surface form, embedding space). Keying on the model id means a
-- model swap leaves the old vectors in place but invisible: they are neither
-- scored nor counted as embedded, so the backfill re-embeds under the new id.
--
-- surface_form_id references public.surface_form but carries no foreign key:
-- public is upstream's and may be recreated wholesale, which reassigns its
-- ids. surface_text_md5 is the guard against that — it records the text each
-- vector was actually produced from, so the backfill can tell a live row from
-- one whose id was handed to a different term and re-embed it. Without it a
-- recreated upstream table would leave every vector pointing at the wrong term
-- while still counting as embedded.
CREATE TABLE IF NOT EXISTS ontorouter.surface_form_embedding (
    surface_form_id    integer     NOT NULL,
    embedding_model_id text        NOT NULL,
    embedding          vector(1024) NOT NULL,
    surface_text_md5   text        NOT NULL,
    created_at         timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (surface_form_id, embedding_model_id)
);

CREATE INDEX IF NOT EXISTS ontorouter_surface_form_embedding_hnsw
    ON ontorouter.surface_form_embedding
    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS ontorouter_surface_form_embedding_model
    ON ontorouter.surface_form_embedding (embedding_model_id);
