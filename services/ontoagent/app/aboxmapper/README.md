# aboxmapper

Ontology ABox loading + TBox graph module. It has two roles:

1. **ABox loader** — reads `individuals` (the ABox) from `component.json`, resolves each class/data property
   token through `surface_form ⋈ tbox_term`, then idempotently loads `abox_individual` (individuals) and
   `abox_data_assertion` (values).
   - Handles class typing (rdf:type) + data properties only (object properties excluded).
   - The loader is a **consumer** of the TBox — choosing preferred terms and deduplication belong to the TBox pipeline.
   - Surface-form matching is case-insensitive (`lower(surface_text)` index); only `status='ACTIVE'` resolves.
   - An individual is identified by `(class_id, identity_key)` — with a name it merges across sources; without one
     it falls back per source to `src:<source_id>`. Accumulate + upsert, so reruns do not pile up duplicates.
   - Provenance is **source-neutral**: input `source_ref` (singular) → `source_id`
     (NOT NULL, FK → `source`). A document CQ and an RDB row are tracked the same way. The document path still
     accepts the legacy `source_cq_id` for now, which the core interprets as `doc:cq:<cq_id>`.
2. **TBox → FalkorDB graph loader** — reads `TBox.ttl` (the canonical ontology snapshot) and builds the
   TBox knowledge graph (classes + sub/superclass + object property relations) in FalkorDB as **one named graph per document**. Individuals (ABox) are not
   put in the graph; they are queried where they are loaded, in `abox_individual`.

Both roles are available from the CLI and FastAPI.

> The TBox tables (`tbox_term`/`surface_form`/`source`, …) are created by the **tboxcraft module's** schema
> (`app/tboxcraft/db/schema/`). The ABox tables are owned by this module (see Schema / migration below).

## Configuration

Set via environment variables (or `.env`). In this repository the root `docker-compose.yml` sets them
from the repository root `.env` (template: the root `.env.example`).

```
# ABox loader → Postgres (tboxcraft)
TBOXCRAFT_DATABASE_URL=postgresql://tboxcraft:tboxcraft@localhost:5432/tboxcraft

# Default document namespace — the scope of term resolution and the base for minting individual IRIs.
# A load request's ?base_iri= overrides it. A trailing '#' or '/' is read as the separator.
ONTOLOGY_BASE_IRI=http://example.org/military#

# TBox graph loader → FalkorDB (named graph = document)
FALKORDB_HOST=falkordb
FALKORDB_PORT=6379
FALKORDB_GRAPH=ipb
```

## Running locally

```sh
uv sync
uv run aboxmapper --dry-run       # resolve and report only (DB unchanged)
uv run aboxmapper                 # load
```

## API

This module's router is **mounted under the `/abox` prefix** in the integrated app (`app.main:app`, port **8602**):

```sh
uv run uvicorn app.main:app --host 0.0.0.0 --port 8602 --reload
```

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness (no DB connection) |
| GET | `/ready` | Readiness — checks the DB is reachable |
| GET | `/target` | **Where it writes** — Postgres and FalkorDB targets (password excluded) |
| POST | `/load` | Loads the individuals in the **JSON body**. `dry_run` and `base_iri` are query parameters |
| POST | `/load/upload` | Loads an uploaded `component.json` file (multipart, same `?base_iri=`) |
| POST | `/graph/tbox` | Builds the FalkorDB TBox graph from TBox TTL (request body). `?graph=` selects the target |
| POST | `/graph/tbox/upload` | Builds the FalkorDB TBox graph from an uploaded `.ttl` file (same `?graph=`) |

`/load` takes the individuals as a JSON body — an array, `{"individuals": [...]}`,
or a whole component.json (its `individuals` are used); any of these forms is accepted.

The response is a load report:
`base_iri` (the namespace actually used), `individuals` (input count), `individuals_upserted` (distinct individuals in the ABox),
`assertions_built` / `assertions_inserted` (newly inserted),
`assertions_deduped` (already present and skipped — normal) / `assertions_dropped` (lost because the subject was not loaded — must be 0),
`object_assertions_received` / `object_assertions_skipped` (relations between individuals are outside the storage scope, so
as many as are received are skipped — counted so this is not misread as lossless), `miss`, `ambiguous`, `skip_counts`, `skips`.

#### Counts and lists are separate

`skips` is a **diagnostic list**; `skip_counts` is the **full count per reason**. Counts are not derived
from the list — if they were, a decision about response size would silently lower contract numbers such as `miss`.

Lists are handled differently by reason.

| Reason | List | Why |
|---|---|---|
| `miss` · `ambiguous` | **kept in full** | Each token that failed, and why, needs individual action, and this is a gate contract |
| `empty` | **sample of at most 50** | All have the same shape, so individual diagnosis has no value (precedent: `missing_sources`) |
| Others (`null_value`, `unidentifiable`, etc.) | kept in full | Rare and individually actionable |

`skip_counts` is never truncated, so `miss` and `ambiguous` are always exact even when the list is a sample.
`individuals_skipped` also counts everything, so the identity below is unaffected.

#### Individual counts decompose the input completely

```
individuals = individuals_upserted + individuals_deduped + individuals_dropped + individuals_skipped
```

Same form as the value side (`built = inserted + deduped + dropped`). Back-computing the difference between
input and loaded counts by hand **cannot tell a merge from a loss**, and that distinction is exactly why this
report exists.

| Count | Meaning |
|---|---|
| `individuals_upserted` | Distinct individuals left in the ABox |
| `individuals_deduped` | **Merged** because they share a natural key with another input (normal) |
| `individuals_dropped` | A subject was built but did not come back from the write (**real loss**, must be 0) |
| `individuals_skipped` | No subject could be built — each comes with a `scope=individual` skip |

The skip reasons counted in `individuals_skipped` are `miss` and `ambiguous` (class unresolved), `unidentifiable`
(neither a name nor a source), and **`empty`** (has a source but neither a name nor values, so there is nothing to store). `empty`
is common on real outputs, and previously it vanished without appearing in any count.

### Document namespace — `?base_iri=`

**One document is one KG**, and the boundary is the **IRI namespace**, not a DB or schema. So the document's
base IRI is given at load time. The env `ONTOLOGY_BASE_IRI` remains as the default and **the request
wins** (same policy as `?graph=`).

```sh
curl -X POST 'http://localhost:8602/abox/load?base_iri=http://example.org/IPB%23' \
  -H 'Content-Type: application/json' -d @individuals.json

uv run aboxmapper --base-iri 'http://example.org/IPB#'
```

The base IRI scopes **both sides** of the identity scheme.

- **Term resolution**: only `tbox_term` rows under that namespace are candidates. With more than one document, the same
  name (`Process`) exists in several namespaces at once; without the scope there would be 2 candidates and the token would be
  **dropped** as `ambiguous` — loading the second document would break the first document's ABox.
- **Individual IRI minting**: new individuals are created under that namespace. If classes were split by namespace but
  individuals were not, the identity scheme would be only half done.

Prefix comparison uses `starts_with`, not `LIKE` — if the base IRI contains `%` (percent-encoding) or `_`,
`LIKE` reads them as wildcards and silently widens the scope. No separator is appended either (the prefix is used as is,
not `'/%'`) — because a hash namespace's base already ends with `#`.

**If the namespace has no ACTIVE terms while the store does have some, the load fails with 422.** Left alone,
every token would fall to `miss` and the result would be "a success that loaded nothing"; a wrong base IRI
must not slip through that way.

#### Individual IRI minting rules

The target output is a hash-namespace ontology, so `base` ends with `#`. Minting follows the style the base
declares — the IRI is stored in `abox_individual.id`, so it is hard to undo.

| base | Result |
|---|---|
| `http://x/IPB#` | `http://x/IPB#individual_<uuid>` |
| `http://x/mil/` | `http://x/mil/individual/<uuid>` |
| `http://x/mil` | `http://x/mil/individual/<uuid>` (unchanged behavior) |

The fragment local-name rules are agreed with the TBox side — classes `UpperCamelCase`, object and data
properties `lowerCamelCase`, individuals `individual_<uuid>`. `individual_` starts lowercase and so shares the
property pattern, but the trailing UUID means no actual collision.

#### base IRI hygiene

It must be an absolute IRI; backticks, whitespace or control characters are **rejected with 422**. A trailing `#` or `/` is
a separator and is fine, and in no case is the base **silently normalized** — minted IRIs are stored, so a silently
fixed base would become a permanent identity that cannot be reproduced.

### Checking the load target — `GET /target`

The URL alone does not tell which store an instance writes to. This endpoint exists so you can compare it with the
expected target **before** loading and avoid loading into the wrong place.

```jsonc
GET /target
{
  "postgres": {"host": "db", "port": 5432, "dbname": "ontology", "user": "abox"},
  "falkordb": {"host": "falkordb", "port": 6379, "graph": "ipb"},       // null when no default graph is set
  "default_base_iri": "http://example.org/IPB#",                      // default. null when unset
  "default_base_iri_source": "env"                                      // "env" | null
}
```

`default_base_iri_source` is where that default came from. **This service has no built-in default** — a load
that arrives without a namespace is not made up but fails with 422, so the value is `"env"` or `null` and never
`"built-in"`. The sibling service (tboxcraft) does have a built-in default, and the field lets consumers read both
responses with the same code, without branching.

When the env is set but its value is invalid, the report is `default_base_iri: null` + `source: "env"` —
if it looked the same as "not set", the misconfiguration would hide.

`default_base_iri` is, as the name says, a **default** — `?base_iri=` on `/load` overrides it, and this
endpoint cannot know what a later request will use. **The namespace actually used is in that load's
response** (`LoadResponse.base_iri`).

Postgres details are read from the **actual connection** (`conn.info`), **not from environment variables** — with `--dsn`
a process can point somewhere other than the env, so echoing the env would make the gauge lie.
No field includes the password.

### A failed precondition is 422, not an outage (503)

If an individual's `source_id` is **not** in the `source` registry, the load fails, answering `422`.
503 means "try again later", but an unregistered source is not fixed by retrying; it is fixed when upstream registers it.

The check runs once, **before** any write (it also works under `dry_run`), so instead of the transaction dying on the
first row it returns **everything that is missing**. Nothing is written.

```jsonc
422 {
  "detail": {
    "code": "unregistered_sources",       // branch on this field instead of parsing strings
    "message": "...",
    "missing_sources": ["doc:cq:a1b2…"],  // lists at most 50
    "n_missing_sources": 12,              // full count, not truncated
    "n_unsourced_individuals": 0,         // individuals with no source at all
    "n_unsourced_values": 0,              // values with no source for the same reason
    "n_individuals": 120
  }
}
```

- Only **source_ids that would actually be written** are checked. Individuals skipped because their class did not
  resolve never reach the FK, so their sources do not block the whole load.
- Individuals and values are **counted separately.** Individuals with the same `(class, identity_key)` fold into one, but every value
  remains, so the unsourced values of an individual that was folded away do not show in the individual count alone (a hard-to-reproduce
  failure that was caught or not depending on input order). The upstream fix is the same — a missing `source_ref`.
- If the `source` table itself is missing, the answer is `503` — not an input problem but a deployment-order problem
  (the tboxcraft schema, which creates `source`, must be applied before `aboxmapper-migrate`).
- Unregistered sources are not absorbed into `skips` with a 200. Skipping them silently would make a half-filled
  store a "success".
- The CLI (`aboxmapper`) follows the same rule: it exits with **exit code 3** and logs the list of unregistered sources.

There are three more 422s of the same family — branch on `detail.code` for all of them.

| `detail.code` | Meaning |
|---|---|
| `unregistered_sources` | A source missing from the `source` registry (upstream must register it first) |
| `invalid_base_iri` | The base IRI is not absolute or contains backticks, whitespace or control characters |
| `base_iri_required` | Neither the request nor the env provides a base IRI |
| `empty_term_namespace` | The namespace has no ACTIVE terms (wrong base IRI, or TBox not loaded) |

## TBox → FalkorDB (KG)

Reads TBox.ttl (the canonical ontology snapshot) and builds a **TBox knowledge graph** in FalkorDB (OpenCypher/Redis).
Individuals (ABox) are not put in the graph — they are queried where they are loaded, in Postgres.

- Connection: `FALKORDB_HOST` / `FALKORDB_PORT` / `FALKORDB_GRAPH`
- One FalkorDB instance holds **one named graph per document** (`ipb`/`adp`), and each run writes
  to only one graph (other documents' graphs are left untouched).
- **The request wins** for the target graph — `POST /graph/tbox?graph=ipb`, or `--graph ipb` on the CLI.
  Without it, the deployment default `FALKORDB_GRAPH` is used. One instance can handle several documents, so
  there is no need to run a separate container per document type.
- A misspelled graph name **silently creates a new graph.** The report therefore returns the name of the graph it actually
  wrote (`graph`); compare it with the expected value from `/target`.
- The graph holds **classes + sub/superclass + object property relations**. Data properties (xsd range) and
  individuals are left out. A class's `name` is the last IRI segment (no `/class/` prefix).
- Object properties are written as **a single relation type `OBJECT_PROP`** (iri and name on the edge) — no type is created
  per property (Cypher cannot take a relation type as a parameter, and thousands of types would defeat schema
  lookups). Consumers filter by `name`: `MATCH (a:Class)-[r:OBJECT_PROP {name:$p}]->(b:Class)`.
  The number of properties that got no edge because domain or range was missing is exposed in the report as `skipped_props`.

Graph model:

```
(:Class {iri, name})                       # name = last IRI segment
(:Class)-[:SUBCLASS_OF]->(:Class)          # child -> parent
(:Class)-[:OBJECT_PROP {iri, name}]->(:Class)   # domain -> range (domain×range per property)
INDEX ON :Class(iri)
```

Response `GraphReport`: `graph` (the named graph actually written), `classes`, `subclass_edges`,
`object_edges`, `skipped_props`, `statements`.

```sh
curl -X POST 'http://localhost:8602/abox/graph/tbox?graph=ipb' \
  -H 'Content-Type: text/turtle' --data-binary @path/to/tbox.ttl

uv run aboxmapper-graph path/to/tbox.ttl --graph ipb
```

## Schema / migration

The ABox tables (`abox_individual`, `abox_data_assertion`) are **owned by this module**; their DDL is in
`app/aboxmapper/sql/abox_schema.sql`. The TBox tables (`tbox_term`, `surface_form`, …) are owned by the
TBox pipeline and are only read, from **the same DB**.

**DDL is separate from the application runtime** — the API/CLI issue no DDL, so they can run with
data-write privileges only. Schema creation is the job of a one-shot migration (idempotent):

```sh
uv run aboxmapper-migrate            # create if missing, leave as is if present
uv run aboxmapper-migrate --wait 60  # wait for a DB that is still starting
```

### Injecting the target DB from outside (DSN)

To choose the target DB at run time without editing `.env`, pass a psycopg conninfo
string to `--dsn`. Both URL and keyword forms are accepted.

```sh
# URL form
uv run aboxmapper-migrate --dsn "postgresql://abox:PW@db.example.net:5432/tboxcraft"

# Keyword form — passwords with special characters need no URL encoding
uv run aboxmapper-migrate --dsn "host=db.example.net port=5432 dbname=tboxcraft user=abox"

# Keep the password out of the arguments (avoids exposure in ps and shell history)
PGPASSWORD='...' uv run aboxmapper-migrate \
  --dsn "host=db.example.net port=5432 dbname=tboxcraft user=abox"
```

The same applies when running the image — connection details enter only on the command line and stay out of the image.

```sh
docker run --rm --network "$DB_NETWORK" -e PGPASSWORD ontoagent:latest \
  python -m app.aboxmapper.cli.migrate --wait 60 \
  --dsn "host=db port=5432 dbname=tboxcraft user=abox"
```

Without `--dsn`, `TBOXCRAFT_DATABASE_URL` (and `.env`) is used as before. Either way it logs
`migrating <user>@<host>:<port>/<dbname>` at start, so the log shows which target received the DDL.
Only this step needs DDL privileges; the API/CLI still need only write
privileges.

In this repository's root `docker-compose.yml`, the one-shot `abox-migrate` service completes first
and then `ontoagent` starts (`depends_on: service_completed_successfully`). Loading with the migration
skipped makes `/load` return 503 with "ABox schema missing — run the migration step first".

## Docker

The image is the ontoagent image (`services/ontoagent/Dockerfile`); the root `docker-compose.yml` builds it
and wires Postgres and FalkorDB on the compose network (`TBOXCRAFT_DATABASE_URL=...@postgres:5432/...`,
`FALKORDB_HOST=falkordb`).

- Inside a container, `localhost` means the container itself, not the host, so do not use
  `localhost`/`127.0.0.1` as the host.
