# tboxcraft

Build and maintain a domain **OWL TBox** from externally-produced, adjudicated term
components, then serve it with a Postgres/pgvector store and a local embedding model.

Each batch flows through one layered ingest pipeline; the first batch (empty store)
bootstraps the ontology, and later batches extend it without a separate mode:

- **Embedding** — assemble a deterministic gloss per term (name + notes + property
  domain/range + sampled CQs, no LLM) and embed it with a local model. Cold start whitens the
  batch and seeds the mean; later batches center against that stored mean.
- **Clustering (attach-only)** — existing DB concepts are **frozen anchors**. Each new term
  either **attaches** to its nearest live anchor or, with the leftovers, **self-clusters** into a
  new concept. Anchors are never re-partitioned, so existing IRIs stay stable.
- **DomainTermApproval** — an LLM domain-term verifier judges every cluster and only the
  **risky** ones reach a human review queue; the rest apply automatically.
- **TboxMatcher** — apply one of four outcomes to the DB, then re-pass the same validation gate
  (all-or-nothing) regardless of outcome:
  - **ADD** — a genuinely new concept: mint one IRI + representative label per cluster.
  - **SynonymADD** — the term is an existing concept: add its surface form to the mapping
    table; the IRI is untouched.
  - **UPDATE** — renew a concept's IRI to a more appropriate official term. Rare and
    **double-gated** (verifier + human): the IRI is re-minted, every reference migrated, and the
    old IRI left as an `owl:deprecated` tombstone.
  - **DELETE** — human-gated removal. A dangling-orphan scan collects structurally-broken
    concepts as delete proposals; removal is **deprecate-in-place** when still referenced, else a
    hard remove.
- **Validate** — a standalone module runs Syntax + HermiT (local, hard gate) + OOPS! (advisory).
- **Serve** — an HTTP API and a local e5 model, backed by a Postgres/pgvector store whose schema
  is [`db/schema/01_tbox_schema.sql`](db/schema/01_tbox_schema.sql).

The canonical TBox lives in the DB as a **versioned Turtle snapshot** (`ontology_snapshot`); the
relational tables (`tbox_term` with its embedding, `surface_form`, `term_cq_map`, `source`,
`review_queue`) are the resolution / vector / provenance / workflow layer.

## Input

The pipeline consumes a `component.json` document of the shape `{ "components": [...],
"individuals": [...] }`. Only the TBox **`components`** are used; the ABox **`individuals`**
block is ignored. Competency questions are **not** part of this file — they are read from the DB
`cq` table that CQ generation owns (keyed by `cq_id` = a component's `source_cq_ids`). The gloss
is viable from name + notes alone, so CQs are an optional enrichment.

## Requirements

- Python 3.12, [uv](https://docs.astral.sh/uv/)
- A JRE 17+ (HermiT reasoning) — `java` on PATH or `TBOXCRAFT_JAVA_EXE`
- Docker + Docker Compose (for the service stack)
- An NVIDIA GPU + Container Toolkit (for the TEI embedding server)

## Install

```bash
cd services/ontoagent
uv sync      # --extra local for in-process e5
```

## CLI

```bash
# Generate (file mode): bootstrap a TBox from the first batch
tboxcraft generate --components path/to/component.json --output data/output

# Update (DB mode): route a later batch against the accumulated ontology
tboxcraft update --components path/to/component.json
```

The repository ships no application input of its own — a batch comes from a producer
(`ontoagent` for documents, a relational-database producer for relational sources).

Competency questions come from the DB `cq` table; load them upstream if you
want CQ-enriched glosses and concept↔CQ provenance links.

`generate` writes a validated `tbox.ttl`, a validation report, and a human-reviewable merge
report. `update` reads/writes the DB and prints per-outcome counts; if no ontology exists yet it
**bootstraps via generate** (the first batch) rather than failing, so callers need not decide
generate-vs-update. A syntax or HermiT-inconsistency failure blocks the return in both modes.

## Service stack (docker-compose)

This module is served by the `ontoagent` container of the repository-root `docker-compose.yml`
(the unified FastAPI app; this module's routes live under **`/tbox/*`**). The same file runs
Postgres+pgvector, the TEI embedding server and a one-shot `tbox-schema` container that applies
`db/schema/*.sql`. All values come from the repository-root `.env` (template: `.env.example`).

| Service | Host port | What |
|---|---|---|
| `ontoagent` | `127.0.0.1:${ONTOAGENT_PORT:-8082}` (container 8602) | unified FastAPI — `/tbox/*` is this module |
| `postgres` | `127.0.0.1:${POSTGRES_PORT:-55432}` | Postgres + pgvector |
| `tei` | none (in-network `http://tei:80`) | TEI serving the e5 embedding model |

```bash
uv run evaluation up      # from the repository root: build, start, apply schemas, wait healthy
```

The api needs the database up and the schema applied (it fails closed if the DB is
unreachable). Tests and a host-run CLI only need the DB running.

### Endpoints (by role)

| Endpoint | Role | Purpose |
|---|---|---|
| `POST /generate` | admin / ingest | bootstrap the ontology (batch 1) explicitly |
| `POST /update` | admin / ingest | route a batch — **bootstraps via generate if no ontology exists yet** |
| `GET /review/pending` | human reviewer | list queued gray / relabel / delete items (with any LLM verdict) |
| `POST /review/verify` | human reviewer | attach the advisory LLM verdict to pending review items |
| `POST /review/{id}/resolve` | human reviewer | apply a decision (merge/new/relabel/delete/reject) |
| `POST /resolve` | consumer / query | surface→IRI resolution with a confidence gate |
| `GET /health` | — | liveness |

```bash
curl -s localhost:8082/tbox/health

# Resolve a term to a canonical IRI (exact surface, then confident k-NN)
curl -s -XPOST localhost:8082/tbox/resolve -H 'content-type: application/json' \
     -d '{"term": "operation order", "k": 5}'

# Route a batch — POST the components inline (bootstraps if the store is empty)
curl -s -XPOST localhost:8082/tbox/update -H 'content-type: application/json' \
     -d '{"components": [ {"name": "InfantryBrigade", "kind": "class", ...} ]}'

# Human review lifecycle
curl -s localhost:8082/tbox/review/pending
curl -s -XPOST localhost:8082/tbox/review/42/resolve -H 'content-type: application/json' \
     -d '{"action": "merge", "target_iri": "http://example.org/military/class/InfantryBrigade"}'
```

## Return values

### `POST /update` — 200

The caller consumes the improved TBox (`ttl`) and reconciles raw class names to canonical IRIs
(`mapping_delta` + `pending`).

| Field | Type | Meaning |
|---|---|---|
| `ok` | bool | always `true` on 200 |
| `ttl` | string | the improved TBox for this batch (serialized Turtle) |
| `version_id` | int | the DB snapshot id this update produced |
| `domain` / `domain_source` | string | the domain this batch was actually judged against and its source (`request`/`env`/`built-in`) — `domain_description` in `/tbox/target` is only the default; what a batch was judged against is reported only here |
| `n_candidates` | int | routed candidates in the batch |
| `n_new` / `n_merged` / `n_relabel` / `n_gray` | int | per-outcome counts (ADD / SynonymADD / UPDATE-queued / gray-queued) |
| `mapping_delta` | list | **row-level** raw name → IRI for every *resolvable* candidate (see below) |
| `pending` | list | candidates sent to human review — **not resolvable yet** (see below) |

`mapping_delta[i]` — one per ADD/SynonymADD candidate, all kinds (classes **and** properties):

```json
{ "candidate_uid": "class:InfantryBrigade", "name": "InfantryBrigade",
  "kind": "class", "outcome": "merge", "iri": ".../class/InfantryBrigade" }
```

`pending[i]` — one per gray/relabel candidate:

```json
{ "candidate_uid": "class:Foo", "name": "Foo", "kind": "class",
  "entry_type": "gray", "nearest": [{"iri": "...", "score": 0.71}] }
```

**Partition invariant:** `len(mapping_delta) + len(pending) == n_candidates` — every routed
candidate appears exactly once. (Components filtered upstream by `verdict` never enter routing,
so there is no "rejected" bucket here.)

On a validation-gate failure the call returns **422** with `detail = { "gate_reasons": [...] }`
and **nothing is committed**. The Python `update()` returns an `UpdateResult` with these same
fields (`ttl`, `mapping_delta`, `pending`, …).

**Cold start:** with no ontology yet, `/update` bootstraps via generate and returns the same
shape — every concept as a `"new"` (or in-batch `"merge"`) `mapping_delta` row, `pending: []`.

### `POST /resolve` — 200

The **server owns the acceptance decision** so callers never re-judge similarity (that would
diverge from the whitened space the TBox was built in).

| Field | Type | Meaning |
|---|---|---|
| `iri` | string\|null | the resolved IRI, or `null` when there is no confident match |
| `match` | string | `"exact"` (surface hit) · `"knn"` (confident nearest) · `"none"` |
| `confident` | bool | whether `iri` is a confident match |
| `score` | float\|null | top k-NN cosine (in the whitened space) |
| `threshold` | float | the server-owned acceptance cutoff (`TBOXCRAFT_RESOLVE_THRESHOLD`) |
| `nearest` | list | `[{iri, score}, …]` — full candidate list, for inspection |

A bare `term` is also tried as an exact surface, so one field gives exact-then-k-NN. Send
`vector` to skip embedding, or `surface` for exact-only.

### `POST /review/{id}/resolve` — 200

`{ "ok": true, "action": "merge|new|relabel|delete|reject", "version_id": <int|null>,
"detail": <obj|null> }`. `detail` carries action-specific outcome — a **delete** reports
`{ "deprecated": bool, "references": [...] }` (deprecate-in-place when still referenced, else
hard-removed). A **relabel** is an IRI renewal: the concept's IRI is re-minted from the new
term, every reference is migrated, and the old IRI is left as an `owl:deprecated` tombstone.
Returns **404** if the item is gone, **409** on a stale target, a renewal IRI collision, or a
gate rejection (the item stays `pending`). Request body:
`{ "action", "target_iri"?, "label"?, "run_reasoner"? }`.

### `GET /review/pending` — 200

`{ "pending": [ { id, candidate_uid, kind, entry_type, gloss, nearest, payload, llm_verdict,
created_at }, … ] }`, oldest first. `entry_type` is `gray` (low-confidence attach), `relabel`
(UPDATE proposal), or `delete` (removal / dangling-orphan proposal). `llm_verdict` is `null`
until `POST /review/verify` fills it (advisory; the human decision stays authoritative).

### `POST /generate` — 200

`{ "ok", "ttl_path", "persisted", "n_components", "n_concepts", "mapping_delta" }` — where
`mapping_delta` is the per-component raw name → minted IRI for the initial ontology. **422**
with `detail = { "gate_reasons": [...] }` on a validation-gate failure.

### Offline model (on-machine constraint)

Document data stays on the host. TEI caches the e5 weights in the `teicache` volume at the
revision pinned by `EMBEDDING_REVISION`. The api runs with `HF_HUB_OFFLINE=1` and pins
`TBOXCRAFT_EMBEDDING_PROVIDER=remote_e5` so nothing loads the model in-process.

### Data-locality note

Localhost port binding controls **inbound** reach only. The OOPS! scan has no default endpoint —
unset means skipped, so the whole-ontology POST only ever goes to a host the operator chose (the
repository-root stack disables the scan: `TBOXCRAFT_OOPS_ENABLED=false`). Langfuse tracing must
target a **local** instance or stay disabled — a hosted `LANGFUSE_BASE_URL` would send gloss/CQ
text off-host. The domain-term verifier is likewise **local only**: an OpenAI-compatible server
at `TBOXCRAFT_LLM_URL`, as the verifier prompt carries document text; there is no hosted
provider.

### Tracing (optional)

Set `LANGFUSE_{BASE_URL,PUBLIC_KEY,SECRET_KEY}` to trace
`generate` / `update` / `resolve` (inputs, per-outcome counts, resolve match/score/threshold —
useful for tuning `TBOXCRAFT_RESOLVE_THRESHOLD`). Tracing activates **only** when the base URL
is a loopback/private host; a public host, missing keys, or a missing library disables it
(no-op), and a tracing failure never breaks the pipeline. The repository-root stack leaves
them empty.

### Schema

The schema is [`db/schema/01_tbox_schema.sql`](db/schema/01_tbox_schema.sql), idempotent and
applied with `psql` (the repository-root stack does it in the `tbox-schema` container). Term
roles live in `surface_form.form_type`: **PREF** (the representative label, one per term and
English), **ALTER** (kept synonym) and **HIDDEN** (dropped but retained for downstream text
mapping) — written by ingest (auto, verifier-judged) and by the `hide` review action. The ABox
(individuals) is populated by the aboxmapper module of the same service.
