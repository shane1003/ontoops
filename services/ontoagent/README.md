# ontoagent — CQ → TBox → ABox integrated pipeline

This repository packs the three stages that take a CQ (Competency Questions) document to a
loaded domain ontology into **one FastAPI app** (port **8602**). The modules, originally three
repositories, are separated by path prefix and deployed as one image on one port.

```
 CQ document (JSON)
   │  POST /extract                       conceptualizer (formerly cq2component)
   ▼
 TBox components + ABox individuals (no TTL; the verdict is a label, not a gate)
   │  POST /tbox/generate|update          tboxcraft
   ▼
 OWL TBox (Postgres+pgvector snapshot chain · HermiT gate · review queue)
   │  POST /abox/load                     aboxmapper
   ▼
 ABox load (surface_form resolution) + TBox graph (FalkorDB)
```

| Module | Path | Role | Details |
|---|---|---|---|
| `app/conceptualizer/` | `POST /extract` · `GET /target` | CQ → annotated TBox components + separate ABox individuals | [README](app/conceptualizer/README.md) |
| `app/tboxcraft/` | `/tbox/*` (generate·update·resolve·review·target·health) | Components → OWL TBox generation, update and review | [README](app/tboxcraft/README.md) |
| `app/aboxmapper/` | `/abox/*` (load·graph/tbox·target·ready·health) | Individuals → ABox load + FalkorDB TBox graph | [README](app/aboxmapper/README.md) |
| `app/core/` | — | Shared infrastructure: config · LLM client · language policy · Langfuse locality guard · logging | — |

- **Every LLM is local only (OpenAI-compatible server).** Hosted-provider paths were removed,
  and Langfuse tracing is disabled unless the host is local/private — the code layer closes every
  path by which document text could leave the box (the OOPS! scan has no default target either;
  unset means skipped).
- Root endpoints: `GET /health` · `GET /` (endpoint list) · `GET /target` (default classification
  domain). tboxcraft appears in the combined `/docs`; aboxmapper is a sub-app and has its own
  `/abox/docs`.

## Quick start

### Local

```bash
uv sync --dev

# Integrated API server (port 8602) — always pass --host (omitted, it binds to 127.0.0.1 only)
uv run uvicorn app.main:app --host 0.0.0.0 --port 8602

# CLI (per module)
uv run conceptualizer extract <cq.json>                 # CQ → components + individuals
uv run tboxcraft generate --components <comps.json>   # components → TBox
uv run aboxmapper <comps.json>                        # individuals → ABox load
```

Lint:

```bash
uv run ruff check --fix . && uv run ruff format .
```

### Docker

The image defaults to the resident FastAPI server (8602). HermiT reasoning is not a separate service; it
runs in-process on the **JRE built into the image**. In this repository the root `docker-compose.yml` builds the image and
starts it with Postgres, FalkorDB and TEI, and one-shot steps in that compose file apply the TBox schema and the ABox migration.

## Environment variables (`.env`)

In this repository the root `docker-compose.yml` sets the values. Prefixes by module:

| Prefix/variable | Owning module | Key points |
|---|---|---|
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` | conceptualizer | Local LLM server address / key (a dummy is fine, empty is not) |
| `CONCEPTUALIZER_*` | conceptualizer | Model · domain · concurrency · log level (the old `CQ2COMPONENT_*` and `ONTOAGENT_*` are **silently ignored**) |
| `TBOXCRAFT_*` | tboxcraft | DB DSN · embeddings · validation LLM · OOPS! endpoint, etc. |
| `FALKORDB_*` / `ONTOLOGY_BASE_IRI` | aboxmapper | TBox graph store / default namespace |
| `LANGFUSE_*` | shared | Tracing (optional) — local/private hosts only |

**Note**: if `TBOXCRAFT_DATABASE_URL` is set but the DB is unreachable, the app starts degraded with only the
`/tbox` DB routes returning 503 (CRITICAL log; stays that way until restart) — the same shape as an unset DSN.
`/extract` and `/abox` are unaffected.

## Folder layout

```
Dockerfile                          # single image: API server (8602) + CLIs
config/
  settings.yaml                     # conceptualizer settings (model · paths · chunk_size)
  prompts/classifier.txt            # classification prompt
data/                               # runtime outputs (created at run time, untracked)
app/
  main.py                           # app assembly: router includes + /abox mount + root endpoints
  core/                             # shared: config · llm/ (client · tracing) · lang · tracing · logging_setup
  conceptualizer/                     # module 1 — see its README
  tboxcraft/                        # module 2 — see its README
  aboxmapper/                       # module 3 — see its README
```

## Runtime requirements by module

| Module | Required | Optional |
|------|------|------|
| conceptualizer (`/extract`) | Local LLM (OpenAI-compatible) + `OPENAI_API_KEY` (a dummy is fine) | Langfuse |
| tboxcraft (`/tbox/*`) | Postgres+pgvector (`TBOXCRAFT_DATABASE_URL` — without it only the DB routes return 503) | TEI embedding server (GPU, in the stack) · OOPS! (local endpoint only; off in this repository's stack) · HermiT (in-process on the **JRE built into the image**) · vLLM validation LLM · Langfuse |
| aboxmapper (`/abox/*`) | The same Postgres as tboxcraft (`TBOXCRAFT_DATABASE_URL`) | FalkorDB (TBox graph), `ONTOLOGY_BASE_IRI` (without it, `?base_iri=` is required on every call) |
