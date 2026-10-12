# cq-generation

Turns a PDF document into **Competency Questions (CQs)**: the questions an ontology
with a given purpose and scope must be able to answer. The document is converted with
opendataloader-pdf, split into intact section chunks, and an LLM writes, verifies and
refines CQs per chunk; a review gate decides which drafts are promoted, and promoted
CQs are structured into ontology terms (classes, object and data properties). All state
lives in PostgreSQL.

## Endpoints

FastAPI app `src.api:app`; the Swagger UI is at `/docs`.

| Method | Path | What it does |
| --- | --- | --- |
| POST | `/api/v1/parser-odl` | Converts an uploaded PDF with opendataloader-pdf and returns `<name>.odl.json`. |
| POST | `/api/v1/structure-odl` | Extracts the top-level skeleton (parts, chapters, appendices) from the ODL JSON with an LLM and returns `<name>.structure.json`. |
| POST | `/api/v1/chunk-odl` | Splits the ODL JSON (optionally with the structure file) into section chunks of about `budget` characters and loads them into `paragraph`. |
| POST | `/api/v1/generate` | Generates CQs for one paragraph and loads them as drafts. |
| POST | `/api/v1/refine` | Verifies one draft (ill-posed, composite question, composite answer) and fixes it until it converges. |
| POST | `/api/v1/draft-review` | Records the reviewer's action and feedback on a draft (gate 1). |
| POST | `/api/v1/regenerate` | Regenerates a draft from the reviewer's feedback and its source paragraph. |
| POST | `/api/v1/draft-verdict` | Records the reviewer's verdict on a regenerated draft (gate 2). |
| GET | `/api/v1/drafts` | Lists the drafts of a run. |
| POST | `/api/v1/promote-drafts` | Promotes the accepted drafts of a run to `cq` and returns their ids in paragraph order. |
| POST | `/api/v1/terminologize` | Structures one promoted CQ into ontology terms. |

The tables are created on startup from [sql/schema.sql](sql/schema.sql); the app does
not start without a reachable database.

## Configuration

Every field of `Settings` in [config.py](config.py) is read from the environment (or a
`.env` file in the working directory), named in upper case. The main ones:

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | `local` | `local` (vLLM, OpenAI-compatible) \| `openai` \| `anthropic` |
| `VLLM_BASE_URL`, `VLLM_MODEL` | | Endpoint (with `/v1`) and model id for `local` |
| `OPENAI_API_KEY`, `OPENAI_MODEL` | | Used with `openai` |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | | Used with `anthropic` |
| `POSTGRES_DSN` | | libpq connection string |
| `POSTGRES_CONNECT_TIMEOUT` | `5` | Seconds to wait for a connection |
| `CQ_MAX_PDF_BYTES`, `CQ_MAX_JSON_BYTES` | 20MiB, 13MiB | Upload caps |
| `ODL_CONVERT_TIMEOUT`, `ODL_CONVERT_MAX_HEAP` | `120`, `512m` | Wall-clock and heap caps of the opendataloader-pdf JVM |
| `CQ_CHUNK_CHAR_BUDGET` | `1500` | Default chunk size for `/api/v1/chunk-odl` |
| `CQ_GENERATOR_MAX_TOKENS` | `15000` | Output cap of the generation call |
| `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST` | empty | Langfuse tracing (optional) |

Temperatures, token caps, timeouts and prompt paths of each step (`ODL_STRUCTURE_*`,
`CQ_GENERATOR_*`, `CQ_VERIFY_*`, `CQ_REFINE_*`, `CQ_HUMAN_REFINE_*`,
`CQ_TERMINOLOGIZER_*`) are also settings; [config.py](config.py) explains each default.

## Running in this repository

The root `docker-compose.yml` builds this directory as the `cq-generation` service,
connects it to the stack's Postgres, and sets its environment from the root `.env`
(copied from the root `.env.example`): `CQGEN_LLM_BASE_URL` and `CQGEN_LLM_MODEL` for
the external vLLM endpoint, `CQ_CHUNK_CHAR_BUDGET`, and `CQ_GENERATOR_MAX_TOKENS`.
The container listens on 8080, published on `127.0.0.1:${CQGEN_PORT:-8081}`. The
evaluation code drives it over HTTP; start the stack with `uv run evaluation up` from the
repository root.

The image ([Dockerfile](Dockerfile)) installs a pinned-major Java runtime for
opendataloader-pdf and runs `uvicorn src.api:app` on port 8080.

For local development outside the stack:

```bash
uv sync
uv run uvicorn src.api:app --port 8080
```

## Project structure

```
.
├── config.py                 # Settings (pydantic-settings)
├── Dockerfile
├── prompts/                  # One system prompt per LLM step
│   ├── odl_structure_identify.txt, odl_structure_hierarchy.txt
│   ├── cq_generator_system.txt
│   ├── verify_ill_posed.txt, verify_composite_question.txt, verify_composite_answer.txt
│   ├── refine_ill_posed.txt, refine_split.txt
│   ├── human_refine_rewrite.txt, human_refine_reanswer.txt
│   └── terminologizer_system.txt, terminologizer_split_system.txt
├── sql/
│   └── schema.sql            # Table definitions (single source of truth)
└── src/
    ├── api.py                # FastAPI app and endpoints
    ├── client.py             # LLM client per provider
    ├── db.py                 # PostgreSQL access
    ├── odl_converter.py      # PDF -> ODL JSON (opendataloader-pdf)
    ├── odl_tree.py           # ODL JSON -> section tree
    ├── odl_structure.py      # LLM skeleton extraction
    ├── markers.py            # Numbering marker vocabulary
    ├── mcs_chunker.py        # Section tree -> intact chunks
    ├── cq_generator.py       # CQ generation
    ├── format_validator.py   # Naming checks for ontology terms
    ├── cq_refiner.py         # Verify-and-fix loop
    ├── cq_human_refiner.py   # Regeneration from reviewer feedback
    ├── cq_terminologizer.py  # CQ -> ontology terms
    ├── prompt.py             # User prompt builders
    ├── models.py             # Structured LLM output models
    ├── schemas.py            # Shared DTOs and table row models
    └── text_normalizer.py    # Unicode/whitespace normalization
```
