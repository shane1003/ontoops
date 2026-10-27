# Setup

Everything an experiment needs, step by step. Recomputing the paper's tables
needs only step 2 (see the README); the other steps are for running the
experiments again.

## 1. Tools

- [uv](https://docs.astral.sh/uv/)
- Docker with the Compose plugin (`docker compose version` must work)
- An NVIDIA H100-class GPU for the embedding server
- Two OpenAI-compatible servers serving `google/gemma-4-31B-it`: one for CQ
  generation (`max_model_len` 16384, reasoning parser `gemma4`) and one for
  extraction and TBox construction (`max_model_len` 32768, served as
  `local-model`). Local servers and external OpenAI-compatible providers both
  work; the paper's runs used local servers.

## 2. Python environment

```sh
uv sync
```

## 3. Configuration (`.env`)

`.env.example` is the only configuration file. Copy it to `.env` (git-ignored):

```sh
cp .env.example .env
chmod 600 .env
```

The database password keeps its default (`change-me`): the stack's ports bind
to 127.0.0.1 only. Change it if you like.

Then set section 1 ("Machine") for your machine: the two LLM addresses
(`CQGEN_LLM_BASE_URL`, `ONTOAGENT_LLM_BASE_URL`, `ONTOAGENT_LLM_ROOT_URL`), the
GPU index (`TEI_GPU`), host ports if they collide, and `ANTHROPIC_API_KEY` only
for experiment 4. Leave section 2 ("Stack settings") as it is: those are the
values the paper's runs used, and the environment check fails when `.env`
differs from them.

When a newer `.env.example` adds keys, copy the new lines into your `.env`.

## 4. Manuals

The manuals are public but not redistributed here. Obtain each PDF listed in
`corpus/SOURCES.yaml` by title and edition, save it under `corpus/pdf/` with
the file name given there, then prepare them:

```sh
uv run evaluation prep
```

`prep` checks every SHA-256 before it reads a page, cuts the body pages the
services receive, and extracts the glossaries and indexes.

## 5. Environment check

```sh
uv run evaluation envcheck --profile run
```

Fix every line that is not OK (most live in `.env`) and run it again.

## 6. Stack

```sh
uv run evaluation up      # build the images, start the stack, wait until healthy
```

`evaluation up` builds the service images from `services/`. `evaluation run`
then starts only from a tagged, unmodified checkout, and takes
`--allow-untagged` otherwise; either way the manifest records the commit, any
changed files and the stack settings, so a run is traceable to its source.
Stop the stack with `uv run evaluation down`
(`--volumes` also drops its data).

Continue with "Experiments" in the README.

## Where the stack differs from production

Each deviation except tracing is a value in section 2 of `.env.example`.

| Setting | Production | Here | Why |
|---|---|---|---|
| CQ generation output cap | 15000 tokens | 8192 | 15000 on a 16384-token server leaves 1384 tokens of input, so every chunk above about 6000 characters fails (about 7% of the IPB body). Outputs are a few hundred tokens, so no successful call changes |
| Lookup index refresh | 300 s | 30 s, and the runner waits 60 s before probing | The lexical index is an in-memory copy; probing right after a run would measure a stale index |
| OOPS! pitfall scan | on | off | An advisory scan with no service in this stack |
| Langfuse tracing | on | off | Observation only, it changes no output. The services keep their optional Langfuse hook, but this repository provides no Langfuse server; `docker-compose.yml` passes empty `LANGFUSE_*` keys |

## Notes

- The LLM servers run at temperature 0 but are shared and batched, so
  repeated runs are not bit-identical; that variation is what experiment 1
  measures. Each cell records the servers' request counters around it.
- The embedding image is the Hopper build of text-embeddings-inference; other
  GPUs need another tag and may shift embeddings slightly.
- FalkorDB is licensed under SSPL v1; it runs here as a local store and is
  never offered as a service.
