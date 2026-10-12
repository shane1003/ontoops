# conceptualizer — DomainSpecificClassifier (formerly cq2component)

A single module that reads CQs (Competency Questions, JSON), **decides whether each item is a TBox element (concept
schema) or an ABox label (data)**, and returns **TBox components and ABox individuals as two separate outputs**.
In the integrated app it owns `POST /extract`.

When CQs have been drawn from a document (by a separate module), conceptualizer reads
those CQs, extracts components, judges TBox/ABox from the domain's point of view and splits them into two bundles. It does not
produce TTL. The later stages (TBox authoring → ABox attachment) are the responsibility of [tboxcraft](../tboxcraft/README.md) and
[aboxmapper](../aboxmapper/README.md) and lie outside this module.

## Architecture — a single classification module

```
 CQ document (JSON)
   │
   │ ① chunk fan-out (chunk_size) — chunks run concurrently (CONCEPTUALIZER_CHUNK_CONCURRENCY, default 8)
   ▼
 per chunk: component extraction → DomainSpecificClassifier (domain layer + TBox/ABox verdict) → individual extraction
   │
   │ ② reintegration fan-in
   │   - exact (kind,name) merge (deterministic) — no semantic merging
   │   - individuals are concatenated
   ▼
 ├─► TBox components (AnnotatedComponent[])
 └─► ABox individuals (Individual[])
```

**Core principles**

- **No TTL.** Only the two outputs, TBox components and ABox individuals, are produced separately (shape preserved).
- **DomainSpecificClassifier is limited to the CQ batch.** It does not read a master ontology; it only attaches
  a domain layer + TBox/ABox verdict to the components it is given.
- **Large documents are chunked.** To cope with the local LLM's context window, the input is read
  in chunk_size units and reintegrated.
- **Reintegration merges exactly, nothing more.** Only `(kind, name)` pairs identical to the character are merged deterministically, so no
  name is discarded. When the TBox/ABox verdict differs between chunks, the merged verdict is set to `conflict` and a
  `verdict_conflict` flag records the competing verdicts and their sources.
- **No semantic merging or preferred-term selection.** That belongs downstream, which owns the global TBox.
  Merging here would keep the discarded names from ever reaching downstream, and individuals that refer to those names
  would fail to resolve.
- **`verdict` is a label, not a gate.** Whatever the verdict, the component is passed on as is —
  individuals refer to components by name, so filtering would leave individuals with nothing to attach to.
- **The classifier judges by structural role.** Each component carries "who uses me as domain/range"
  (`domain_of`/`range_of`). In OWL a property's domain/range must be a class, so this is
  a strong TBox signal. If an `abox` verdict still comes out, that is a **structural contradiction**: it is raised to `conflict`
  and the competing candidates (classifier `abox` vs structure `tbox`) are recorded as evidence in `disagreement_flags`.

## HTTP API

Mounted at the root path of the integrated app (port 8602). See the repository README for how to start it.

- `GET /target` — `{"default_domain": ...}`. A **runtime report of fact** on which domain this instance
  judges against. The request document's `ontology.domain` wins, so this value is the default used when the
  document's domain is empty. It reads the loaded settings, not an echo of the env — a wrong domain raises no
  exception and only degrades judgment quality, a silent defect, so the harness needs a place to
  check it.
- `POST /extract` — body = **the CQ document JSON as is** (`{ontology, competency_questions[]}`,
  no wrapper schema), query `chunk_size`/`language`. Response `{ components[], individuals[],
  n_unmatched, n_promoted_conflict, n_degraded_batches, domain_source }`.
  **No TTL.** 503 without an LLM key.
  - The classification domain is chosen in the order `ontology.domain` (document) > `CONCEPTUALIZER_DOMAIN` (env) >
    `agents.domain` (settings), and the response's `domain_source` echoes which one was used
    (the old practice of reusing the prefix as the domain is retired).
  - `language` — **fallback** language of the domain-layer text (BCP-47, e.g. `ko`). Used only for values
    that are not judged Hangul and for which the classifier gave no language; without it, `und` is emitted. Check
    that it arrived via `correct_term_lang`/`notes_lang` in the response (a value not shaped like a tag is ignored
    + WARNING). See the table in the language section below for the decision order.
  - `n_unmatched` — number of components left with the default verdict and no domain layer attached. Equals the sum
    of `(unmatched N)` in the chunk logs (every path prints that line).
  - `n_degraded_batches` — number of chunks whose LLM call/parse broke, leaving the whole domain layer empty.
    If `n_unmatched` is large and this value is not 0, the cause is that chunk, not name matching
    (a large chunk gets its response truncated at max_tokens → reduce `extraction.chunk_size`).
  - `n_promoted_conflict` — number of properties' domains/ranges that were judged abox and so promoted to conflict.
  - All three are **per-chunk sums** (before reintegration), so do not compare them directly with the length of `components`. When chunks
    run concurrently the per-chunk logs interleave, so the same figures are put in the response to be read there instead of from the logs.

## CLI (`conceptualizer`)

argparse entry point (`cli.py`). The only subcommand is `extract`.

```bash
uv run conceptualizer extract <cq.json>
uv run conceptualizer extract <cq.json> --config config/settings.yaml
uv run conceptualizer extract <cq.json> --components-out comps.json --individuals-out inds.json
uv run conceptualizer extract <cq.json> --language ko   # domain-layer language fallback (und if omitted)
```

Splits the CQList into `extraction.chunk_size` units, runs DomainSpecificClassifier on each chunk and
reintegrates (exact `(kind,name)` merge, individuals concatenated). **No TTL is produced.**

- Output: a **TBox component artifact** + an **ABox individual artifact** (default `data/intermediate/`,
  created at run time — local outputs that are not committed). Individuals never go into the component file, and components never
  into the individual file.
- Each individual carries the class name raw as extracted, together with property values and the **source question
  text**.

## Domain-layer language tagging

**Domain-layer text always ships with its language.** `correct_term` is paired with `correct_term_lang`, and
`notes` with `notes_lang` (BCP-47). When there is text there is always a language,
and when there is no text there is no language. Without the language, the receiver attaches its own default, so a Hangul
`correct_term` gets stamped as an `rdfs:label "..."@en` and an English note as `@ko` — leaving
false assertions in the ontology. The language is decided in this order:

| Rank | Evidence | Result |
|---|---|---|
| 1 | Hangul is the **majority** of the letters | `ko` — definitive (overrides the LLM tag) |
| 1 | Hangul present but not a majority | `mul` — mixed (ISO 639-2) |
| 2 | Tag given by the classifier | that value |
| 3 | Request `?language=` (CLI `--language`) | that value |
| 4 | None | `und` — undetermined (RFC 5646) |

Hangul is a script used only for Korean, so its presence is an **observation**, not a guess, and it therefore
overrides the model's judgment (even if the model calls a Hangul term `en`, it is `ko`). The reverse does not hold, so
no decision is made from it — Latin script is no evidence of English (romanized Korean and Spanish are both
Latin). Han characters are shared by Korean, Chinese and Japanese, so they are not counted as evidence, along with digits and symbols.
Because the rule is majority-based, a Korean sentence quoting a foreign term stays `ko`, and only a bilingual label
(Hangul and Latin together) becomes `mul`.

The policy implementation is owned by `app/core/lang.py` (`script_lang`, `normalize_lang`) — the producer (this module) and
the consumer (tboxcraft) see the same constants.

## Configuration (`config/settings.yaml`)

```yaml
llm:                                      # local LLM (OpenAI-compatible server) only
  openai_model: local-model               # local (OpenAI-compatible) model name
  openai_fast_model: local-model          # every call in the pipeline uses the fast tier
  temperature: 0.0

paths:   { intermediate_dir: data/intermediate, prompts_dir: config/prompts }

# Large CQLists fan out/fan in by chunk_size and are reintegrated.
# Keep it small so the classifier's JSON response fits within max_tokens (8192).
extraction: { chunk_size: 10 }

agents:  { domain: military }             # document ontology.domain > env > this value
```

- **`extraction.chunk_size`** — classifier read chunk size. Chunks are reintegrated after fan-out/fan-in.
- **`CONCEPTUALIZER_CHUNK_CONCURRENCY`** (env, default `8`) — how many chunks run **concurrently**.
  Chunks have no data dependencies on each other (reintegration happens after all finish), so they go to a thread pool. Results
  are reintegrated in original index order, not completion order, so the concurrency value does not change the output.
  `1` means serial. `0`, negatives and unparsable values do not mean unlimited; they fold to serial and to the default respectively
  (without a cap, a large document floods the LLM server's queue and causes timeouts).
- The pipeline's LLM calls (classification) are **all fast tier**, so `*_fast_model` is what is actually used.
- The LLM path forces `response_format={"type":"json_object"}` on responses (to stop a local model that breaks
  JSON form from causing parse failure → lost verdict → 0 individuals). If the server rejects this
  parameter, it falls back once without JSON mode.
- There is one LLM prompt, `config/prompts/classifier.txt` (classification).

### Overriding model and domain with env

These can be swapped in the container/compose without rebuilding `settings.yaml`.
Precedence is **env > settings.yaml > code default**, and **empty strings are ignored**
(the yaml value is kept even when compose's `${VAR:-}` inserts a blank).

| Environment variable | Overrides |
|---|---|
| `CONCEPTUALIZER_OPENAI_MODEL` | `llm.openai_model` |
| `CONCEPTUALIZER_OPENAI_FAST_MODEL` | `llm.openai_fast_model` |
| `CONCEPTUALIZER_DOMAIN` | `agents.domain` (classifier default domain) |

### Log level

`CONCEPTUALIZER_LOG_LEVEL` (default `INFO`). `basicConfig` exists only in the CLI and uvicorn does not touch
root, which defaults to WARNING, so progress logs were invisible when running as the API. Therefore
`app.main` sets the `conceptualizer` logger level to this value **at import time**
(when starting with `uvicorn app.main:app`, import time is the only point where our code can step in).
It does not override settings in environments that already have handlers (pytest, etc.).

- `INFO` (default) — shows chunk progress, verdict tallies, LLM calls/tokens and the completion summary.
- `WARNING` — failure/degrade signals only. **Regardless of level**, degrades, fallbacks and silent failures
  (0 matches, 0 individuals) are WARNING and always logged.

## Runtime requirements

Local LLM (OpenAI-compatible server) + `OPENAI_API_KEY` (a dummy is fine; empty raises RuntimeError) +
`OPENAI_BASE_URL` (the openai SDK reads it as a fallback). Langfuse tracing is optional — it is enabled only when the keys and
host are all present and the host is local/private (the locality guard in `app/core/tracing.py`).
