# Changelog

All notable changes to this repository are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the version
numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-27

First tagged release of the OntoOps evaluation repository: the three method
services, the harness that runs and scores them, and the scored results of
the paper's runs. Everything below ships for the first time, so the whole
release is an addition; the notes at the end record what a reader should know
before trusting a number or re-running an experiment.

### Added

#### Method services (`services/`)

- `cq_generation` — competency-question generation. Converts a PDF to an ODL
  document, recovers its section skeleton with a validated two-step LLM pass,
  cuts it into maximal complete subtrees against a character budget, generates
  a set of CQ drafts for each paragraph, refines each draft through a three-check
  verify-and-repair loop, promotes the survivors in one transaction with an
  invariant check, and terminologizes them into the classes and properties the
  ontology is built from. A two-gate human review workflow sits between
  generation and promotion.
- `ontoagent` — conceptualizer, TBox builder and ABox loader behind one
  FastAPI application. The conceptualizer extracts annotated components and
  individuals from the terminologized CQs and labels each one TBox, ABox or
  conflict, a conflict being recorded with its evidence rather than resolved.
  `tboxcraft` builds deterministic glosses, embeds and mean-centers them,
  calibrates per-kind thresholds, clusters under a cohesion gate, mints IRIs
  and emits an OWL TBox, with a human-gated review queue for the cases it
  refuses to decide. `aboxmapper` loads individuals and data assertions and
  writes the class graph to FalkorDB.
- TBox validation runs a syntax stage (an rdflib Turtle parse when it is
  handed a `.ttl` path; the pipeline hands it the graph it just emitted), an
  OWL 2 DL punning check, a HermiT consistency check through owlready2, and an
  optional advisory OOPS! pitfall scan. The gate fails closed: a reasoner that
  could not run blocks the commit, and the response exposes `reasoner_ran` so
  a caller can tell a proven-consistent TBox from an unchecked one. OOPS!
  never blocks.
- `ontorouter` — lookup service. Resolves a term to the per-document
  ontologies that define it over three tiers: an exact tier that normalizes
  the query and expands it into separator variants, an in-memory fuzzy lexical
  tier, and an optional pgvector semantic tier that is off unless configured.
  It returns `(graph, IRI)` pointers and leaves locating the graph to the
  caller.
- Each service is a separate uv project with its own lock file, Dockerfile and
  version, vendored under `services/<name>/` from the upstream commit recorded
  in `experiments/pins.yaml`.

#### Evaluation harness (`evaluation/`)

- A Typer CLI published as the `evaluation` console script with seven
  subcommands: `prep`, `envcheck`, `up`, `down`, `run`, `score`, `aggregate`.
- `evaluation run <config.yaml>` executes every cell of an experiment config
  against the running stack. A `doc2onto` cell (the end-to-end pipeline run
  type) walks the twelve pipeline steps from document registration to graph
  upload and then exports its rows, writing each service answer as it arrives
  (`00_document.json` through `40_graph.json`, `21_tbox.ttl`) together with
  `timings.json`, `vllm.json` and `status.json`. A cell ends as `ok`,
  `parse_failure`, `inconsistent` or `timeout`, and a failed cell keeps
  everything written before the failure.
- A freeze gate on provenance: `run` refuses to start unless HEAD carries an
  exact tag and no tracked file has changed, with `--allow-untagged` as the
  documented escape hatch for pilot runs. Changes under `experiments/runs/`
  and `experiments/results/` are excluded from the dirty check, so the
  harness's own outputs never block a run.
- A run manifest (`manifest.json`) recording the run id, the config file name
  and the SHA-256 of its bytes, the git commit, tag and dirty list, the full
  contents of `experiments/pins.yaml`, the IDs of the running container
  images, each LLM server's model id, root and `max_model_len`, the run
  settings, the stack settings as the run saw them, and the harness and Python
  versions. The config file itself is copied into the run directory verbatim.
- Scorer assets travel with the run: each document's `glossary.csv`, and its
  `index.csv` when the manual prints one, are copied into
  `<run_dir>/assets/<slug>/` before the cells start, so a run directory moved
  elsewhere is still scored against the exact files its cells were scored
  with.
- `evaluation score <run_dir>` re-derives every metric row from a run's
  exported files alone, with no database, model, GPU or network. A cell that
  did not finish still produces rows carrying its status, so a failure is
  counted rather than silently omitted.
- `evaluation aggregate` groups result rows by experiment, run type, dataset
  (a manual, or `corpus` for the corpus-wide router metrics) and metric, and
  reports `n`, `n_failed`, `n_undefined`, `mean_with_zero`,
  `mean_ok`, `std_ok` and a percentile bootstrap 95% interval (1000 resamples,
  seed 0). It warns when the append-only results file holds two scorings of
  the same cell and keeps the newest.
- `evaluation envcheck --profile run` verifies Docker on `PATH`, a Dockerfile
  for every pinned service, a `@sha256:` digest on every third-party stack
  image, the required `.env` keys, the eleven frozen stack settings against
  `.env.example`, and a live `/v1/models` probe confirming each LLM server
  serves the pinned weights. `--profile score` deliberately requires nothing
  and says so.
- `evaluation prep` verifies each manual's SHA-256 before reading a page, cuts
  the body pages the services receive, and extracts the glossary and index for
  the scorer. Manuals marked `hand_checked` keep their reviewed CSVs on a
  re-run.
- `evaluation up` / `down` drive the Compose stack, `up` returning only once
  every healthcheck passes and both one-shot schema jobs have exited.
- Service clients with production-matching retry policies and contract checks:
  a timeout is never retried, a batch records a per-item error instead of
  aborting, and a 2xx answer is rejected when the echoed IRI pieces, the cited
  CQs, the verification gate, the returned Turtle, the individual count or the
  uploaded graph name disagree with what was sent.

#### Metrics and scoring

- `METRICS.md`, the written definition of every number in the paper, each
  section naming the module under `evaluation/scoring/` that computes it.
- One deliberately strict normalization behind every string comparison (NFKC,
  private-use glyphs removed, typographic quotes and dashes flattened, leading
  proponent marker removed, CamelCase split, lower case, punctuation to
  spaces, leading article removed), with no stemming, plural folding or
  synonym table, so every coverage figure is a lower bound.
- `Stab.*` — pairwise Jaccard over classes, object and data properties,
  subclass edges, domains, ranges and CQs, and a surface-form match over the
  same sets except CQs. Both compare labels rather than IRIs, because each
  repetition writes under its own namespace.
- `Cov.*` — glossary term, abbreviation and index recall against the manual's
  own back matter, plus the share of body paragraphs reached by a CQ and by
  the ontology. A manual without a glossary or index gets no row, never a
  zero.
- `Acct.*` — seven conservation identities over the pipeline's inputs and
  outputs, with `Acct.NoSilentLoss` holding only when all seven do, plus size,
  hierarchy, latency and LLM truncation counters.
- `Router.*` — namespace purity, overlap recall, glossary recall and
  agreement, typo recall and tie recall, out-of-domain leakage and provenance
  chain.
- `Faith.*` — RAGAS Faithfulness of each CQ answer against the paragraph it
  came from, including the split at a 100-character paragraph length.

#### Experiments, configs and results (`experiments/`)

- Four experiment configs: `doctrine.yaml` (the pre-registered stability,
  generality and lookup experiments), `doctrine_probe.yaml` (k=5),
  `doctrine_probe_k50.yaml` (k=50) and `faithfulness_ipb.yaml`, plus
  `ood_terms.txt`, the frozen list of fifty out-of-domain queries.
- `pins.yaml`, recording the upstream commit each vendored service was copied
  from and the three third-party stack images by digest.
- Four run snapshots under `experiments/runs/`: the paper's seven-cell
  `doctrine` run with every service response and exported table, the k=5 and
  k=50 router probes with every query and its ranked suggestions, and the
  faithfulness run with every statement, verdict and reason.
- The scored numbers: `results.jsonl` (474 rows) and `aggregates.jsonl` (328
  rows), with the k=50 probe kept in separate files
  (`results_probe_k50.jsonl`, `aggregates_probe_k50.jsonl`, 32 rows each).
- `experiments/results/external/`, a RAGAS evaluation run by another team of
  the project against the production store rather than by this evaluation
  code, documented as never mixed into this repository's own results.

#### Corpus (`corpus/`)

- `SOURCES.yaml`, naming each of the five public US Army and Joint doctrine
  manuals by title, edition and SHA-256.
- Per-manual scorer assets: `glossary.csv` for all five manuals and
  `index.csv` for the three Army manuals that print one. Beside them, not
  read by the scorer, `contents.json` for IPB and a `report.json` per manual
  recording page ranges, checksums and extraction caveats. The manual PDFs
  themselves are not redistributed.

#### Stack and configuration

- `docker-compose.yml`, bringing up the Postgres/pgvector store, FalkorDB, the
  GPU embedding server, the three OntoOps services and two one-shot schema
  jobs as a single project, with every published port bound to `127.0.0.1`
  and the three third-party images pinned by digest.
- `.env.example` as the single configuration file, split into the per-machine
  settings a user fills in and the frozen stack settings `envcheck` compares
  against. Required Compose variables fail fast with a readable message rather
  than starting a half-configured stack.
- `SETUP.md`, including the table of the four places this stack deliberately
  differs from production and why.

#### Project files

- `README.md`, `LICENSE` (Apache-2.0), `NOTICE` with the third-party licence
  inventory, and `CITATION.cff`.
- Lint and type gates configured in `pyproject.toml`: `ruff` (E, F, I, UP, B)
  and `mypy --strict` over `evaluation/`. Both pass on this release, as does
  `ruff format --check`.

### Notes

#### Version

Every version string in the repository reads `0.1.0`: the tag `v0.1.0`, the
`version` key of `pyproject.toml` and of each vendored service, the `version`
field of `CITATION.cff`, and the `harness` field a run manifest records
through `importlib.metadata`. The stored runs under `experiments/runs/` also
record `"harness": "0.1.0"`, but they were made in the upstream development
repository, so the matching string is not evidence that the code is
identical; `experiments/pins.yaml` lists the differences that remain.

#### Reproducing the paper's numbers

`evaluation score` and `evaluation aggregate` reproduce the shipped
`results.jsonl` and `aggregates.jsonl` from the stored run files alone. They
need neither the manuals, nor a GPU, nor a database, nor a model.

#### Re-running the experiments

A re-run will not reproduce the snapshots exactly, and `pins.yaml` records
why: the vendored service sources differ from the commits that produced the
stored runs in their prompt examples, in identifiers renamed to the paper's
terms, and in the number of fields `/tbox/update` answers. The LLM servers
also run at temperature 0 but are shared and batched, so repeated runs are not
bit-identical — that variation is what experiment 1 measures.

#### Provenance recorded in the stored runs

The `git` block of each stored manifest names tags of the upstream development
repository (`v0.4.2-exp`, `v0.4.8-exp`, `v0.4.9-exp`) and commits from its
history. Neither the tags nor the commits exist in this public repository,
whose history begins at its own initial commit, so those manifest values are a
record of where a run came from rather than something a reader can check out
here. `v0.1.0` is the first tag of this repository.

#### Post-hoc additions, declared in place

Experiment 4 was added after the first three were reported, and
`Stab.<Set>.Match.*`, `Router.Typo.TieRecall` and the k=50 probe were each
added after the data they describe had been seen. `METRICS.md` and the configs
say so at the point of definition. Only experiments 1 to 3 at k=5 are
pre-registered.

#### Limits a reader should know

- `experiments/pins.yaml` pins the three third-party images by digest, and
  `envcheck` verifies that each one carries a digest. Drift between a pinned
  digest and a running container is recorded in the manifest, not refused.
  The three OntoOps services are built locally and are not digest-pinned at
  all; their provenance is the upstream commit string plus the built image ID.
- `evaluation score` appends to the results file and writes
  `coverage_misses.json` and `router_examples.json` into the run directory, so
  it is neither append-safe against double scoring nor read-only with respect
  to the run it scores. `aggregate` reports a repeated scoring as a warning.
- `evaluation/__init__.py` falls back to a `0.0.0` version when the package is
  not installed, which a bare source checkout would then stamp into a run
  manifest. Run through `uv run` so the installed metadata is used.
- The stack as pinned expects an H100-class GPU for the embedding server and
  two separately configured OpenAI-compatible LLM servers. `envcheck`
  confirms that each server lists the pinned weights, but it cannot see a
  wrong `max_model_len` or a missing reasoning parser.
- `ANTHROPIC_API_KEY` is needed only by the experiment 4 judge and is not
  checked by `envcheck`; a missing key fails at run time.
- `POSTGRES_PASSWORD` ships as `change-me`, which is safe only because every
  published port binds to `127.0.0.1`.
- The service images are not uniformly pinned. Only `ontoagent` pins both its
  base image (`python:3.12.8-slim-bookworm`) and the `uv` binary (0.7.0);
  `ontorouter` pins `uv` (0.11.22) but builds on the floating
  `python:3.12-slim` tag, and `cq_generation` pins neither, building on
  `python:3.12-slim` and copying `uv` from `:latest`. Two builds of the
  unpinned images from the same source are therefore not guaranteed to be
  identical.
- `tboxcraft` withholds rather than resolves two classes of conflict: an
  object property and a data property of the same name under a hash
  namespace, and two distinct names that slugify to one IRI. Both sides are
  dropped and reported in the response counters, so a term can be absent
  without any error.
- Several paths degrade to a 200 response with empty content rather than
  failing: a classifier chunk, the terminologizer and the domain-term verifier
  each fall back when the LLM errors. The accounting counters, not the status
  code, are what show it.
- `ontorouter`'s `/admin` router carries no authentication and is documented
  as internal only.
- FalkorDB is licensed under SSPL v1 and runs here as a local store, never
  offered as a service. `NOTICE` also discloses an AGPL Pellet jar that ships
  inside the `owlready2` wheel and is never invoked.
- No test suite and no CI workflow ship with this release. The lint and type
  gates above are run by hand.

[Unreleased]: https://github.com/shane1003/ontoops/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/shane1003/ontoops/releases/tag/v0.1.0
