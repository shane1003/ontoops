# OntoOps

Evaluation code and results of the OntoOps paper.

OntoOps turns a document into an OWL ontology through competency questions
(CQs): it generates and refines CQs per paragraph, extracts annotated
components from them, and builds a validated TBox (plus individuals). A lookup
service then routes a term to every per-document ontology that defines it.

This repository holds the three OntoOps services (`services/`), the code that
runs and scores them (`evaluation/`), and the scored results of the paper's
runs (`experiments/results/`).

## Setup

```sh
uv sync
```

This is enough to recompute the paper's numbers. Running the experiments again
needs the full stack (Docker, a GPU and two LLM servers); see
[SETUP.md](SETUP.md).

## Reproduce the paper's numbers

Every number in the paper is an aggregation of the scored rows in
`experiments/results/`. Recomputing them needs no GPU, database or model:

```sh
uv run evaluation aggregate          # -> experiments/results/aggregates.jsonl
uv run evaluation aggregate --results experiments/results/results_probe_k50.jsonl \
    --out experiments/results/aggregates_probe_k50.jsonl
```

The run directories themselves (every service response and exported table)
are not part of this repository. A run you make yourself (see below) is
scored from its files alone:

```sh
uv run evaluation score experiments/runs/<run_id>
```

## Experiments

The corpus is five public US Army and Joint doctrine manuals: ATP 2-01.3
(IPB), ADP 3-90, ADRP 3-0, JP 2-01.3 and JP 3-0. Each manual's own glossary
and index are the gold standard. Metrics are defined in
[METRICS.md](METRICS.md).

| Experiment | Question | Config |
|---|---|---|
| 1 Stability | Does IPB, run three times at temperature 0, give the same ontology, and does it cover the manual? | `doctrine.yaml` |
| 2 Generality | Does the pipeline cover the other four manuals as well? | `doctrine.yaml` |
| 3 Lookup | Do the per-manual ontologies stay apart, and does the lookup service route a shared term to every manual that has it? | `doctrine_probe.yaml` (k=5), `doctrine_probe_k50.yaml` (k=50) |
| 4 Faithfulness (added after the first three were reported) | Is each CQ's answer supported by the paragraph it came from? | `faithfulness_ipb.yaml` |

With the stack running (see [SETUP.md](SETUP.md)):

```sh
uv run evaluation run --allow-untagged experiments/configs/doctrine.yaml          # experiments 1-3
uv run evaluation run --allow-untagged experiments/configs/faithfulness_ipb.yaml  # experiment 4
```

`--allow-untagged` lets the run start from a checkout without a release tag
(this repository has none); the manifest still records the exact commit and
any changed files.

Experiments 3 and 4 read a finished `doctrine.yaml` run. The three configs
(`doctrine_probe.yaml`, `doctrine_probe_k50.yaml`, `faithfulness_ipb.yaml`)
name the paper's run in `from_run`; replace it with the id of your own run.
Experiment 4 judges that run with an external model and needs
`ANTHROPIC_API_KEY` in `.env`.

## Repository layout

| Path | What it is |
|---|---|
| `services/` | The method: CQ generation, ontoagent (conceptualizer, TBox, ABox) and the lookup service |
| `evaluation/` | Runs the services on the stack, exports what each run stored, and scores the exported files (`scoring/` reads files only) |
| `experiments/` | Configs, `pins.yaml` (the service sources and image digests every run records) and the scored results |
| `corpus/` | `SOURCES.yaml` (file, edition and SHA-256 of each manual) and each manual's glossary and index |
| `docker-compose.yml`, `.env.example` | The stack and its single configuration file |

## Citation

The paper is under review. Until it is published, please cite this repository
with the metadata in [CITATION.cff](CITATION.cff).

## License

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
