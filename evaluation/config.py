"""Experiment configuration: one run setting plus experiment blocks.

The three experiments of the paper (stability, generality, router) share one
document specification, so a document is described once and the experiment
blocks only say which documents and how many repetitions.
"""

import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class RunSettings(BaseModel):
    """Where the services are and the run-wide knobs that must match production.

    The LLM servers are not configured here: their addresses are machine
    specific and live in ``.env``, which the stack and ``envcheck`` both read.

    Attributes:
        model_weights: Weights both LLM servers must serve (the ``root`` of
            their ``/v1/models`` entry).
        cq_generation_url: The CQ generation service.
        ontoagent_url: The component extraction, TBox and ABox service.
        ontorouter_url: The lookup service.
        timeout_s: Wall-clock budget for one cell.
        components_chunk_size: CQs per extraction call. Production lowered the
            service default of 10 to 4 after truncation, so 4 is the value.
        chunk_budget: Target characters per paragraph chunk (a target, not a
            cap, exactly as the service treats it).
        iri_root: Root of every generated IRI.
        iri_domain: Domain segment under the root.
        domain: Domain the TBox verifier and the component classifier judge
            terms against; production uses ``military``.
        cq_concurrency: Parallel calls per CQ generation step; production's
            flow runs 72.
    """

    model_weights: str
    cq_generation_url: str = "http://localhost:8081"
    ontoagent_url: str = "http://localhost:8082"
    ontorouter_url: str = "http://localhost:8083"
    timeout_s: int = 10800
    components_chunk_size: int = 4
    chunk_budget: int = 1500
    iri_root: str = "http://example.org"
    iri_domain: str = "doctrine"
    domain: str = "military"
    cq_concurrency: int = 72


class DocumentSpec(BaseModel):
    """One document the pipeline runs on and the assets it is scored against.

    Attributes:
        slug: Dataset name of the cells and the asset directory.
        name: The manual's title without its publication number, spaces as
            underscores; it names the IRI namespace of every cell
            (``<root>/<domain>/<name>_<run stamp>_r<i>#``). Defaults to
            the slug.
        pdf: The prepared PDF (body pages only); never the full manual.
        purpose: Ontology purpose passed to CQ generation.
        scope: Ontology scope passed to CQ generation.
        glossary: The manual's glossary as CSV, or None when it has none.
        index: The manual's index as CSV, or None when it has none.
        n_runs: Independent repetitions.

    There is no temperature here: the services are configured at 0 in the
    stack and a cell cannot change that.
    """

    slug: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_]*$")
    pdf: Path
    purpose: str
    scope: str
    glossary: Path | None = None
    index: Path | None = None
    n_runs: int = 3

    @property
    def iri_name(self) -> str:
        """The namespace name of the manual."""
        return self.name or self.slug


class Exp1StabilityConfig(BaseModel):
    """Experiment 1: one document, repeated, scored for stability and coverage."""

    document: DocumentSpec


class Exp2GeneralityConfig(BaseModel):
    """Experiment 2: the same procedure over several documents."""

    documents: list[DocumentSpec] = Field(min_length=1)

    @field_validator("documents")
    @classmethod
    def _unique_slugs(cls, value: list[DocumentSpec]) -> list[DocumentSpec]:
        slugs = [doc.slug for doc in value]
        duplicates = sorted({slug for slug in slugs if slugs.count(slug) > 1})
        if duplicates:
            raise ValueError(f"duplicate document slug(s): {duplicates}")
        return value


class CellSelector(BaseModel):
    """Names one finished cell whose ontology the router experiment probes."""

    experiment: str
    dataset: str
    run_index: int = 0


class Exp3RouterConfig(BaseModel):
    """Experiment 3: invariants of the lookup service over finished ontologies.

    Attributes:
        cells: The ontologies to probe, one per document.
        ood_terms: Frozen list of out-of-domain query terms, one per line.
        typo_seed: Seed of the deterministic typo injection.
        k: Suggestions requested per channel.
        from_run: Run id under ``experiments/runs`` holding the cells; None
            means the cells of the same run, run before the probe.
        index_wait_s: Seconds to wait before probing, so the service's
            lexical index has reloaded the cells' terms (the stack refreshes
            it every 30 s).
    """

    cells: list[CellSelector] = Field(min_length=1)
    ood_terms: Path
    typo_seed: int = 42
    k: int = 5
    from_run: str | None = None
    index_wait_s: float = 60.0


class Exp4FaithfulnessConfig(BaseModel):
    """Experiment 4: RAGAS Faithfulness of the CQ answers of finished cells.

    Attributes:
        from_run: Run id under ``experiments/runs`` holding the cells.
        cells: The cells whose CQs are judged.
        judge_model: Model id of the judge (Anthropic API).
        concurrency: CQs judged at once.
    """

    from_run: str
    cells: list[CellSelector] = Field(min_length=1)
    judge_model: str
    concurrency: int = Field(default=8, ge=1)


class ExperimentConfig(BaseModel):
    """Top-level config: one run setting plus optional experiment blocks."""

    run: RunSettings
    exp1_stability: Exp1StabilityConfig | None = None
    exp2_generality: Exp2GeneralityConfig | None = None
    exp3_router: Exp3RouterConfig | None = None
    exp4_faithfulness: Exp4FaithfulnessConfig | None = None

    @model_validator(mode="after")
    def _at_least_one_block(self) -> "ExperimentConfig":
        if not (
            self.exp1_stability
            or self.exp2_generality
            or self.exp3_router
            or self.exp4_faithfulness
        ):
            raise ValueError("at least one experiment block is required")
        return self

    def documents(self) -> list[DocumentSpec]:
        """Returns every document the config runs the pipeline on."""
        docs: list[DocumentSpec] = []
        if self.exp1_stability:
            docs.append(self.exp1_stability.document)
        if self.exp2_generality:
            docs.extend(self.exp2_generality.documents)
        return docs


def _resolve(value: Path | None, base: Path) -> Path | None:
    # A relative path in a config means "relative to the config file", not to
    # the process cwd, so the same config is valid from anywhere.
    if value is None or value.is_absolute():
        return value
    return (base / value).resolve()


def _resolve_paths(cfg: ExperimentConfig, base: Path) -> None:
    for doc in cfg.documents():
        doc.pdf = _resolve(doc.pdf, base) or doc.pdf
        doc.glossary = _resolve(doc.glossary, base)
        doc.index = _resolve(doc.index, base)
    if cfg.exp3_router:
        cfg.exp3_router.ood_terms = (
            _resolve(cfg.exp3_router.ood_terms, base) or cfg.exp3_router.ood_terms
        )


def load_config(path: Path) -> ExperimentConfig:
    """Loads and validates an experiment config file.

    Args:
        path: YAML file with a ``run`` block and at least one experiment block.

    Returns:
        The validated config with file paths made absolute.

    Raises:
        pydantic.ValidationError: If the file violates the schema.
    """
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = ExperimentConfig.model_validate(raw)
    _resolve_paths(cfg, path.resolve().parent)
    return cfg


def config_sha256(path: Path) -> str:
    """Returns the SHA-256 hex digest of the config file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
