"""Runs a config: freeze check, environment check, manifest, cells, probe.

A run directory holds everything scoring needs::

    experiments/runs/<run_id>/
        manifest.json   config.yaml   cells.json
        exp1_stability/<slug>/pipeline/<i>/...      (see stages/doc2onto.py)
        exp2_generality/<slug>/pipeline/<i>/...
        exp3_router/corpus/pipeline/0/lookups.jsonl probe.json
        exp4_faithfulness/<slug>/pipeline/<i>/verdicts.jsonl judge.json

The manifest records what ran (commit, tag, pins, the images actually
running, the models the LLM servers serve) and never an address.
"""

import json
import os
import platform
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation import __version__
from evaluation.clients.cq_generation import CQGenerationClient
from evaluation.clients.judge import Judge, RagasJudge
from evaluation.clients.ontoagent import Namespace, OntoagentClient
from evaluation.clients.ontorouter import OntorouterClient
from evaluation.clients.store import StackStore, stack_dsn
from evaluation.config import DocumentSpec, ExperimentConfig, config_sha256, load_config
from evaluation.contracts import CellRef, RunType
from evaluation.envcheck import (
    LLM_ENDPOINTS,
    STACK_SETTINGS,
    format_report,
    host_view,
    load_pins,
    read_env_file,
    run_checks,
)
from evaluation.stages.doc2onto import Cell, Clients, run_cell
from evaluation.stages.faithfulness import run_faithfulness
from evaluation.stages.router_probe import ProbedCell, run_probe

RUNS = Path("experiments/runs")
PINS = Path("experiments/pins.yaml")
ENV_FILE = Path(".env")
ENV_TEMPLATE = Path(".env.example")
JUDGE_KEY = "ANTHROPIC_API_KEY"
# Outputs of running and scoring; their change does not make a tree dirty.
_OWN_OUTPUTS = ("experiments/runs/", "experiments/results/")


class RunRefusedError(RuntimeError):
    """The run may not start: untagged or dirty tree, or a failed check."""


@dataclass(frozen=True)
class GitState:
    """What the repository was when the run started."""

    commit: str
    tag: str | None
    dirty: list[str]


def _git(*args: str) -> str:
    completed = subprocess.run(["git", *args], capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else ""


def git_state() -> GitState:
    """The HEAD commit, its exact tag if any, and changed paths."""
    changed = [
        line[3:]
        for line in _git("status", "--porcelain", "--untracked-files=no").splitlines()
        if not line[3:].startswith(_OWN_OUTPUTS)
    ]
    tag = _git("describe", "--tags", "--exact-match", "HEAD") or None
    return GitState(_git("rev-parse", "HEAD"), tag, changed)


def _images() -> list[dict[str, str]]:
    completed = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            str(ENV_FILE),
            "images",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return []
    return [
        {k: str(item.get(k, "")) for k in ("ContainerName", "Repository", "Tag", "ID")}
        for item in json.loads(completed.stdout or "[]")
    ]


def _models(env: dict[str, str]) -> dict[str, Any]:
    import urllib.request

    out: dict[str, Any] = {}
    for name, base_key, _ in LLM_ENDPOINTS:
        url = host_view(env.get(base_key, "")).rstrip("/") + "/models"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
                data = json.load(response).get("data", [])
        except Exception:
            data = None
        out[name] = (
            [{k: m.get(k) for k in ("id", "root", "max_model_len")} for m in data]
            if data is not None
            else None
        )
    return out


def llm_roots(env: dict[str, str]) -> dict[str, str]:
    """Host-side server roots (no ``/v1``) for the metrics snapshots."""
    cq = (
        host_view(env.get("CQGEN_LLM_BASE_URL", ""))
        .removesuffix("/")
        .removesuffix("/v1")
    )
    agent = host_view(env.get("ONTOAGENT_LLM_ROOT_URL", ""))
    return {
        name: root
        for name, root in (("cq_generation", cq), ("ontoagent", agent))
        if root
    }


def default_clients(cfg: ExperimentConfig, env: dict[str, str]) -> Clients:
    """Clients of the running stack, addressed as ``.env`` describes it."""
    run = cfg.run
    return Clients(
        cq=CQGenerationClient(run.cq_generation_url),
        agent=OntoagentClient(run.ontoagent_url, timeout_s=float(run.timeout_s)),
        store=StackStore(
            stack_dsn(env), ("127.0.0.1", int(env.get("FALKORDB_PORT", "56379")))
        ),
        llm_roots=llm_roots(env),
    )


def plan_cells(
    cfg: ExperimentConfig, run_dir: Path, stamp: str
) -> list[tuple[str, DocumentSpec, int, Cell]]:
    """Every document cell of the config, in run order."""
    blocks: list[tuple[str, list[DocumentSpec]]] = []
    if cfg.exp1_stability:
        blocks.append(("exp1_stability", [cfg.exp1_stability.document]))
    if cfg.exp2_generality:
        blocks.append(("exp2_generality", cfg.exp2_generality.documents))
    cells = []
    for experiment, documents in blocks:
        for doc in documents:
            for index in range(doc.n_runs):
                namespace = Namespace(
                    cfg.run.iri_root,
                    cfg.run.iri_domain,
                    f"{doc.iri_name}_{stamp}_r{index}",
                )
                cell_dir = (
                    run_dir
                    / experiment
                    / doc.slug
                    / RunType.PIPELINE.value
                    / str(index)
                )
                cells.append(
                    (experiment, doc, index, Cell(cell_dir, doc, index, namespace))
                )
    return cells


def _write_cells(run_dir: Path, refs: list[CellRef]) -> None:
    payload = [ref.model_dump(mode="json") for ref in refs]
    (run_dir / "cells.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")


def run_experiment(
    config_path: Path,
    *,
    runs_root: Path = RUNS,
    allow_untagged: bool = False,
    env: dict[str, str] | None = None,
    make_clients: Callable[
        [ExperimentConfig, dict[str, str]], Clients
    ] = default_clients,
    make_router: Callable[[ExperimentConfig], OntorouterClient] | None = None,
    make_judge: Callable[[ExperimentConfig], Judge] | None = None,
    check_environment: bool = True,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    state: Callable[[], GitState] = git_state,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Runs every cell of a config and returns the run directory.

    Progress goes to ``events.jsonl`` in the run directory.

    Raises:
        RunRefusedError: If the tree is not frozen (unless allowed) or the
            run-profile environment check fails.
    """
    cfg = load_config(config_path)
    env = env if env is not None else {**read_env_file(ENV_FILE), **os.environ}
    repo = state()
    if not allow_untagged and (repo.tag is None or repo.dirty):
        raise RunRefusedError(
            f"HEAD must be tagged and clean (tag={repo.tag}, "
            f"changed={repo.dirty[:5]}); pass --allow-untagged for a trial run"
        )
    if cfg.exp4_faithfulness and not make_judge and not env.get(JUDGE_KEY):
        raise RunRefusedError(f"{JUDGE_KEY} is not set (.env); the judge needs it")
    if check_environment:
        checks = run_checks(
            "run",
            pins=load_pins(PINS),
            env=env,
            stack_defaults=read_env_file(ENV_TEMPLATE),
            weights=cfg.run.model_weights,
        )
        if not all(check.ok for check in checks):
            raise RunRefusedError("environment check failed:\n" + format_report(checks))

    started = now()
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{config_path.stem}"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)
    shutil.copy(config_path, run_dir / "config.yaml")
    manifest = {
        "run_id": run_id,
        "started": started.isoformat(),
        "config": config_path.name,
        "config_sha256": config_sha256(config_path),
        "git": {"commit": repo.commit, "tag": repo.tag, "dirty": repo.dirty},
        "pins": load_pins(PINS).model_dump(mode="json"),
        "images": _images() if check_environment else [],
        "llm_models": _models(env) if check_environment else {},
        "run_settings": cfg.run.model_dump(mode="json"),
        "stack_settings": {key: env.get(key) for key in STACK_SETTINGS},
        "harness": __version__,
        "python": platform.python_version(),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )

    # The scorer assets travel with the run, so a run copied elsewhere is
    # scored from its own directory.
    for doc in cfg.documents():
        for asset in (doc.glossary, doc.index):
            if asset is not None:
                target = run_dir / "assets" / doc.slug / asset.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(asset, target)

    planned = plan_cells(cfg, run_dir, stamp)
    events = _Events(run_dir, now)
    events.emit(
        "run_started",
        f"{len(planned)} cells{' + router probe' if cfg.exp3_router else ''}"
        f"{' + faithfulness' if cfg.exp4_faithfulness else ''}, "
        f"commit {repo.commit[:10]}, tag {repo.tag}, dirty {len(repo.dirty)}",
    )
    try:
        _run_cells_and_probe(
            cfg,
            run_dir,
            runs_root,
            planned,
            make_clients(cfg, env),
            make_router,
            make_judge,
            env,
            sleep,
            events,
        )
    except Exception as exc:
        events.emit("run_aborted", f"{type(exc).__name__}: {str(exc)[-300:]}")
        raise
    events.emit("run_finished", f"{(now() - started).total_seconds() / 60:.1f} min")
    return run_dir


class _Events:
    """Timestamped progress, written to the run directory."""

    def __init__(self, run_dir: Path, now: Callable[[], datetime]) -> None:
        self.path = run_dir / "events.jsonl"
        self.now = now

    def emit(self, event: str, detail: str) -> None:
        record = {"ts": self.now().isoformat(), "event": event, "detail": detail}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def cell_summary(cell_dir: Path) -> str:
    """One line on what a finished cell produced, from its own files."""
    parts = []
    paragraphs = _load(cell_dir / "03_paragraphs.json")
    if paragraphs is not None:
        parts.append(f"paragraphs {len(paragraphs)}")
    generate = _load(cell_dir / "04_generate.json")
    if generate:
        parts.append(f"generate failed {len(generate['failed'])}")
    concept = _load(cell_dir / "07_terminologize.json")
    if concept:
        parts.append(f"CQs {concept['n_terminologized']}")
    tbox = _load(cell_dir / "20_tbox_response.json")
    if tbox:
        parts.append(f"TBox terms {tbox.get('n_new', 0) + tbox.get('n_merged', 0)}")
    abox = _load(cell_dir / "30_abox_response.json")
    if abox:
        parts.append(f"individuals {abox.get('individuals_upserted', 0)}")
    return ", ".join(parts)


def _run_cells_and_probe(
    cfg: ExperimentConfig,
    run_dir: Path,
    runs_root: Path,
    planned: list[tuple[str, DocumentSpec, int, Cell]],
    clients: Clients,
    make_router: Callable[[ExperimentConfig], OntorouterClient] | None,
    make_judge: Callable[[ExperimentConfig], Judge] | None,
    env: dict[str, str],
    sleep: Callable[[float], None],
    events: "_Events",
) -> None:
    refs: list[CellRef] = []
    for number, (experiment, doc, index, cell) in enumerate(planned, start=1):
        events.emit(
            "cell_started", f"{number}/{len(planned)} {experiment} {doc.slug} r{index}"
        )
        status = run_cell(cell, clients, cfg.run)
        events.emit(
            "cell_finished",
            f"{number}/{len(planned)} {doc.slug} r{index}: {status.status.value}"
            + (f" at {status.stage.value}" if status.stage else "")
            + f", {status.elapsed_s / 60:.1f} min; {cell_summary(cell.dir)}"
            + (f"; {status.reason[-200:]}" if status.reason else ""),
        )
        refs.append(
            CellRef(
                experiment=experiment,
                run_type=RunType.PIPELINE,
                dataset=doc.slug,
                run_index=index,
                kind="doc2onto",
                dir=str(cell.dir.relative_to(run_dir)),
            )
        )
        _write_cells(run_dir, refs)

    if cfg.exp3_router:
        probe = cfg.exp3_router
        source_dir = runs_root / probe.from_run if probe.from_run else run_dir
        source_cfg = load_config(source_dir / "config.yaml") if probe.from_run else cfg
        glossaries = {d.slug: d.glossary for d in source_cfg.documents()}
        if probe.from_run:
            # The stored config's relative paths resolve against the run
            # directory, not the configs folder; the run's own asset copies
            # are the files its cells were scored with.
            glossaries = {
                slug: source_dir / "assets" / slug / path.name if path else None
                for slug, path in glossaries.items()
            }
        probed = [
            ProbedCell(
                sel.dataset,
                source_dir
                / sel.experiment
                / sel.dataset
                / RunType.PIPELINE.value
                / str(sel.run_index),
                glossaries.get(sel.dataset),
            )
            for sel in probe.cells
        ]
        out = run_dir / "exp3_router" / "corpus" / RunType.PIPELINE.value / "0"
        router = (
            make_router(cfg)
            if make_router
            else OntorouterClient(cfg.run.ontorouter_url)
        )
        # The lexical index reloads on an interval; probing before it has
        # seen the new terms would score a stale index, not the service.
        sleep(probe.index_wait_s)
        result = run_probe(
            out,
            probed,
            router,
            ood_terms=probe.ood_terms,
            seed=probe.typo_seed,
            k=probe.k,
        )
        refs.append(
            CellRef(
                experiment="exp3_router",
                run_type=RunType.PIPELINE,
                dataset="corpus",
                run_index=0,
                kind="router",
                dir=str(out.relative_to(run_dir)),
            )
        )
        _write_cells(run_dir, refs)
        events.emit("probe_finished", f"queries {result['queries']}")

    if cfg.exp4_faithfulness:
        block = cfg.exp4_faithfulness
        for sel in block.cells:
            # One judge per cell: its HTTP pool belongs to the cell's event loop.
            judge = (
                make_judge(cfg)
                if make_judge
                else RagasJudge(block.judge_model, api_key=env[JUDGE_KEY])
            )
            tail = Path(sel.dataset) / RunType.PIPELINE.value / str(sel.run_index)
            out = run_dir / "exp4_faithfulness" / tail
            summary = run_faithfulness(
                out,
                runs_root / block.from_run / sel.experiment / tail,
                judge,
                dataset=sel.dataset,
                concurrency=block.concurrency,
            )
            refs.append(
                CellRef(
                    experiment="exp4_faithfulness",
                    run_type=RunType.PIPELINE,
                    dataset=sel.dataset,
                    run_index=sel.run_index,
                    kind="faithfulness",
                    dir=str(out.relative_to(run_dir)),
                )
            )
            _write_cells(run_dir, refs)
            events.emit(
                "faithfulness_finished",
                f"{sel.dataset} r{sel.run_index}: {summary['n_judged']}/"
                f"{summary['n_cq']} CQs judged, {summary['n_errors']} errors, "
                f"{summary['elapsed_s'] / 60:.1f} min",
            )
