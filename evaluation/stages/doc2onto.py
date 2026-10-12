"""One cell: a prepared manual through the three services, then exported.

The steps and their order are production's: register,
parse, skeleton, chunk, generate, refine, promote, terminologize, extract,
TBox, ABox, graph. Every service answer is written to the cell directory as it
arrives, so a cell that fails still leaves everything up to the failure, and
the export at the end copies the cell's rows out of the stores so that scoring
reads files only.

Cell directory layout (numbered in step order)::

    00_document.json   01_odl.json   02_structure.json | 02_structure_skipped.json
    03_paragraphs.json 04_generate.json 05_refine.json 06_promote.json
    07_terminologize.json 08_cq_document.json 09_preflight.json
    10_extract.json 20_tbox_response.json 21_tbox.ttl 30_abox_response.json
    40_graph.json  timings.json  vllm.json  status.json  export/
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from evaluation.clients.common import ServiceError, map_concurrently
from evaluation.clients.cq_generation import CQGenerationClient, Refined, cq_document
from evaluation.clients.ontoagent import Namespace, OntoagentClient
from evaluation.clients.store import StoreError
from evaluation.config import DocumentSpec, RunSettings
from evaluation.contracts import CellStatus, Stage, Status
from evaluation.stages import vllm_metrics


class CellTimeoutError(RuntimeError):
    """The cell ran past its wall-clock budget."""


class Store(Protocol):
    """The direct store operations a cell needs (``evaluation.clients.store``)."""

    def register(self, pdf: bytes, name: str) -> str: ...

    def export(
        self, run_id: str, base_iri: str, graph: str
    ) -> dict[str, list[dict[str, Any]]]: ...


@dataclass
class Clients:
    """Everything a cell talks to; injected so tests can stand in for it."""

    cq: CQGenerationClient
    agent: OntoagentClient
    store: Store
    llm_roots: dict[str, str] = field(default_factory=dict)
    metrics: Callable[[str], dict[str, float] | None] = vllm_metrics.snapshot


@dataclass
class Cell:
    """What identifies one cell.

    Attributes:
        dir: The cell directory.
        document: The manual and its settings.
        run_index: Repetition index.
        namespace: Where the cell writes; its ``document`` doubles as the CQ
            generation run id, so both stores key the cell the same way.
    """

    dir: Path
    document: DocumentSpec
    run_index: int
    namespace: Namespace

    @property
    def run_id(self) -> str:
        """The CQ generation run id."""
        return self.namespace.document


def _write(cell: Cell, name: str, payload: Any) -> None:
    path = cell.dir / name
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    elif isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8",
        )


def _failures(outcomes: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "item": o.item if isinstance(o.item, str) else str(o.item),
            "status": o.error.status,
            "error": str(o.error)[-400:],
        }
        for o in outcomes
        if o.error is not None
    ]


class _Clock:
    def __init__(self, budget_s: float) -> None:
        self.start = time.monotonic()
        self.budget_s = budget_s
        self.marks: dict[str, float] = {}
        self._last = self.start

    def mark(self, step: str) -> None:
        now = time.monotonic()
        self.marks[step] = round(now - self._last, 3)
        self._last = now
        if now - self.start > self.budget_s:
            raise CellTimeoutError(f"over {self.budget_s:.0f} s after {step}")

    @property
    def elapsed(self) -> float:
        return round(time.monotonic() - self.start, 3)


def _cq_stage(
    cell: Cell,
    clients: Clients,
    run: RunSettings,
    clock: _Clock,
    queues: dict[str, Any],
) -> dict[str, Any]:
    doc = cell.document
    pdf = doc.pdf.read_bytes()
    document_id = clients.store.register(pdf, doc.slug)
    _write(
        cell,
        "00_document.json",
        {"document_id": document_id, "run_id": cell.run_id, "file": doc.pdf.name},
    )
    clock.mark("register")

    odl = clients.cq.parser_odl(pdf, doc.pdf.name)
    _write(cell, "01_odl.json", odl)
    clock.mark("parser_odl")

    structure = clients.cq.structure_odl(odl)
    if structure.content is None:
        _write(cell, "02_structure_skipped.json", {"reason": structure.skipped_reason})
    else:
        _write(cell, "02_structure.json", structure.content)
    clock.mark("structure_odl")

    paragraphs = clients.cq.chunk_odl(
        odl,
        structure.content,
        document_id=document_id,
        run_id=cell.run_id,
        budget=run.chunk_budget,
    )
    _write(cell, "03_paragraphs.json", paragraphs)
    clock.mark("chunk_odl")
    queues["after_chunk"] = _queues(clients)

    generated = map_concurrently(
        lambda pid: clients.cq.generate(
            purpose=doc.purpose, scope=doc.scope, paragraph_id=pid, run_id=cell.run_id
        ),
        paragraphs,
        run.cq_concurrency,
    )
    _write(
        cell,
        "04_generate.json",
        {
            "drafts": {o.item: o.result for o in generated if o.result is not None},
            "failed": _failures(generated),
        },
    )
    if not any(o.result for o in generated):
        raise ServiceError(
            "cq_generation", "/api/v1/generate", "no paragraph produced a draft"
        )
    clock.mark("generate")
    queues["after_generate"] = _queues(clients)

    drafts = clients.cq.drafts(cell.run_id)
    refined = map_concurrently(
        lambda d: clients.cq.refine(d["draft_id"], cell.run_id),
        drafts,
        run.cq_concurrency,
    )
    _write(
        cell,
        "05_refine.json",
        {
            "drafts": len(drafts),
            "refined": {
                o.item["draft_id"]: _refined(o.result)
                for o in refined
                if o.result is not None
            },
            "failed": _failures(refined),
        },
    )
    clock.mark("refine")

    promoted = clients.cq.promote(cell.run_id)
    _write(cell, "06_promote.json", promoted)
    clock.mark("promote")

    terminologized = map_concurrently(
        clients.cq.terminologize, promoted, run.cq_concurrency
    )
    cqs = [o.result for o in terminologized if o.result is not None]
    _write(
        cell,
        "07_terminologize.json",
        {
            "n_promoted": len(promoted),
            "n_terminologized": len(cqs),
            "failed": _failures(terminologized),
        },
    )
    if not cqs:
        raise ServiceError(
            "cq_generation", "/api/v1/terminologize", "no CQ was terminologized"
        )
    clock.mark("terminologize")
    queues["after_cq_stage"] = _queues(clients)

    document = cq_document(
        cqs,
        base_iri=cell.namespace.base_iri,
        prefix=cell.run_id,
        purpose=doc.purpose,
        scope=doc.scope,
        domain=run.domain,
    )
    _write(cell, "08_cq_document.json", document)
    return document


def _refined(result: Refined) -> dict[str, Any]:
    return {"cqs": result.cqs, "kept_original": result.kept_original}


def _onto_stage(
    cell: Cell,
    clients: Clients,
    run: RunSettings,
    clock: _Clock,
    document: dict[str, Any],
) -> None:
    _write(cell, "09_preflight.json", clients.agent.preflight())
    clock.mark("preflight")
    extraction = clients.agent.extract(document, chunk_size=run.components_chunk_size)
    _write(cell, "10_extract.json", extraction)
    clock.mark("extract")
    tbox = clients.agent.tbox_update(
        extraction["components"], cell.namespace, domain=run.domain
    )
    _write(cell, "20_tbox_response.json", {k: v for k, v in tbox.items() if k != "ttl"})
    _write(cell, "21_tbox.ttl", tbox["ttl"])
    clock.mark("tbox_update")
    _write(
        cell,
        "30_abox_response.json",
        clients.agent.abox_load(extraction["individuals"], cell.namespace),
    )
    clock.mark("abox_load")
    _write(
        cell, "40_graph.json", clients.agent.graph_upload(tbox["ttl"], cell.namespace)
    )
    clock.mark("graph_upload")


def _queues(clients: Clients) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, root in clients.llm_roots.items():
        snap = clients.metrics(root)
        out[name] = {g: snap.get(g) for g in vllm_metrics.GAUGES} if snap else None
    return out


def _status_of(exc: Exception) -> Status:
    if isinstance(exc, CellTimeoutError) or (
        isinstance(exc, ServiceError) and exc.timed_out
    ):
        return Status.TIMEOUT
    # The TBox gate is where the reasoner's verdict surfaces.
    if (
        isinstance(exc, ServiceError)
        and exc.endpoint == "/tbox/update"
        and "gate" in str(exc)
    ):
        return Status.INCONSISTENT
    return Status.PARSE_FAILURE


def run_cell(cell: Cell, clients: Clients, run: RunSettings) -> CellStatus:
    """Runs one cell and exports it; never raises for a service failure.

    Returns:
        The cell's status, also written to ``status.json``. A failed cell
        keeps every file written before the failure and gets no export.
    """
    cell.dir.mkdir(parents=True, exist_ok=True)
    clock = _Clock(run.timeout_s)
    before = {name: clients.metrics(root) for name, root in clients.llm_roots.items()}
    queues: dict[str, Any] = {"start": _queues(clients)}
    stage = Stage.CQ
    status = CellStatus(
        status=Status.OK,
        run_id=cell.run_id,
        base_iri=cell.namespace.base_iri,
        graph=cell.namespace.graph,
    )
    try:
        document = _cq_stage(cell, clients, run, clock, queues)
        stage = Stage.ONTO
        _onto_stage(cell, clients, run, clock, document)
        exported = clients.store.export(
            cell.run_id, cell.namespace.base_iri, cell.namespace.graph
        )
        export_dir = cell.dir / "export"
        export_dir.mkdir(exist_ok=True)
        for table, rows in exported.items():
            with (export_dir / f"{table}.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
                        + "\n"
                    )
        clock.mark("export")
    except (ServiceError, CellTimeoutError, StoreError) as exc:
        status = status.model_copy(
            update={
                "status": _status_of(exc),
                "stage": stage,
                "reason": str(exc)[-1000:],
            }
        )
    after = {name: clients.metrics(root) for name, root in clients.llm_roots.items()}
    queues["end"] = _queues(clients)
    _write(
        cell,
        "vllm.json",
        {
            "delta": {n: vllm_metrics.delta(before[n], after[n]) for n in before},
            "queues": queues,
        },
    )
    _write(cell, "timings.json", clock.marks)
    status = status.model_copy(update={"elapsed_s": clock.elapsed})
    _write(cell, "status.json", status.model_dump(mode="json"))
    return status
