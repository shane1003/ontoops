"""Scores every cell of a stored run from its files alone.

The runner writes ``cells.json`` (a list of ``CellRef``) at the run root.
Document cells yield coverage and accounting rows, and every pair of
repetitions of one manual yields stability rows; the router cell yields the
invariants per manual. Nothing here touches the network, a database or
Docker: a run is scored from its exported files only.

A failed cell still yields rows, with no value and its status, so that
aggregation counts it as a failure instead of leaving it out.
"""

import json
from collections import defaultdict
from pathlib import Path

from evaluation.contracts import CellRef, ResultRow, RunType, Stage, Status
from evaluation.scoring.accounting import accounting
from evaluation.scoring.cell_files import CellFiles, read_json
from evaluation.scoring.coverage import coverage
from evaluation.scoring.faithfulness import faithfulness_metrics
from evaluation.scoring.router import router_metrics, write_examples
from evaluation.scoring.stability import stability

CELLS_FILE = "cells.json"
# Rows every document cell reports, so a failed cell is counted in each.
FAILURE_METRICS = ("Cov.Para.CQ", "Cov.Para.Onto", "Acct.NoSilentLoss")


def read_cells(run_dir: Path) -> list[CellRef]:
    """Reads the cell list the runner left at the run root."""
    raw = json.loads((run_dir / CELLS_FILE).read_text(encoding="utf-8"))
    return [CellRef.model_validate(item) for item in raw]


def _row(
    ref: CellRef,
    metric: str,
    value: float | None,
    *,
    status: Status = Status.OK,
    stage: Stage | None = None,
    dataset: str | None = None,
    run_index: int | None = None,
) -> ResultRow:
    return ResultRow(
        experiment=ref.experiment,
        run_type=ref.run_type,
        run_index=ref.run_index if run_index is None else run_index,
        attempt=1,
        dataset=dataset or ref.dataset,
        metric=metric,
        value=value,
        status=status,
        stage=stage,
        repair_attempted=False,
        repair_succeeded=False,
        ts=ResultRow.now_ts(),
    )


def _document_rows(cell: CellFiles) -> list[ResultRow]:
    if not cell.ok:
        return [
            _row(
                cell.ref,
                metric,
                None,
                status=cell.status.status,
                stage=cell.status.stage,
            )
            for metric in FAILURE_METRICS
        ]
    metrics, misses = coverage(cell)
    metrics.update(accounting(cell))
    (cell.dir / "coverage_misses.json").write_text(
        json.dumps(misses, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return [_row(cell.ref, m, v) for m, v in sorted(metrics.items()) if v is not None]


def _stability_rows(cells: list[CellFiles]) -> list[ResultRow]:
    rows = []
    for k, (_pair, metrics) in enumerate(stability(cells)):
        for metric, value in sorted(metrics.items()):
            if value is not None:
                rows.append(_row(cells[0].ref, metric, value, run_index=k))
    return rows


def _router_rows(cell: CellFiles) -> list[ResultRow]:
    probe = read_json(cell.dir / "probe.json")
    probed: dict[str, CellFiles] = {}
    for slug, info in probe["cells"].items():
        path = (cell.dir / info["dir"]).resolve()
        ref = CellRef(
            experiment="probed",
            run_type=RunType.PIPELINE,
            dataset=slug,
            run_index=0,
            kind="doc2onto",
            dir=str(path),
        )
        probed[slug] = CellFiles(ref, path, cell.run_dir)
    metrics, examples = router_metrics(cell.dir, probed)
    write_examples(cell.dir, examples)
    return [
        _row(cell.ref, metric, value, dataset=dataset)
        for dataset, values in sorted(metrics.items())
        for metric, value in sorted(values.items())
        if value is not None
    ]


def score_run(run_dir: Path, exclude: frozenset[str] = frozenset()) -> list[ResultRow]:
    """Scores every cell listed in the run directory.

    Args:
        run_dir: ``experiments/runs/<run_id>``.
        exclude: Manual slugs whose cells are left unscored.

    Returns:
        Result rows, cell by cell, then stability per manual, then router,
        then faithfulness.
    """
    cells = [
        CellFiles(ref, run_dir / ref.dir, run_dir)
        for ref in read_cells(run_dir)
        if ref.dataset not in exclude
    ]
    rows: list[ResultRow] = []
    groups: dict[tuple[str, str], list[CellFiles]] = defaultdict(list)
    for cell in cells:
        if cell.ref.kind == "doc2onto":
            rows.extend(_document_rows(cell))
            groups[(cell.ref.experiment, cell.ref.dataset)].append(cell)
    for group in groups.values():
        rows.extend(_stability_rows(group))
    for cell in cells:
        if cell.ref.kind == "router":
            rows.extend(_router_rows(cell))
        if cell.ref.kind == "faithfulness":
            rows.extend(
                _row(cell.ref, metric, value)
                for metric, value in sorted(faithfulness_metrics(cell.dir).items())
                if value is not None
            )
    return rows
