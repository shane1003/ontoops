"""results.jsonl and aggregates.jsonl.

Aggregation reads rows, never artifacts, so it runs anywhere the rows exist -
that is what makes every paper number recomputable from published files.
Failure accounting follows the pre-registered rule: a failed cell counts as 0
in one mean and is excluded from the other, and both are reported with the
per-cell N and failure count beside them.

A cell that ran fine but whose metric is undefined (the benchmark itself
reports null, for instance when nothing of that kind was predicted) is a
third case. It is neither a failure nor a zero, so it is counted separately
and left out of both means: scoring it 0 would invent a bad measurement, and
calling it a failure would overstate how often the pipeline broke.
"""

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import pstdev

import numpy as np
from pydantic import BaseModel

from evaluation.contracts import ResultRow, RunType, Status


class AggRow(BaseModel):
    """One aggregated cell: experiment, arm, dataset and metric.

    Attributes:
        n: Cells in the group, counting failures and undefined metrics.
        n_failed: Cells whose status was not ``ok``.
        n_undefined: Cells that succeeded but reported no value.
        mean_with_zero: Mean over failed and successful cells, failures
            contributing 0; ``None`` when every cell was undefined.
        mean_ok: Mean over successful cells only, ``None`` if there are none.
        std_ok: Population standard deviation over successful cells.
        ci_low: Lower bound of the bootstrap 95% interval over runs.
        ci_high: Upper bound; ``None`` when fewer than two runs succeeded.
    """

    experiment: str
    run_type: RunType
    dataset: str
    metric: str
    n: int
    n_failed: int
    n_undefined: int
    mean_with_zero: float | None
    mean_ok: float | None
    std_ok: float | None
    ci_low: float | None
    ci_high: float | None


def append_rows(path: Path, rows: Iterable[ResultRow]) -> int:
    """Appends rows as JSON lines; returns how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")
            count += 1
    return count


def read_rows(path: Path) -> list[ResultRow]:
    """Reads every non-empty line of a results file."""
    return [
        ResultRow.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def bootstrap_ci(
    values: list[float], n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap 95% interval for the mean over run-level values.

    Args:
        values: One value per successful run.
        n_boot: Number of resamples.
        seed: RNG seed, so the interval is reproducible from the rows alone.

    Returns:
        ``(low, high)`` at the 2.5th and 97.5th percentiles.
    """
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    means = rng.choice(array, size=(n_boot, array.size), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


CellKey = tuple[str, RunType, str, str, int]


def _cell_key(row: ResultRow) -> CellKey:
    return (row.experiment, row.run_type, row.dataset, row.metric, row.run_index)


def _latest_attempts(rows: Iterable[ResultRow]) -> list[ResultRow]:
    # A repaired cell writes a second attempt; only the final one is the
    # result. Keeping both would double-count the run.
    #
    # The same cell can also appear from a *second scoring*, because the file
    # is appended to. Then the later timestamp wins. Ordering on attempt alone
    # left the tie to file order, which kept whichever row was written first -
    # so a rerun was silently scored with the older run's numbers. Use
    # ``superseded_cells`` to see when this happened.
    latest: dict[CellKey, ResultRow] = {}
    for row in rows:
        key = _cell_key(row)
        current = latest.get(key)
        if current is None or (row.attempt, row.ts) > (current.attempt, current.ts):
            latest[key] = row
    return list(latest.values())


def superseded_cells(rows: Iterable[ResultRow]) -> dict[CellKey, list[str]]:
    """Cells that were scored twice, and the timestamps aggregation drops.

    A repaired cell writes a second row under a higher ``attempt``; that is the
    repair policy, not a rescoring, so it is not reported here. What is
    reported is the same cell *and the same attempt* appearing twice, which
    means the file holds rows from two scorings and one of them will be
    ignored.

    Args:
        rows: Result rows, as read from ``results.jsonl``.

    Returns:
        Cell key -> timestamps that will be ignored, oldest first. Empty when
        every cell was scored once, which is the normal case.
    """
    seen: dict[tuple[CellKey, int], list[str]] = defaultdict(list)
    for row in rows:
        seen[(_cell_key(row), row.attempt)].append(row.ts)
    dropped: dict[CellKey, list[str]] = {}
    for (key, _attempt), stamps in seen.items():
        if len(stamps) < 2:
            continue
        dropped.setdefault(key, []).extend(sorted(stamps)[:-1])
    return {key: sorted(stamps) for key, stamps in dropped.items()}


def aggregate(
    rows: Iterable[ResultRow], n_boot: int = 1000, seed: int = 0
) -> list[AggRow]:
    """Aggregates rows per cell, counting failures both ways.

    Args:
        rows: Result rows; only the highest ``attempt`` per run is used.
        n_boot: Bootstrap resamples for the interval over successful runs.
        seed: RNG seed so aggregates are reproducible.

    Returns:
        One ``AggRow`` per (experiment, run_type, dataset, metric), sorted.
    """
    cells: dict[tuple[str, RunType, str, str], list[ResultRow]] = defaultdict(list)
    for row in _latest_attempts(rows):
        cells[(row.experiment, row.run_type, row.dataset, row.metric)].append(row)

    out: list[AggRow] = []
    for (experiment, run_type, dataset, metric), cell in sorted(cells.items()):
        ok_values = [
            row.value
            for row in cell
            if row.status is Status.OK and row.value is not None
        ]
        n_failed = sum(1 for row in cell if row.status is not Status.OK)
        n_undefined = len(cell) - n_failed - len(ok_values)
        scored = n_failed + len(ok_values)
        interval: tuple[float | None, float | None] = (None, None)
        if len(ok_values) >= 2:
            interval = bootstrap_ci(ok_values, n_boot=n_boot, seed=seed)
        out.append(
            AggRow(
                experiment=experiment,
                run_type=run_type,
                dataset=dataset,
                metric=metric,
                n=len(cell),
                n_failed=n_failed,
                n_undefined=n_undefined,
                mean_with_zero=sum(ok_values) / scored if scored else None,
                mean_ok=sum(ok_values) / len(ok_values) if ok_values else None,
                std_ok=pstdev(ok_values) if len(ok_values) >= 2 else None,
                ci_low=interval[0],
                ci_high=interval[1],
            )
        )
    return out


def write_aggregates(path: Path, aggs: Iterable[AggRow]) -> int:
    """Overwrites the aggregates file; returns the number of rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [agg.model_dump_json() for agg in aggs]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)
