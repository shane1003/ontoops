"""Shared types of the evaluation package.

This module is the single abstraction every other module agrees on. It must
not import anything else from ``evaluation`` so that the scoring side and the
execution side can depend on it without depending on each other.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class Status(StrEnum):
    """Outcome of one execution cell after validation and repair."""

    OK = "ok"
    PARSE_FAILURE = "parse_failure"
    INCONSISTENT = "inconsistent"
    TIMEOUT = "timeout"


class Stage(StrEnum):
    """Pipeline stage in which a failure occurred."""

    CQ = "cq_stage"
    ONTO = "onto_stage"


class RunType(StrEnum):
    """Which arm of the experiment a row belongs to."""

    PIPELINE = "pipeline"
    BASELINE = "baseline"
    REFERENCE = "reference"


@dataclass(frozen=True)
class Verdict:
    """Result of validating one OWL file.

    Attributes:
        status: Harness-owned status. A reasoner crash (for example an
            unsupported datatype) is reported as ``INCONSISTENT`` with the
            crash text in ``reason``, because the results schema has no
            separate state for it.
        reason: Human-readable detail, empty when ``status`` is ``OK``.
    """

    status: Status
    reason: str = ""


@dataclass
class StageResult:
    """What a stage hands back to the runner.

    Attributes:
        status: Outcome after the runner applied validation and repair.
        stage: The pipeline stage that produced (or failed to produce) the
            outputs, or None for work that is not a pipeline stage, such as
            the question-answering evaluation.
        outputs: Named artifact paths written under the stage directory.
        repair_attempted: Whether the deterministic repair ran.
        repair_succeeded: Whether the repair turned a failure into ``OK``.
        generation_attempts: How many completions a single-prompt stage
            needed before one parsed; 1 for everything else.
        elapsed_s: Wall-clock seconds spent in the stage.
        stderr_tail: Last lines of stderr or traceback, for the run log only.
        counters: Numeric counters the services reported (``n_*`` fields of
            the extraction, TBox and ABox responses), the input of the
            accounting metrics.
    """

    status: Status
    stage: Stage | None
    outputs: dict[str, Path] = field(default_factory=dict)
    repair_attempted: bool = False
    repair_succeeded: bool = False
    generation_attempts: int = 1
    elapsed_s: float = 0.0
    stderr_tail: str = ""
    counters: dict[str, int | float] = field(default_factory=dict)


class CellStatus(BaseModel):
    """``status.json`` of one cell: how it ended and where.

    Attributes:
        status: Outcome of the cell.
        stage: The pipeline stage that failed, None when the cell succeeded.
        reason: Failure detail; empty on success.
        elapsed_s: Wall-clock seconds for the whole cell.
        run_id: The CQ generation run id of the cell.
        base_iri: The namespace the cell wrote under.
        graph: The FalkorDB graph the cell loaded.
    """

    status: Status
    stage: Stage | None = None
    reason: str = ""
    elapsed_s: float = 0.0
    run_id: str = ""
    base_iri: str = ""
    graph: str = ""


class ResultRow(BaseModel):
    """One line of ``results.jsonl``. The field order is the schema."""

    experiment: str
    run_type: RunType
    run_index: int
    attempt: int
    dataset: str
    metric: str
    value: float | None
    status: Status
    stage: Stage | None
    repair_attempted: bool
    repair_succeeded: bool
    ts: str

    @staticmethod
    def now_ts() -> str:
        """Returns the current UTC time as an ISO 8601 string with offset."""
        return datetime.now(UTC).isoformat()


class CellRef(BaseModel):
    """Locates one cell's artifacts inside a run directory.

    Attributes:
        experiment: Config block name, e.g. ``exp2_generality``.
        run_type: Experiment arm.
        dataset: Dataset or fixture name.
        run_index: Repetition index (0-based).
        kind: How the cell is scored: ``doc2onto`` cells (one document run
            through the services, exported) yield coverage, accounting and,
            across repetitions, stability; ``router`` cells hold lookup
            responses probed against finished ontologies; ``faithfulness``
            cells hold a judge's verdicts on a finished cell's CQ answers.
        dir: Cell directory, relative to the run directory.
    """

    experiment: str
    run_type: RunType
    dataset: str
    run_index: int
    kind: Literal["doc2onto", "router", "faithfulness"]
    dir: str
