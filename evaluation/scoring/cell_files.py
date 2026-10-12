"""Reading a scored cell: status, service answers, exported tables.

Every reader tolerates a missing file (a failed cell stops early) by
returning an empty value; the caller decides whether that is a failure.
"""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from evaluation.contracts import CellRef, CellStatus, Status
from evaluation.scoring.normalize import normalize


def read_json(path: Path) -> Any:
    """Parsed JSON, or None when the file is absent."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parsed JSON Lines, or an empty list when the file is absent."""
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


@dataclass
class CellFiles:
    """One cell directory, read lazily.

    Attributes:
        ref: The cell as ``cells.json`` lists it.
        dir: Its directory.
        run_dir: The run directory (for the copied scorer assets).
    """

    ref: CellRef
    dir: Path
    run_dir: Path

    @cached_property
    def status(self) -> CellStatus:
        """The cell's ``status.json``; a missing one means the cell never ended."""
        raw = read_json(self.dir / "status.json")
        if raw is None:
            return CellStatus(status=Status.TIMEOUT, reason="no status.json")
        return CellStatus.model_validate(raw)

    @property
    def ok(self) -> bool:
        """Whether the cell finished and was exported."""
        return self.status.status == Status.OK

    def json(self, name: str) -> Any:
        """A service answer the cell stored."""
        return read_json(self.dir / name)

    def table(self, name: str) -> list[dict[str, Any]]:
        """An exported table."""
        return read_jsonl(self.dir / "export" / f"{name}.jsonl")

    def asset(self, name: str) -> Path | None:
        """A scorer asset of the cell's manual, copied into the run."""
        path = self.run_dir / "assets" / self.ref.dataset / name
        return path if path.exists() else None

    @cached_property
    def pref_labels(self) -> dict[str, str]:
        """Term id -> normalized preferred label."""
        return {
            row["term_id"]: normalize(row["surface_text"])
            for row in self.table("surface_form")
            if row.get("form_type") == "PREF"
        }

    @cached_property
    def term_types(self) -> dict[str, str]:
        """Term id -> ``CLASS``, ``OBJECT_PROP`` or ``DATA_PROP``."""
        return {row["id"]: row.get("term_type", "") for row in self.table("tbox_term")}

    def surface_set(self, *, all_forms: bool) -> set[str]:
        """Normalized surface forms: preferred only, or all three kinds."""
        kinds = {"PREF", "ALTER", "HIDDEN"} if all_forms else {"PREF"}
        return {
            normalize(row["surface_text"])
            for row in self.table("surface_form")
            if row.get("form_type") in kinds
        }
