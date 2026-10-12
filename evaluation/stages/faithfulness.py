"""Experiment 4: is a CQ's answer supported by the paragraph it came from?

For every CQ of a finished cell the judge splits the answer into statements
and judges each against the paragraph's text (RAGAS Faithfulness). Nothing is
re-generated: the inputs are the cell's exported ``cq`` and ``paragraph``
tables.

Output: ``verdicts.jsonl`` (one line per CQ: its paragraph, the length of
the paragraph's text, the statements with verdict and reason, or the error
that kept it from being judged) and ``judge.json`` (the judge, the source
cell and the counts).
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from evaluation.clients.judge import Judge
from evaluation.contracts import CellStatus


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


async def _judge_cq(
    judge: Judge, cq: dict[str, Any], context: str | None
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "cq_id": cq["cq_id"],
        "paragraph_id": cq.get("paragraph_id"),
        "context_chars": len(context or ""),
        "statements": [],
        "error": None,
    }
    if not context or not cq.get("answer") or not cq.get("question"):
        record["error"] = "missing question, answer or paragraph text"
        return record
    try:
        statements = await judge.statements(cq["question"], cq["answer"])
        if statements:
            record["statements"] = await judge.verdicts(statements, context)
    except Exception as exc:  # noqa: BLE001 - one CQ's failure is recorded, not fatal
        record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return record


async def _judge_all(
    judge: Judge,
    cqs: list[dict[str, Any]],
    text: dict[str, str],
    path: Path,
    concurrency: int,
) -> list[dict[str, Any]]:
    gate = asyncio.Semaphore(concurrency)
    records: list[dict[str, Any]] = []
    # Written as each CQ finishes, so a killed run keeps what it paid for.
    with path.open("w", encoding="utf-8") as handle:

        async def one(cq: dict[str, Any]) -> None:
            async with gate:
                record = await _judge_cq(
                    judge, cq, text.get(cq.get("paragraph_id", ""))
                )
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

        await asyncio.gather(*(one(cq) for cq in cqs))
    return records


def run_faithfulness(
    out: Path, cell_dir: Path, judge: Judge, *, dataset: str, concurrency: int
) -> dict[str, Any]:
    """Judges every CQ of a finished cell and writes the verdicts.

    Args:
        out: The faithfulness cell directory to write.
        cell_dir: The finished cell (holding ``status.json`` and ``export/``).
        judge: The judge.
        dataset: The manual's slug.
        concurrency: CQs judged at once.

    Returns:
        The content of ``judge.json``.

    Raises:
        ValueError: If the source cell did not finish successfully.
        RuntimeError: If no CQ could be judged (for example a rejected key).
    """
    status = CellStatus.model_validate_json(
        (cell_dir / "status.json").read_text(encoding="utf-8")
    )
    if status.status != "ok":
        raise ValueError(f"{dataset}: cell {cell_dir} ended {status.status}")
    out.mkdir(parents=True, exist_ok=True)
    cqs = _rows(cell_dir / "export" / "cq.jsonl")
    text = {
        row["paragraph_id"]: row.get("text") or ""
        for row in _rows(cell_dir / "export" / "paragraph.jsonl")
    }
    started = time.monotonic()
    records = asyncio.run(
        _judge_all(judge, cqs, text, out / "verdicts.jsonl", concurrency)
    )
    errors = [r["error"] for r in records if r["error"]]
    summary = {
        "judge": judge.describe(),
        "dataset": dataset,
        # Relative, so the two run directories can move together.
        "source": os.path.relpath(cell_dir, out),
        "n_cq": len(records),
        "n_judged": sum(1 for r in records if r["statements"]),
        "n_errors": len(errors),
        "elapsed_s": round(time.monotonic() - started, 1),
    }
    (out / "judge.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    if records and len(errors) == len(records):
        raise RuntimeError(
            f"{dataset}: no CQ could be judged; first error: {errors[0]}"
        )
    return summary
