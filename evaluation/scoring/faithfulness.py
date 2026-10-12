"""Experiment 4: RAGAS Faithfulness of CQ answers to their paragraphs.

Reads the verdicts the judge stage stored (``verdicts.jsonl``). A CQ's score
is the share of its answer's statements the judge found supported by the
paragraph. CQs without statements (an answer the judge could not split, or a
judge error) have no score; ``Faith.Judged`` says how many were scored, and
every share below is over the scored CQs.

``Faith.CQ.Mean.Ctx100Plus`` and ``.CtxShort`` split the mean by the length
of the paragraph's text at 100 characters: shorter paragraphs are titles and
table or figure captions, which hold almost nothing to support an answer.
"""

from pathlib import Path

from evaluation.scoring.cell_files import read_jsonl

SHORT_CONTEXT_CHARS = 100


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def faithfulness_metrics(cell_dir: Path) -> dict[str, float | None]:
    """Faithfulness metrics of one judged cell.

    Args:
        cell_dir: The faithfulness cell directory.

    Returns:
        ``{metric: value}``; a metric without a defined value is None.
    """
    records = read_jsonl(cell_dir / "verdicts.jsonl")
    scored = [
        (
            sum(s["verdict"] for s in r["statements"]) / len(r["statements"]),
            r["context_chars"],
        )
        for r in records
        if r["statements"]
    ]
    scores = [score for score, _ in scored]
    verdicts = [s["verdict"] for r in records for s in r["statements"]]
    short = [score for score, chars in scored if chars < SHORT_CONTEXT_CHARS]
    return {
        "Faith.CQ.Mean": _mean(scores),
        "Faith.CQ.Full": _mean([float(score == 1.0) for score in scores]),
        "Faith.CQ.Low": _mean([float(score < 0.5) for score in scores]),
        "Faith.CQ.Mean.Ctx100Plus": _mean(
            [score for score, chars in scored if chars >= SHORT_CONTEXT_CHARS]
        ),
        "Faith.CQ.Mean.CtxShort": _mean(short),
        "Faith.Statement.Mean": _mean([float(v) for v in verdicts]),
        "Faith.Judged": len(scored) / len(records) if records else None,
        "Faith.Size.CQs": float(len(records)),
        "Faith.Size.Statements": float(len(verdicts)),
        "Faith.Size.CtxShortCQs": float(len(short)),
        "Faith.Size.Errors": float(sum(1 for r in records if r.get("error"))),
    }
