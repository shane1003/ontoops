"""Experiment 3: probe the lookup service over finished cells.

The service reads the whole stack store, which also holds the other
repetitions and any earlier namespaces, so the probe records the namespaces
of the cells under test and the scorer judges answers against those only.

Query families (the invariants they serve are defined in evaluation/scoring/router.py):

* ``overlap``: a preferred label present in two or more of the cells (I2).
* ``glossary``: every glossary term and abbreviation of each cell's manual (I3).
* ``typo``: a glossary term with one letter substituted, seeded (I4).
* ``ood``: the frozen out-of-domain list (I5).

Output: ``lookups.jsonl`` (one line per query with its family, the manual it
came from, the original term for a typo, and the service's answer) and
``probe.json`` (the cells, their namespaces and graphs, and the counts).
"""

import json
import os
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.clients.ontorouter import OntorouterClient
from evaluation.contracts import CellStatus
from evaluation.scoring.assets import load_glossary


@dataclass(frozen=True)
class ProbedCell:
    """A finished cell whose ontology is probed.

    Attributes:
        dataset: The manual's slug.
        dir: The cell directory (holding ``status.json`` and ``export/``).
        glossary: The manual's glossary, or None.
    """

    dataset: str
    dir: Path
    glossary: Path | None


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _key(text: str) -> str:
    return " ".join(text.lower().split())


def typo(term: str, rng: random.Random) -> str | None:
    """One letter of the term replaced by another; None for short terms."""
    positions = [i for i, ch in enumerate(term) if ch.isalpha()]
    if len(positions) < 4:
        return None
    at = rng.choice(positions)
    choices = [c for c in string.ascii_lowercase if c != term[at].lower()]
    return term[:at] + rng.choice(choices) + term[at + 1 :]


def build_queries(
    cells: list[ProbedCell], ood: list[str], seed: int
) -> list[dict[str, Any]]:
    """Every query of the probe, in a stable order."""
    queries: list[dict[str, Any]] = []
    owners: dict[str, set[str]] = {}
    surface: dict[str, str] = {}
    for cell in cells:
        for row in _rows(cell.dir / "export" / "surface_form.jsonl"):
            if row.get("form_type") == "PREF":
                key = _key(row["surface_text"])
                owners.setdefault(key, set()).add(cell.dataset)
                surface.setdefault(key, row["surface_text"])
    for key in sorted(k for k, who in owners.items() if len(who) > 1):
        queries.append(
            {
                "family": "overlap",
                "query": surface[key],
                "datasets": sorted(owners[key]),
            }
        )

    rng = random.Random(seed)
    for cell in cells:
        if cell.glossary is None:
            continue
        for entry in load_glossary(cell.glossary):
            term = entry.term.lstrip("*")
            queries.append(
                {
                    "family": "glossary",
                    "query": term,
                    "dataset": cell.dataset,
                    "entry_type": entry.entry_type,
                }
            )
            if entry.entry_type == "term":
                wrong = typo(term, rng)
                if wrong:
                    queries.append(
                        {
                            "family": "typo",
                            "query": wrong,
                            "original": term,
                            "dataset": cell.dataset,
                        }
                    )
    queries.extend({"family": "ood", "query": term} for term in ood)
    return queries


def run_probe(
    out: Path,
    cells: list[ProbedCell],
    router: OntorouterClient,
    *,
    ood_terms: Path,
    seed: int,
    k: int,
) -> dict[str, Any]:
    """Probes the service and writes the answers next to their queries.

    Returns:
        The content of ``probe.json``.

    Raises:
        ValueError: If a probed cell did not finish successfully.
    """
    out.mkdir(parents=True, exist_ok=True)
    namespaces: dict[str, dict[str, str]] = {}
    for cell in cells:
        status = CellStatus.model_validate_json(
            (cell.dir / "status.json").read_text(encoding="utf-8")
        )
        if status.status != "ok":
            raise ValueError(f"{cell.dataset}: cell {cell.dir} ended {status.status}")
        namespaces[cell.dataset] = {
            "base_iri": status.base_iri,
            "graph": status.graph,
            # Relative, so the run directory can move (the published archive).
            "dir": os.path.relpath(cell.dir, out),
        }
    ood = [
        line.strip()
        for line in ood_terms.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    queries = build_queries(cells, ood, seed)
    answers = router.lookup([q["query"] for q in queries], k=k, tiers="lexical")
    with (out / "lookups.jsonl").open("w", encoding="utf-8") as handle:
        for query, answer in zip(queries, answers, strict=True):
            handle.write(
                json.dumps({**query, "answer": answer}, ensure_ascii=False) + "\n"
            )
    families: dict[str, int] = {}
    for query in queries:
        families[query["family"]] = families.get(query["family"], 0) + 1
    probe = {"cells": namespaces, "seed": seed, "k": k, "queries": families}
    (out / "probe.json").write_text(
        json.dumps(probe, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return probe
