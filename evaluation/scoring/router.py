"""Experiment 3: the lookup service's invariants.

The service reads the whole stack store, where other repetitions share the
same labels, so every answer is judged only against the namespaces of the
probed cells (``probe.json``).

I1 ``Router.NS.Purity``     per manual: graph classes and terms under its namespace.
I2 ``Router.Overlap.Recall`` shared labels answered with every owner's IRI.
I3 ``Router.Glossary.Recall`` per manual: glossary entries answered with an IRI
   of the manual; ``Router.Glossary.Agreement`` how often that agrees with
   the coverage matcher (all surface forms).
I4 ``Router.Typo.Recall``   per manual: a one-letter typo of a term the service
   finds exactly gets that term's IRI as the first suggestion among the
   probed namespaces. ``Router.Typo.TieRecall`` (added after the doctrine
   probe was seen): the term's IRI is among the suggestions tied for first,
   since manuals sharing a label tie and the service orders a tie by graph
   name, not by relevance.
I5 ``Router.OOD.Leak``      out-of-domain queries answered inside the probed namespaces.
I6 ``Router.Prov.Chain``    per manual: terms that reach a paragraph through a CQ.
"""

import json
import random
from pathlib import Path
from typing import Any

from evaluation.scoring.cell_files import CellFiles, read_json, read_jsonl
from evaluation.scoring.normalize import normalize

EXAMPLES = 10


def _in(iri: str, namespaces: list[str]) -> bool:
    return any(iri.startswith(ns) for ns in namespaces)


def _exact(answer: dict[str, Any], namespace: str) -> list[str]:
    return [
        r["iri"] for r in answer.get("results", []) if r["iri"].startswith(namespace)
    ]


def router_metrics(
    probe_dir: Path, cells: dict[str, CellFiles]
) -> tuple[dict[str, dict[str, float | None]], dict[str, Any]]:
    """Router metrics per manual (``corpus`` for the corpus-wide ones).

    Args:
        probe_dir: The router cell directory.
        cells: The probed cells by manual slug.

    Returns:
        ``{dataset: {metric: value}}`` and the example tables.
    """
    probe = read_json(probe_dir / "probe.json")
    lookups = read_jsonl(probe_dir / "lookups.jsonl")
    ns = {slug: info["base_iri"] for slug, info in probe["cells"].items()}
    all_ns = list(ns.values())
    out: dict[str, dict[str, float | None]] = {slug: {} for slug in ns}
    out["corpus"] = {}

    for slug, cell in cells.items():
        base = ns[slug]
        iris = [r["iri"] for r in cell.table("graph_class")] + list(cell.term_types)
        out[slug]["Router.NS.Purity"] = (
            sum(i.startswith(base) for i in iris) / len(iris) if iris else None
        )
        paragraphs = {r["paragraph_id"] for r in cell.table("paragraph")}
        cq_paragraph = {r["cq_id"]: r["paragraph_id"] for r in cell.table("cq")}
        chained = {
            r["term_id"]
            for r in cell.table("term_cq_map")
            if cq_paragraph.get(r["cq_id"]) in paragraphs
        }
        terms = list(cell.term_types)
        out[slug]["Router.Prov.Chain"] = (
            sum(t in chained for t in terms) / len(terms) if terms else None
        )

    overlap = [q for q in lookups if q["family"] == "overlap"]
    if overlap:
        hits = sum(
            all(_exact(q["answer"], ns[d]) for d in q["datasets"]) for q in overlap
        )
        out["corpus"]["Router.Overlap.Recall"] = hits / len(overlap)
    ood = [q for q in lookups if q["family"] == "ood"]
    if ood:
        leaks = sum(
            any(
                _in(r["iri"], all_ns)
                for r in q["answer"].get("results", [])
                + q["answer"]["suggestions"].get("lexical", [])
            )
            for q in ood
        )
        out["corpus"]["Router.OOD.Leak"] = leaks / len(ood)

    exact_by_term: dict[tuple[str, str], list[str]] = {}
    for slug in ns:
        glossary = [
            q for q in lookups if q["family"] == "glossary" and q["dataset"] == slug
        ]
        if not glossary:
            continue
        forms = cells[slug].surface_set(all_forms=True) if slug in cells else set()
        found = 0
        agree = 0
        for q in glossary:
            iris = _exact(q["answer"], ns[slug])
            exact_by_term[(slug, q["query"])] = iris
            found += bool(iris)
            agree += bool(iris) == (normalize(q["query"]) in forms)
        out[slug]["Router.Glossary.Recall"] = found / len(glossary)
        out[slug]["Router.Glossary.Agreement"] = agree / len(glossary)

        typos = [q for q in lookups if q["family"] == "typo" and q["dataset"] == slug]
        judged = [q for q in typos if exact_by_term.get((slug, q["original"]))]
        if judged:
            right = tied = 0
            for q in judged:
                ranked = [
                    s
                    for s in q["answer"]["suggestions"].get("lexical", [])
                    if _in(s["iri"], all_ns)
                ]
                target = exact_by_term[(slug, q["original"])]
                right += bool(ranked) and ranked[0]["iri"] in target
                tied += bool(ranked) and any(
                    s["iri"] in target
                    for s in ranked
                    if s["score"] == ranked[0]["score"]
                )
            out[slug]["Router.Typo.Recall"] = right / len(judged)
            out[slug]["Router.Typo.TieRecall"] = tied / len(judged)
    return out, _examples(overlap, ns, cells)


def _examples(
    overlap: list[dict[str, Any]], ns: dict[str, str], cells: dict[str, CellFiles]
) -> dict[str, Any]:
    """Shared labels with each manual's IRI and the paragraph behind it."""
    rng = random.Random(0)
    chosen = rng.sample(overlap, min(EXAMPLES, len(overlap)))
    rows = []
    for q in chosen:
        owners = []
        for slug in q["datasets"]:
            iris = _exact(q["answer"], ns[slug])
            cell = cells.get(slug)
            excerpt = ""
            if iris and cell:
                cq_ids = [
                    r["cq_id"]
                    for r in cell.table("term_cq_map")
                    if r["term_id"] == iris[0]
                ]
                cq_paragraph = {r["cq_id"]: r["paragraph_id"] for r in cell.table("cq")}
                text = {
                    r["paragraph_id"]: r.get("text", "")
                    for r in cell.table("paragraph")
                }
                if cq_ids and cq_ids[0] in cq_paragraph:
                    excerpt = " ".join(text.get(cq_paragraph[cq_ids[0]], "").split())[
                        :200
                    ]
            owners.append(
                {
                    "dataset": slug,
                    "iri": iris[0] if iris else None,
                    "paragraph": excerpt,
                }
            )
        rows.append({"label": q["query"], "owners": owners})
    return {"overlap_examples": rows}


def write_examples(probe_dir: Path, examples: dict[str, Any]) -> None:
    """Writes the example tables next to the probe (paper tables 3-1, 3-2)."""
    (probe_dir / "router_examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=1), encoding="utf-8"
    )
