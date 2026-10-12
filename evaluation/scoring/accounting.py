"""Experiment 1 and 2, metric group D: what the pipeline kept and dropped.

The production gate asks whether everything dropped was reported. Here each
identity is its own metric (1 when it holds), so a failure says where the
count leaks; ``Acct.NoSilentLoss`` is 1 only when all of them hold.

CQ side (from the stored answers):
    paragraphs = paragraphs with drafts + paragraphs whose generation failed
    drafts     = drafts refined + drafts whose refinement failed
    promoted   = terminologized + terminologization failures
Ontology side (from the service counters):
    components extracted = TBox candidates + terms withheld + names folded
    TBox candidates      = new + merged + relabelled + gray
    individuals sent     = upserted + deduped + dropped + skipped
    assertions built     = inserted + deduped + dropped

Alongside: the size of what the cell produced (``Acct.Size.*``, from the
exported tables and the ABox answer), the shape of the class hierarchy
(``Acct.Hier.*``), and the wall-clock time of every step the runner timed
(``Acct.Latency.<step>``, from ``timings.json``).
"""

from typing import Any

from evaluation.scoring.cell_files import CellFiles


def _share(part: float, whole: float) -> float | None:
    return part / whole if whole else None


def _hierarchy(cell: CellFiles) -> dict[str, float | None]:
    """How deep and how connected the exported class hierarchy is."""
    classes = {t for t, kind in cell.term_types.items() if kind == "CLASS"}
    parents: dict[str, set[str]] = {}
    for row in cell.table("tbox_hierarchy"):
        if row["child_id"] in classes and row["parent_id"] in classes:
            parents.setdefault(row["child_id"], set()).add(row["parent_id"])
    depth: dict[str, int] = {}

    def depth_of(node: str, path: frozenset[str]) -> int:
        # The service breaks cycles before export; the guard keeps a stray one
        # from recursing forever.
        if node in depth:
            return depth[node]
        above = [p for p in parents.get(node, ()) if p not in path]
        value = 1 + max((depth_of(p, path | {node}) for p in above), default=-1)
        depth[node] = value
        return value

    depths = [depth_of(c, frozenset()) for c in classes]
    return {
        "Acct.Hier.WithParent": _share(len(parents), len(classes)),
        "Acct.Hier.MaxDepth": float(max(depths)) if depths else None,
        "Acct.Hier.MultiParent": float(sum(len(p) > 1 for p in parents.values())),
    }


def accounting(cell: CellFiles) -> dict[str, float | None]:
    """Accounting metrics of one finished cell."""
    paragraphs: list[str] = cell.json("03_paragraphs.json") or []
    generate: dict[str, Any] = cell.json("04_generate.json") or {
        "drafts": {},
        "failed": [],
    }
    refine: dict[str, Any] = cell.json("05_refine.json") or {
        "drafts": 0,
        "refined": {},
        "failed": [],
    }
    promoted: list[str] = cell.json("06_promote.json") or []
    # Runs before 2026-10-06 named this stage "conceptualize" (the paper's
    # name is terminologize); they stay scorable from their own files.
    concept: dict[str, Any] = (
        cell.json("07_terminologize.json") or cell.json("07_conceptualize.json") or {}
    )
    concept = {
        ("n_terminologized" if k == "n_conceptualized" else k): v
        for k, v in concept.items()
    }
    document: dict[str, Any] = cell.json("08_cq_document.json") or {
        "competency_questions": []
    }
    extract: dict[str, Any] = cell.json("10_extract.json") or {}
    tbox: dict[str, Any] = cell.json("20_tbox_response.json") or {}
    abox: dict[str, Any] = cell.json("30_abox_response.json") or {}
    timings: dict[str, float] = {
        ("terminologize" if step == "conceptualize" else step): seconds
        for step, seconds in (cell.json("timings.json") or {}).items()
    }
    vllm: dict[str, Any] = (cell.json("vllm.json") or {}).get("delta", {})

    failed_paragraphs = {f["item"] for f in generate["failed"]}
    refined_outputs = [c for r in refine["refined"].values() for c in r["cqs"]]
    identities = {
        "Acct.Identity.Paragraphs": len(paragraphs)
        == len(generate["drafts"]) + len(failed_paragraphs),
        "Acct.Identity.Drafts": refine["drafts"]
        == len(refine["refined"]) + len(refine["failed"]),
        "Acct.Identity.Promoted": len(promoted)
        == concept.get("n_terminologized", 0) + len(concept.get("failed", [])),
        "Acct.Identity.Components": len(extract.get("components", []))
        == tbox.get("n_candidates", 0)
        + tbox.get("n_terms_withheld", 0)
        + tbox.get("n_slug_terms_withheld", 0)
        # Components whose names fold into another's before candidacy; the
        # service lists them, as the production gate's n_folded counts.
        + len(tbox.get("folded_names") or []),
        "Acct.Identity.Candidates": tbox.get("n_candidates", 0)
        == sum(tbox.get(k, 0) for k in ("n_new", "n_merged", "n_relabel", "n_gray")),
        "Acct.Identity.Individuals": abox.get("individuals", 0)
        == sum(
            abox.get(k, 0)
            for k in (
                "individuals_upserted",
                "individuals_deduped",
                "individuals_dropped",
                "individuals_skipped",
            )
        ),
        "Acct.Identity.Assertions": abox.get("assertions_built", 0)
        == sum(
            abox.get(k, 0)
            for k in ("assertions_inserted", "assertions_deduped", "assertions_dropped")
        ),
    }
    metrics: dict[str, float | None] = {
        name: float(held) for name, held in identities.items()
    }
    metrics["Acct.NoSilentLoss"] = float(all(identities.values()))

    text = {
        row["paragraph_id"]: len(row.get("text") or "")
        for row in cell.table("paragraph")
    }
    cqs = document["competency_questions"]
    n_terms = [
        len(cq.get("classes", []))
        + len(cq.get("object_properties", []))
        + len(cq.get("data_properties", []))
        for cq in cqs
    ]
    metrics.update(
        {
            "Acct.Generate.FailedParagraphs": _share(
                len(failed_paragraphs), len(paragraphs)
            ),
            "Acct.Generate.FailedText": _share(
                sum(text.get(p, 0) for p in failed_paragraphs), sum(text.values())
            ),
            "Acct.CQ.PerParagraph": _share(len(cqs), len(paragraphs)),
            "Acct.CQ.Flagged": _share(
                sum(1 for c in refined_outputs if c.get("flag")), len(refined_outputs)
            ),
            "Acct.CQ.KeptOriginal": _share(
                sum(1 for r in refine["refined"].values() if r.get("kept_original")),
                refine["drafts"],
            ),
            "Acct.CQ.NoTerms": _share(sum(1 for n in n_terms if n == 0), len(n_terms)),
            "Acct.CQ.TermsPerCQ": _share(sum(n_terms), len(n_terms)),
            "Acct.TBox.Withheld": _share(
                tbox.get("n_terms_withheld", 0), len(extract.get("components", []))
            ),
            "Acct.TBox.Unresolved": float(
                tbox.get("n_unresolved_ranges", 0) + tbox.get("n_unresolved_domains", 0)
            ),
            "Acct.ABox.Skipped": _share(
                abox.get("individuals_skipped", 0), abox.get("individuals", 0)
            ),
            "Acct.Latency.Cell": cell.status.elapsed_s,
        }
    )
    for step, seconds in timings.items():
        metrics[f"Acct.Latency.{step}"] = seconds
    kinds = list(cell.term_types.values())
    metrics.update(
        {
            "Acct.Size.Paragraphs": float(len(paragraphs)),
            "Acct.Size.BodyChars": float(sum(text.values())),
            "Acct.Size.CQs": float(len(cell.table("cq"))),
            "Acct.Size.Classes": float(kinds.count("CLASS")),
            "Acct.Size.ObjProps": float(kinds.count("OBJECT_PROP")),
            "Acct.Size.DataProps": float(kinds.count("DATA_PROP")),
            "Acct.Size.SubClassEdges": float(len(cell.table("tbox_hierarchy"))),
            "Acct.Size.Individuals": float(abox.get("individuals_upserted", 0)),
        }
    )
    metrics.update(_hierarchy(cell))
    for server, delta in vllm.items():
        if delta.get("available"):
            metrics[f"Acct.LLM.{server}.Truncated"] = delta.get(
                "vllm:request_success_total:length", 0.0
            )
    return metrics
