"""Experiment 1 and 2, metric group B: coverage of the manual's own assets.

``Cov.Term`` and ``Cov.Abbr`` over the glossary, ``Cov.Index`` over the
index headings, each twice: against preferred labels only and against all
surface forms (preferred, alternative, hidden). ``Cov.Para.CQ`` and
``Cov.Para.Onto`` over the body paragraphs. A manual without a glossary or
an index has no row for it, never a zero.
"""

from evaluation.scoring.assets import index_headings, load_glossary, load_index
from evaluation.scoring.cell_files import CellFiles
from evaluation.scoring.normalize import normalize, variants


def _recall(hits: int, total: int) -> float | None:
    return hits / total if total else None


def coverage(cell: CellFiles) -> tuple[dict[str, float | None], dict[str, list[str]]]:
    """Coverage metrics of one finished cell.

    Returns:
        The metrics, and the missed entries per metric (for the record).
    """
    metrics: dict[str, float | None] = {}
    misses: dict[str, list[str]] = {}
    forms = {
        False: cell.surface_set(all_forms=False),
        True: cell.surface_set(all_forms=True),
    }
    suffix = {False: "Pref", True: "AllForms"}

    glossary_path = cell.asset("glossary.csv")
    if glossary_path:
        entries = load_glossary(glossary_path)
        terms = [e for e in entries if e.entry_type == "term"]
        abbrs = [e for e in entries if e.entry_type == "abbreviation"]
        for all_forms in (False, True):
            have = forms[all_forms]
            hit_terms = [e for e in terms if normalize(e.term) in have]
            hit_abbrs = [
                e
                for e in abbrs
                if normalize(e.term) in have or normalize(e.definition) in have
            ]
            metrics[f"Cov.Term.Recall.{suffix[all_forms]}"] = _recall(
                len(hit_terms), len(terms)
            )
            metrics[f"Cov.Abbr.Recall.{suffix[all_forms]}"] = _recall(
                len(hit_abbrs), len(abbrs)
            )
            if all_forms:
                misses["Cov.Term"] = sorted(e.term for e in terms if e not in hit_terms)
                misses["Cov.Abbr"] = sorted(e.term for e in abbrs if e not in hit_abbrs)

    index_path = cell.asset("index.csv")
    if index_path:
        headings = index_headings(load_index(index_path))
        for all_forms in (False, True):
            hit = [h for h in headings if variants(h) & forms[all_forms]]
            metrics[f"Cov.Index.Recall.{suffix[all_forms]}"] = _recall(
                len(hit), len(headings)
            )
            if all_forms:
                misses["Cov.Index"] = sorted(set(headings) - set(hit))

    paragraphs = {row["paragraph_id"] for row in cell.table("paragraph")}
    cqs = cell.table("cq")
    with_cq = {row["paragraph_id"] for row in cqs}
    linked_cqs = {row["cq_id"] for row in cell.table("term_cq_map")}
    with_terms = {row["paragraph_id"] for row in cqs if row["cq_id"] in linked_cqs}
    metrics["Cov.Para.CQ"] = _recall(len(paragraphs & with_cq), len(paragraphs))
    metrics["Cov.Para.Onto"] = _recall(len(paragraphs & with_terms), len(paragraphs))
    return metrics, misses
