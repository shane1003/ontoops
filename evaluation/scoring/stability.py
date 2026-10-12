"""Experiment 1 and 2, metric group A: agreement between repetitions.

For every pair of repetitions of one manual, the Jaccard index of the two
runs' sets. Elements are normalized preferred labels, so two runs that name a
concept the same way agree even though their IRIs sit in different
namespaces.

``Stab.<Set>.Match.Pref`` and ``Stab.<Set>.Match.AllForms`` (added after the
doctrine run was seen) ask instead how many of each run's terms, or term pairs,
have a counterpart in the other run: two terms correspond when they share a
surface form, preferred only or preferred, alternative and hidden. The share
is taken over both runs, so a concept one run prefers under a name the other
keeps only as an alternative still agrees under ``AllForms``.
"""

from collections import defaultdict
from itertools import combinations

from evaluation.scoring.cell_files import CellFiles
from evaluation.scoring.normalize import normalize

KINDS = {"Class": "CLASS", "ObjProp": "OBJECT_PROP", "DataProp": "DATA_PROP"}


def _jaccard(a: set[object], b: set[object]) -> float | None:
    union = a | b
    return len(a & b) / len(union) if union else None


def element_sets(cell: CellFiles) -> dict[str, set[object]]:
    """The sets a run is compared by."""
    label = cell.pref_labels
    types = cell.term_types
    sets: dict[str, set[object]] = {
        name: {label[t] for t, kind in types.items() if kind == wanted and t in label}
        for name, wanted in KINDS.items()
    }
    sets["SubClass"] = {
        (label[row["child_id"]], label[row["parent_id"]])
        for row in cell.table("tbox_hierarchy")
        if row["child_id"] in label and row["parent_id"] in label
    }
    for relation in ("DOMAIN", "RANGE"):
        sets[relation.title()] = {
            (label[row["prop_id"]], label[row["target_class_id"]])
            for row in cell.table("tbox_domain_range")
            if row.get("relation_type") == relation
            and row.get("prop_id") in label
            and row.get("target_class_id") in label
        }
    sets["CQ"] = {normalize(row["question"]) for row in cell.table("cq")}
    return sets


def _forms(cell: CellFiles, *, all_forms: bool) -> dict[str, set[str]]:
    kinds = {"PREF", "ALTER", "HIDDEN"} if all_forms else {"PREF"}
    forms: dict[str, set[str]] = defaultdict(set)
    for row in cell.table("surface_form"):
        if row.get("form_type") in kinds:
            forms[row["term_id"]].add(normalize(row["surface_text"]))
    return forms


def _term_elements(cell: CellFiles) -> dict[str, set[tuple[str, ...]]]:
    """The term-id tuples behind each label and pair set of ``element_sets``."""
    types = cell.term_types
    elements: dict[str, set[tuple[str, ...]]] = {
        name: {(t,) for t, kind in types.items() if kind == wanted}
        for name, wanted in KINDS.items()
    }
    elements["SubClass"] = {
        (row["child_id"], row["parent_id"]) for row in cell.table("tbox_hierarchy")
    }
    for relation in ("DOMAIN", "RANGE"):
        elements[relation.title()] = {
            (row["prop_id"], row["target_class_id"])
            for row in cell.table("tbox_domain_range")
            if row.get("relation_type") == relation
            and row.get("prop_id")
            and row.get("target_class_id")
        }
    return elements


def _matched(
    mine: set[tuple[str, ...]],
    my_forms: dict[str, set[str]],
    theirs: set[tuple[str, ...]],
    their_forms: dict[str, set[str]],
) -> int:
    """How many of ``mine`` have a counterpart among ``theirs``."""
    by_form: dict[str, set[str]] = defaultdict(set)
    for term, forms in their_forms.items():
        for form in forms:
            by_form[form].add(term)
    by_head: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for element in theirs:
        by_head[element[0]].add(element)

    def counterparts(term: str) -> set[str]:
        return set().union(*(by_form[f] for f in my_forms.get(term, set())))

    count = 0
    for element in mine:
        heads = counterparts(element[0])
        if len(element) == 1:
            count += any((head,) in theirs for head in heads)
            continue
        tails = counterparts(element[1])
        count += any(
            other[1] in tails for head in heads for other in by_head.get(head, ())
        )
    return count


def _match_shares(a: CellFiles, b: CellFiles) -> dict[str, float | None]:
    elements_a, elements_b = _term_elements(a), _term_elements(b)
    shares: dict[str, float | None] = {}
    for variant, all_forms in (("Pref", False), ("AllForms", True)):
        forms_a = _forms(a, all_forms=all_forms)
        forms_b = _forms(b, all_forms=all_forms)
        for name in elements_a:
            mine, theirs = elements_a[name], elements_b[name]
            total = len(mine) + len(theirs)
            shares[f"Stab.{name}.Match.{variant}"] = (
                (
                    _matched(mine, forms_a, theirs, forms_b)
                    + _matched(theirs, forms_b, mine, forms_a)
                )
                / total
                if total
                else None
            )
    return shares


def stability(
    cells: list[CellFiles],
) -> list[tuple[tuple[int, int], dict[str, float | None]]]:
    """Pairwise Jaccard per set, for every pair of finished repetitions.

    Args:
        cells: The repetitions of one manual in one experiment.

    Returns:
        ``((i, j), {metric: value})`` per pair of run indexes, with
        ``Stab.<Set>.Jaccard`` and ``Stab.<Set>.Match.{Pref,AllForms}``;
        a pair with a failed repetition is left out (the failure is reported
        by that cell's own rows).
    """
    finished = sorted((c for c in cells if c.ok), key=lambda c: c.ref.run_index)
    by_index = {c.ref.run_index: c for c in finished}
    sets = {index: element_sets(c) for index, c in by_index.items()}
    pairs = []
    for a, b in combinations(sorted(sets), 2):
        values: dict[str, float | None] = {
            f"Stab.{name}.Jaccard": _jaccard(sets[a][name], sets[b][name])
            for name in sets[a]
        }
        values.update(_match_shares(by_index[a], by_index[b]))
        pairs.append(((a, b), values))
    return pairs
