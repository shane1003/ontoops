"""Component merge helpers: order-preserving dedup, field union, verdict reconciliation.

Used by the runner's exact (kind,name) reintegration. Only names identical down to
the character are merged, so no name is discarded: semantic merging ("different
names, same concept") is not done by this module; it belongs to the downstream that
owns the global TBox.
"""

from __future__ import annotations

from app.conceptualizer.schemas import (
    AnnotatedComponent,
    DisagreementFlag,
    VerdictCandidate,
)


def merge_dedup(items: list[str]) -> list[str]:
    """Removes duplicates (and empty strings) while keeping order."""
    return [it for it in dict.fromkeys(items) if it]


def split_slot(value: str | None) -> list[str]:
    """Turns a domain/range slot string back into a list of names (inverse of extract._join).

    When a property has different domain/range values across CQs they are joined with
    ', ', so split them again.
    """
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _union_slot(a: str | None, b: str | None) -> str | None:
    """Unions two comma-joined domain/range slots (same format as extract._join).

    domain/range is not single-valued but **multi-valued**: extract._join already joins
    the values of several CQs within a chunk with ", ". Picking `a or b` in cross-chunk
    reintegration keeps only the first chunk's slot and loses the domain/range seen by
    the other chunks (general-purpose properties always cross chunk boundaries, so this
    really leaks). Hence split and union, like the other multi-valued fields.
    """
    merged = merge_dedup(split_slot(a) + split_slot(b))
    return ", ".join(merged) if merged else None


def union_component_fields(base: AnnotatedComponent, other: AnnotatedComponent) -> None:
    """Merges other's structure/domain layer/provenance into base (in place).

    Single-valued fields (comment, correct_term) take the first non-empty value (refine
    rule); multi-valued fields (domain, range, synonyms, duplicates, hierarchy,
    source_cq_ids) are unioned. The verdict is not touched: the caller settles it
    separately with reconcile_verdict.
    """
    base.comment = base.comment or other.comment
    base.domain = _union_slot(base.domain, other.domain)
    base.range = _union_slot(base.range, other.range)
    # If even one chunk saw a CQ declaration, the component is declared (synthesized
    # means 'nobody declared it').
    base.synthesized = base.synthesized and other.synthesized
    # correct_term and its language are a pair: taking first-non-empty of each could
    # attach other's `en` to base's Korean term, making a claim nobody made. Take the
    # language together with the term (notes/notes_lang are not merged; base's pair
    # stays as it is).
    if not base.domain_layer.correct_term and other.domain_layer.correct_term:
        base.domain_layer.correct_term = other.domain_layer.correct_term
        base.domain_layer.correct_term_lang = other.domain_layer.correct_term_lang
    base.domain_layer.synonyms = merge_dedup(
        base.domain_layer.synonyms + other.domain_layer.synonyms
    )
    base.domain_layer.duplicates = merge_dedup(
        base.domain_layer.duplicates + other.domain_layer.duplicates
    )
    for view, names in other.domain_layer.hierarchy.items():
        cur = base.domain_layer.hierarchy.get(view, [])
        base.domain_layer.hierarchy[view] = merge_dedup(cur + names)
    base.disagreement_flags.extend(other.disagreement_flags)
    base.source_cq_ids = merge_dedup(base.source_cq_ids + other.source_cq_ids)


def flag_hierarchy_contradictions(
    base: AnnotatedComponent,
    view_sources: dict[str, dict[str, list[str]]],
) -> list[str]:
    """Flags a contradiction when one name is observed as both broader and narrower (in place).

    Edges are not removed. Reintegration only unions the observations; which direction
    is right is the judgment of the downstream that sees the global TBox. Removing one
    side here would also remove the evidence for that judgment (the same reason the
    verdict is not settled as tbox). Instead the contradiction is exposed with its
    evidence: without the flag downstream cannot even know a contradiction exists.

    Args:
        base: Component to flag.
        view_sources: {other name: {view: [source, ...]}}, the per-chunk hierarchy
            observations. When a name is observed in both directions, one flag is
            added per name, recording as candidates which chunk saw which direction.

    Returns:
        The names that were flagged.
    """
    flagged: list[str] = []
    for name, views in view_sources.items():
        broader = views.get("broader") or []
        narrower = views.get("narrower") or []
        if not (broader and narrower):
            continue
        base.disagreement_flags.append(
            DisagreementFlag(
                kind="hierarchy_conflict",
                component=base.name,
                subject=name,
                candidates=[
                    VerdictCandidate(value=view, source=src)
                    for view, sources in (("broader", broader), ("narrower", narrower))
                    for src in sources
                ],
                note=(
                    f"'{name}' 이(가) '{base.name}' 의 broader 와 narrower 로 동시에 "
                    "관측됐다 — 간선 방향이 모순이다."
                ),
            )
        )
        flagged.append(name)
    return flagged


def reconcile_verdict(
    base: AnnotatedComponent,
    verdict_sources: dict[str, list[str]],
    *,
    note: str,
) -> None:
    """Settles base.verdict from verdict_sources ({verdict: [source,...]}) (in place).

    - If two or more decisive values (tbox/abox) disagree: conflict + a
      verdict_conflict flag.
    - Otherwise, if any observation is conflict, it sticks as conflict (no
      downgrade; otherwise the verdict would disagree with the remaining conflict
      flag).
    - A single decisive value wins; with nothing at all, conflict.

    A source is a chunk id (reintegration path).
    """
    non_conflict = [v for v in verdict_sources if v != "conflict"]
    if len(set(non_conflict)) > 1:
        base.verdict = "conflict"
        candidates = [
            VerdictCandidate(value=v, source=src)
            for v in sorted(non_conflict)
            for src in verdict_sources[v]
        ]
        base.disagreement_flags.append(
            DisagreementFlag(
                kind="verdict_conflict",
                component=base.name,
                candidates=candidates,
                note=note,
            )
        )
    elif "conflict" in verdict_sources:
        base.verdict = "conflict"
    elif non_conflict:
        base.verdict = non_conflict[0]
    else:
        base.verdict = "conflict"
