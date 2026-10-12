"""Request/response DTOs for the load endpoints."""

from __future__ import annotations

from pydantic import BaseModel

from ..services.loader import LoadReport
from ..services.mapper import Skip


class SkipOut(BaseModel):
    scope: str
    kind: str
    token: str | None
    reason: str
    candidates: list[str]
    class_name: str | None
    source_id: str | None


class LoadResponse(BaseModel):
    base_iri: str | None  # namespace actually used (?base_iri= beats the env default)
    # Echo of the source (request | assembled | env) and the assembly pieces, so a caller
    # that sent pieces can keep checking "were my domain/document used as sent" (same
    # shape as tboxcraft's domain_source: a value that never arrives silently falls back).
    base_iri_source: str | None = None
    base_iri_pieces: dict[str, str] | None = None
    individuals: int  # individuals in the payload
    # individuals = upserted + deduped + dropped + skipped (the input count is accounted for)
    individuals_upserted: int  # distinct individuals now in the ABox
    individuals_deduped: int  # merged into another input's subject (same natural key)
    individuals_dropped: int  # subject built but never came back from the write (loss)
    individuals_skipped: int  # no subject at all — each carries an `individual` scope skip
    assertions_built: int
    assertions_inserted: int  # newly inserted; the rest already existed
    assertions_deduped: int  # mapped but already present (ON CONFLICT — benign)
    assertions_dropped: int  # mapped but their subject never upserted (real loss)
    object_assertions_received: int  # instance relations in the payload
    object_assertions_skipped: int  # of those, not stored (object props are out of scope)
    miss: int
    ambiguous: int
    # Totals per reason over ALL skips — complete even when `skips` below is sampled.
    skip_counts: dict[str, int]
    # Diagnostic list. Every miss/ambiguous is here; interchangeable reasons (`empty`) are
    # sampled to MAX_SAMPLED_SKIPS each. Count from `skip_counts`, not from this.
    skips: list[SkipOut]


# Same idea for skips, but only for `empty` — those items are interchangeable ("this
# individual had nothing to store"), so a thousand read the same and only the total matters.
# A `miss` or `ambiguous` names a token someone has to go fix, so every one is kept.
# `skip_counts` carries the true totals either way, so a gate never reads a sampled list as
# if it were complete.
MAX_SAMPLED_SKIPS = 50


def sampled(skips: list[Skip]) -> list[Skip]:
    """Keep every actionable skip and the first MAX_SAMPLED_SKIPS `empty` ones, in order."""
    empties = 0
    kept = []
    for skip in skips:
        if skip.reason == "empty":
            empties += 1
            if empties > MAX_SAMPLED_SKIPS:
                continue
        kept.append(skip)
    return kept


def to_response(
    report: LoadReport,
    *,
    base_iri_source: str | None = None,
    base_iri_pieces: dict[str, str] | None = None,
) -> LoadResponse:
    return LoadResponse(
        base_iri=report.base_iri,
        base_iri_source=base_iri_source,
        base_iri_pieces=base_iri_pieces,
        individuals=report.individuals,
        individuals_upserted=report.individuals_upserted,
        individuals_deduped=report.individuals_deduped,
        individuals_dropped=report.individuals_dropped,
        individuals_skipped=report.individuals_skipped,
        assertions_built=report.assertions_built,
        assertions_inserted=report.assertions_inserted,
        assertions_deduped=report.assertions_deduped,
        assertions_dropped=report.assertions_dropped,
        object_assertions_received=report.object_assertions_received,
        object_assertions_skipped=report.object_assertions_skipped,
        miss=report.miss,
        ambiguous=report.ambiguous,
        skip_counts=report.skip_counts,
        skips=[
            SkipOut(
                scope=s.scope,
                kind=s.kind,
                token=s.token,
                reason=s.reason,
                candidates=list(s.candidates),
                class_name=s.class_name,
                source_id=s.source_id,
            )
            for s in sampled(report.skips)
        ],
    )
