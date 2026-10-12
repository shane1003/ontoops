"""Map parsed individuals to ABox rows. Pure — the resolver is injected, so no DB needed.

Two shapes mirror the schema:

* :class:`MappedIndividual` -> ``abox_individual``
* :class:`MappedValue`      -> ``abox_data_assertion``

An individual's IRI is minted/returned by the database, so values reference their subject by
its **natural key** ``(class_id, identity_key)`` here; the loader swaps that for the real IRI
after upserting individuals.

``identity_key`` mirrors the DB's generated column: the name when there is one, else
``src:<source_id>``. Named individuals therefore merge across sources, while unnamed ones
stay one-per-class-per-source.

Provenance is **source-neutral**: every individual/value carries a ``source_id`` (FK -> ``source``
registry), which may denote a document CQ or an RDB ref. The input field is ``source_ref``
(singular). For compatibility with the earlier input format the document adapter may still send the
legacy ``source_cq_id``; the core resolves ``cq_id -> doc:cq:<cq_id>`` (the deterministic key the
adapter would otherwise compute).

Out of STORAGE scope by design: ``object_property_values`` (no abox_object_assertion table),
embeddings, evidence fields. Object relations are still **counted** (see
:func:`count_object_assertions`) so the load report shows they were discarded rather than
reading as lossless — whether to persist them is an open decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from .resolver import CLASS, DATA_PROPERTY, Unresolved


def source_id_of(ind: dict) -> str | None:
    """Resolve an individual's source_id from the input contract.

    Preferred: ``source_ref`` (singular) — the deterministic source key the adapter emits.
    Earlier input format: fall back to the legacy ``source_cq_id`` and resolve it to the document
    adapter's key ``doc:cq:<cq_id>``.
    """
    ref = ind.get("source_ref")
    if ref:
        return ref
    cq = ind.get("source_cq_id")
    if cq:
        return f"doc:cq:{cq}"
    return None


@dataclass(frozen=True)
class MappedIndividual:
    class_id: str
    canonical_name: str | None
    source_id: str | None

    @property
    def identity_key(self) -> str:
        return self.canonical_name or f"src:{self.source_id}"

    @property
    def key(self) -> tuple[str, str]:
        return (self.class_id, self.identity_key)


@dataclass(frozen=True)
class MappedValue:
    subject_key: tuple[str, str]  # (class_id, identity_key) of the owning individual
    data_prop_id: str
    value: str
    source_id: str | None


@dataclass(frozen=True)
class Skip:
    scope: str  # "individual" | "value"
    kind: str  # "CLASS" | "DATA_PROP"
    token: str | None
    reason: str  # "miss" | "ambiguous" | "null_value" | ...
    candidates: tuple[str, ...]
    class_name: str | None
    source_id: str | None


def count_object_assertions(ind: dict) -> int:
    """Count the instance-level object relations phase1 emitted for this individual.

    Object properties are out of scope for STORAGE (no abox_object_assertion table — see the
    module docstring), but we still COUNT them so the load report shows they were dropped
    instead of reading as lossless. One relation per object-property value, lists expanded,
    nulls ignored — mirroring how data values are counted.
    """
    opv = ind.get("object_property_values")
    if not isinstance(opv, dict):
        return 0
    total = 0
    for value in opv.values():
        if isinstance(value, list):
            total += sum(1 for item in value if item is not None)
        elif value is not None:
            total += 1
    return total


def _literals(value):
    """Yield ``(literal, skip_reason)`` for a data-property value.

    Scalars -> one literal; lists -> one per item; null/objects -> a skip reason.
    """
    if value is None:
        yield (None, "null_value")
    elif isinstance(value, (str, int, float, bool)):
        yield (str(value), None)
    elif isinstance(value, list):
        for item in value:
            if item is None:
                yield (None, "null_value")
            elif isinstance(item, (str, int, float, bool)):
                yield (str(item), None)
            else:
                yield (None, "non_scalar_value")
    else:  # dict / anything else
        yield (None, "non_scalar_value")


def map_individual(
    ind: dict, resolver
) -> tuple[MappedIndividual | None, list[MappedValue], list[Skip]]:
    class_name = ind.get("class_name")
    source_id = source_id_of(ind)
    values: list[MappedValue] = []
    skips: list[Skip] = []

    cls = resolver.resolve(class_name, CLASS)
    if isinstance(cls, Unresolved):
        skips.append(
            Skip("individual", CLASS, class_name, cls.reason, cls.candidates, class_name, source_id)
        )
        return None, values, skips

    canonical_name = ind.get("local_name") or None
    if canonical_name is None and source_id is None:
        # Neither a name nor a source: nothing can identify this individual.
        skips.append(
            Skip("individual", CLASS, class_name, "unidentifiable", (), class_name, source_id)
        )
        return None, values, skips

    subject = MappedIndividual(cls.term_id, canonical_name, source_id)

    dpv = ind.get("data_property_values")
    if dpv and not isinstance(dpv, dict):
        skips.append(
            Skip(
                "individual",
                DATA_PROPERTY,
                None,
                "invalid_data_property_values",
                (),
                class_name,
                source_id,
            )
        )
        dpv = None

    for dp_token, raw in (dpv or {}).items():
        dp = resolver.resolve(dp_token, DATA_PROPERTY)
        if isinstance(dp, Unresolved):
            skips.append(
                Skip(
                    "value",
                    DATA_PROPERTY,
                    dp_token,
                    dp.reason,
                    dp.candidates,
                    class_name,
                    source_id,
                )
            )
            continue
        for literal, reason in _literals(raw):
            if reason:
                skips.append(
                    Skip("value", DATA_PROPERTY, dp_token, reason, (), class_name, source_id)
                )
                continue
            values.append(MappedValue(subject.key, dp.term_id, literal, source_id))

    # A bare individual — no name and no values — carries nothing worth storing. It has a
    # source, so it is identifiable (unlike the case above); there is simply nothing to say
    # about it. Reported, because a drop that no counter mentions is indistinguishable from
    # a load that never received it.
    if canonical_name is None and not values:
        skips.append(Skip("individual", CLASS, class_name, "empty", (), class_name, source_id))
        return None, [], skips

    return subject, values, skips
