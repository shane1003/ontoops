"""Orchestration: parse individuals, resolve against the TBox, upsert the ABox, report.

Two-stage write, because a value must point at its owning individual:

1. **Individuals** — batch upsert on the natural key ``(class_id, identity_key)``. The
   statement returns the stable IRI whether the row was new or already there, so re-runs
   reuse the same subject.
2. **Values** — batch upsert of ``abox_data_assertion`` using those IRIs, with
   ``xsd_datatype`` taken from the data property's declared RANGE.

Both stages are idempotent (``ON CONFLICT``), so the loader accumulates across sources and a
repeated run is a no-op. DDL is never issued here — see :mod:`app.aboxmapper.cli.migrate`.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.config import get_ontology_base_iri
from .mapper import (
    MappedIndividual,
    MappedValue,
    Skip,
    count_object_assertions,
    map_individual,
)
from .resolver import TermLookup

DEFAULT_INPUT = "data/input/component.json"


class SourceRegistryMissing(RuntimeError):
    """The `source` registry table (tboxcraft-owned) is absent from this database."""


class UnregisteredSources(RuntimeError):
    """Precondition failure: individuals point at source_ids the `source` registry lacks.

    Raised **before** any write, so the load is all-or-nothing and the caller learns every
    offending source at once instead of the first one the FK happens to reject. This is not
    an individual-quality problem (which belongs in ``skips``) but a signal that the upstream
    ran out of order — swallowing it would leave the store half-full and call it success.
    """

    def __init__(
        self,
        missing: list[str],
        unsourced_subjects: int,
        unsourced_values: int,
        individuals: int,
    ) -> None:
        self.missing = missing  # source_ids absent from the registry
        self.unsourced_subjects = unsourced_subjects  # individuals to write with no source
        # Values are counted separately because a value can outlive its individual: identical
        # subjects collapse to one in `build`, but every value survives. Counting subjects
        # alone therefore misses a sourceless value whose subject lost the de-duplication —
        # order-dependently, which is the worst way to fail.
        self.unsourced_values = unsourced_values  # data values to write with no source
        self.individuals = individuals  # individuals in the payload
        super().__init__(
            f"{len(missing)} unregistered source_id(s), {unsourced_subjects} individual(s) "
            f"and {unsourced_values} value(s) with no source, "
            f"of {individuals} individual(s)"
        )


@dataclass
class LoadReport:
    individuals: int = 0  # individuals in the payload
    base_iri: str | None = None  # namespace actually used (request beats env)
    # The individual side decomposes the same way the value side does, so the input count is
    # fully accounted for and nobody has to guess whether a shrinking number means merging or
    # loss:  individuals = upserted + deduped + dropped + skipped
    individuals_upserted: int = 0  # distinct individuals now in the ABox
    individuals_deduped: int = 0  # merged into another input's subject (same natural key)
    individuals_dropped: int = 0  # subject built but never came back from the write (loss)
    individuals_skipped: int = 0  # no subject at all — each has an `individual` scope skip
    assertions_built: int = 0  # data-property values mapped from the payload
    assertions_inserted: int = 0  # newly inserted (the rest already existed)
    assertions_deduped: int = 0  # mapped but already present (ON CONFLICT — benign)
    assertions_dropped: int = 0  # mapped but their subject never upserted (real loss)
    # Object properties are OUT OF SCOPE for storage (no abox_object_assertion table). We still
    # count them so the report never reads as lossless when instance relations are discarded.
    object_assertions_received: int = 0  # instance relations in the payload
    object_assertions_skipped: int = 0  # of those, not stored (currently: all of them)
    skips: list[Skip] = field(default_factory=list)
    # Totals per skip reason, counted once over every skip. The counts are deliberately NOT
    # derived from whatever list a caller ends up seeing: `skips` is diagnostic and may be
    # sampled downstream, while `miss` is a gate contract. Deriving one from the other would
    # let a response-size decision quietly lower a correctness number.
    skip_counts: dict[str, int] = field(default_factory=dict)

    @property
    def miss(self) -> int:
        return self.skip_counts.get("miss", 0)

    @property
    def ambiguous(self) -> int:
        return self.skip_counts.get("ambiguous", 0)

    def summary(self) -> str:
        return (
            f"base_iri={self.base_iri} "
            f"individuals={self.individuals} upserted={self.individuals_upserted} "
            f"(deduped={self.individuals_deduped} dropped={self.individuals_dropped} "
            f"skipped={self.individuals_skipped}) "
            f"assertions={self.assertions_built}/inserted={self.assertions_inserted} "
            f"(deduped={self.assertions_deduped} dropped={self.assertions_dropped}) "
            f"object_assertions={self.object_assertions_received} "
            f"skipped={self.object_assertions_skipped} "
            f"skips={len(self.skips)} (miss={self.miss} ambiguous={self.ambiguous})"
        )


# ------------------------------------------------------------------ input


def extract_individuals(payload: Any) -> list[dict]:
    """Normalize a JSON payload to the list of individuals.

    Accepts a bare list, an object with an ``individuals`` key, or a full component.json.
    """
    if isinstance(payload, list):
        individuals = payload
    elif isinstance(payload, dict):
        if "individuals" not in payload:
            raise ValueError("object payload must contain an 'individuals' key")
        individuals = payload["individuals"]
    else:
        raise ValueError("payload must be a JSON array or an object with 'individuals'")  # noqa: TRY004
    if not isinstance(individuals, list):
        raise ValueError("'individuals' must be a JSON array")  # noqa: TRY004
    return individuals


def read_input_file(path: str | Path) -> list[dict]:
    return extract_individuals(json.loads(Path(path).read_text(encoding="utf-8")))


# ------------------------------------------------------------------ mapping


def build(
    individuals: Iterable[dict], lookup: TermLookup
) -> tuple[list[MappedIndividual], list[MappedValue], list[Skip], int]:
    """Map every individual, de-duplicating subjects by their natural key.

    Returns ``(subjects, values, skips, mapped)`` where ``mapped`` counts the inputs that
    produced a subject — before de-duplication. Without it the caller cannot tell an input
    that collapsed into an existing subject from one that was dropped, and "input count went
    down" stays unexplained.
    """
    by_key: dict[tuple[str, str], MappedIndividual] = {}
    values: list[MappedValue] = []
    skips: list[Skip] = []
    mapped = 0
    for ind in individuals:
        subject, vals, sk = map_individual(ind, lookup)
        skips.extend(sk)
        if subject is None:
            continue
        mapped += 1
        by_key.setdefault(subject.key, subject)  # first wins; identity is the key
        values.extend(vals)
    return list(by_key.values()), values, skips, mapped


def mint_iri(base: str) -> str:
    """Mint an individual IRI in the namespace style the base IRI declares.

    The target ontology is hash-namespaced (``http://…/IPB#``), where the trailing ``#`` is
    the separator and local names follow it directly. Appending a path to that would produce
    ``…/IPB#/individual/<uuid>`` — a path inside a fragment — and these land in
    ``abox_individual.id``, which is not something to undo later.

        base ends '#'  ->  {base}individual_{uuid}    hash: mil:individual_8f3c…
        base ends '/'  ->  {base}individual/{uuid}    slash, separator already there
        otherwise      ->  {base}/individual/{uuid}   slash, as before

    ``individual_`` keeps the fragment local name clear of the TBox's UpperCamel classes and
    lowerCamel properties.
    """
    uid = uuid.uuid4()
    if base.endswith("#"):
        return f"{base}individual_{uid}"
    if base.endswith("/"):
        return f"{base}individual/{uid}"
    return f"{base}/individual/{uid}"


# ------------------------------------------------------------------ preconditions

# Every source_id a write would use must already exist in the `source` registry (tboxcraft
# owns it; we only read). Asking once, up front, turns a first-row FK abort into a complete
# list — in practice the upstream either ran or did not, so this is usually all or nothing.
_UNREGISTERED_SOURCES = """
SELECT s.source_id
FROM unnest(%s::varchar[]) AS s(source_id)
LEFT JOIN source src USING (source_id)
WHERE src.source_id IS NULL
"""


def require_registered_sources(
    conn,
    subjects: list[MappedIndividual],
    values: list[MappedValue],
    individuals: int,
) -> None:
    """Raise unless every source a write would use is registered. Called before any write.

    Runs in dry-run too, so a pipeline can verify it is safe to load without loading. Only
    sources a write would actually use are checked: an individual skipped for an unresolved
    class never reaches the FK, so failing on its source would be a false alarm. Values are
    checked alongside subjects — and counted apart — because `build` collapses identical
    subjects but keeps every value, so a sourceless value can outlive the subject it came in
    with, order-dependently.

    Args:
        conn: Open connection to the ABox database.
        subjects: Individuals that would be upserted.
        values: Data values that would be written.
        individuals: Size of the original payload, for the error report.

    Raises:
        UnregisteredSources: Some source is absent from the registry, or missing entirely.
        SourceRegistryMissing: The registry table itself is not in this database.
    """
    unsourced_subjects = sum(1 for s in subjects if s.source_id is None)
    unsourced_values = sum(1 for v in values if v.source_id is None)
    missing = unregistered_sources(
        conn, [s.source_id for s in subjects] + [v.source_id for v in values]
    )
    if missing or unsourced_subjects or unsourced_values:
        raise UnregisteredSources(missing, unsourced_subjects, unsourced_values, individuals)


def unregistered_sources(conn, source_ids: Iterable[str | None]) -> list[str]:
    """Return the given source_ids that the `source` registry does not have (sorted).

    Raises :class:`SourceRegistryMissing` when the registry itself is absent — that is a
    deployment-order failure (upstream schema not applied), not an input problem.
    """
    ids = sorted({sid for sid in source_ids if sid})
    if not ids:
        return []
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('source') IS NULL")
        if cur.fetchone()[0]:
            raise SourceRegistryMissing(
                "the `source` registry is missing from this database — apply the "
                "tboxcraft schema before loading (order: tboxcraft schema → "
                "aboxmapper-migrate)"
            )
        cur.execute(_UNREGISTERED_SOURCES, (ids,))
        return [row[0] for row in cur.fetchall()]


# ------------------------------------------------------------------ writes

_UPSERT_INDIVIDUALS = """
INSERT INTO abox_individual (id, class_id, canonical_name, source_id)
SELECT * FROM unnest(%s::text[], %s::text[], %s::text[], %s::varchar[])
ON CONFLICT ON CONSTRAINT uq_abox_individual_identity
    DO UPDATE SET class_id = EXCLUDED.class_id   -- no-op; makes RETURNING emit the existing row
RETURNING id, class_id, identity_key
"""

_RANGE_DATATYPES = """
SELECT prop_id, min(xsd_datatype)
FROM tbox_domain_range
WHERE relation_type = 'RANGE' AND xsd_datatype IS NOT NULL AND prop_id = ANY(%s)
GROUP BY prop_id
"""

_UPSERT_ASSERTIONS = """
INSERT INTO abox_data_assertion
    (subject_id, data_prop_id, value, xsd_datatype, source_id)
SELECT * FROM unnest(%s::text[], %s::text[], %s::text[], %s::varchar[], %s::varchar[])
ON CONFLICT ON CONSTRAINT uq_abox_data_assertion DO NOTHING
"""


def upsert_individuals(
    conn, subjects: list[MappedIndividual], base_iri: str
) -> dict[tuple[str, str], str]:
    """Upsert individuals; return ``(class_id, identity_key) -> stable IRI``."""
    if not subjects:
        return {}
    ids = [mint_iri(base_iri) for _ in subjects]
    classes = [s.class_id for s in subjects]
    names = [s.canonical_name for s in subjects]
    source_ids = [s.source_id for s in subjects]
    with conn.cursor() as cur:
        cur.execute(_UPSERT_INDIVIDUALS, (ids, classes, names, source_ids))
        return {(class_id, identity_key): iri for iri, class_id, identity_key in cur.fetchall()}


def fetch_range_datatypes(conn, prop_ids: Iterable[str]) -> dict[str, str]:
    props = sorted(set(prop_ids))
    if not props:
        return {}
    with conn.cursor() as cur:
        cur.execute(_RANGE_DATATYPES, (props,))
        return dict(cur.fetchall())


def upsert_assertions(
    conn,
    values: list[MappedValue],
    iri_by_key: dict[tuple[str, str], str],
    datatypes: dict[str, str],
) -> tuple[int, int]:
    """Insert value assertions idempotently.

    Returns ``(inserted, eligible)`` where ``eligible`` is the number of values whose subject
    was actually upserted (so the caller can tell ON CONFLICT dedup apart from values dropped
    because their subject never landed — the two are indistinguishable from ``inserted`` alone).
    """
    rows = [
        (
            iri_by_key[v.subject_key],
            v.data_prop_id,
            v.value,
            datatypes.get(v.data_prop_id, "xsd:string"),
            v.source_id,
        )
        for v in values
        if v.subject_key in iri_by_key
    ]
    if not rows:
        return 0, 0
    cols = list(zip(*rows))
    with conn.cursor() as cur:
        cur.execute(_UPSERT_ASSERTIONS, tuple(list(c) for c in cols))
        return max(cur.rowcount, 0), len(rows)


# ------------------------------------------------------------------ run


def run(
    conn, individuals: list[dict], dry_run: bool = False, base_iri: str | None = None
) -> LoadReport:
    """Load ``individuals`` into the namespace ``base_iri`` names (default: the env's).

    The base IRI scopes both halves of identity: which TBox terms may resolve (one document
    is one namespace) and where new individual IRIs are minted. It is resolved and validated
    first, so a bad namespace fails before anything is read or written.
    """
    base = get_ontology_base_iri(base_iri)
    lookup = TermLookup.load(conn, base)
    subjects, values, skips, mapped = build(individuals, lookup)
    object_received = sum(count_object_assertions(ind) for ind in individuals)
    report = LoadReport(
        individuals=len(individuals),
        base_iri=base,  # echoed: the request beats the env, so env can't be trusted
        individuals_deduped=mapped - len(subjects),  # merged, not lost
        individuals_skipped=len(individuals) - mapped,  # every one carries a skip
        assertions_built=len(values),
        # Object relations are not stored (out of scope) → every one received is skipped.
        object_assertions_received=object_received,
        object_assertions_skipped=object_received,
        skips=skips,
        skip_counts=dict(Counter(skip.reason for skip in skips)),
    )

    require_registered_sources(conn, subjects, values, len(individuals))

    if dry_run:
        report.individuals_upserted = len(subjects)
        return report

    iri_by_key = upsert_individuals(conn, subjects, base)
    datatypes = fetch_range_datatypes(conn, (v.data_prop_id for v in values))
    report.individuals_upserted = len(iri_by_key)
    # A subject that went in but did not come back is a real loss, like assertions_dropped.
    report.individuals_dropped = len(subjects) - len(iri_by_key)
    inserted, eligible = upsert_assertions(conn, values, iri_by_key, datatypes)
    report.assertions_inserted = inserted
    report.assertions_deduped = eligible - inserted  # already present (benign)
    report.assertions_dropped = report.assertions_built - eligible  # subject never upserted
    conn.commit()
    return report
