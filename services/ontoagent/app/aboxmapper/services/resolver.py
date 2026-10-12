"""Resolve document tokens to ``tbox_term`` IRIs through ``surface_form``.

The loader is a pure *consumer* of the TBox: representative-term selection and dedup are the
TBox pipeline's job. Resolution is exactly "look up ``(kind, surface)``; use it only if it
maps to a single term":

  * one term    -> Resolved(term_id, lang)
  * zero terms  -> Unresolved("miss")
  * two or more -> Unresolved("ambiguous", candidate IRIs)

Matching is case-insensitive because the TBox indexes ``lower(surface_text)``. Only ACTIVE
terms resolve — DRAFT/DEPRECATED must never be attached to an individual.

Resolution is **scoped to one namespace**. One document is one KG, distinguished by its base
IRI, so the same surface ("Process") legitimately exists in several documents at once. Loading
the whole store would make every shared name ambiguous and silently drop those individuals —
so the second document would break the first. :meth:`TermLookup.load` therefore takes the
document's ``base_iri`` and keeps only terms under it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

CLASS = "CLASS"
DATA_PROPERTY = "DATA_PROP"
KINDS = (CLASS, DATA_PROPERTY)


@dataclass(frozen=True)
class Resolved:
    term_id: str  # tbox_term.id (an IRI)
    lang: str


@dataclass(frozen=True)
class Unresolved:
    reason: str  # "miss" | "ambiguous"
    candidates: tuple[str, ...] = ()  # term IRIs, populated when ambiguous


@dataclass(frozen=True)
class _Entry:
    term_id: str
    lang: str
    form_type: str  # PREF | ALTER | HIDDEN


class EmptyTermNamespace(RuntimeError):
    """The store has ACTIVE terms, but none under the requested base IRI.

    Almost always a wrong/stale base IRI or a TBox that was never loaded for this document.
    Left unchecked it is invisible: every token misses, the load "succeeds" with nothing
    written, and the report reads as a clean run.
    """


# Prefix match, not LIKE: a base IRI can legitimately contain '%' (percent-encoding) or '_',
# which LIKE would read as wildcards and silently widen the scope. Also note there is no
# separator appended — a hash namespace ends in '#' and its local names follow immediately,
# so anything like `base || '/%'` would match nothing at all under hash IRIs. A NULL base
# means "whole store", so one statement covers both callers.
_SQL = """
SELECT sf.term_id, t.term_type, sf.surface_text, sf.lang, sf.form_type
FROM surface_form sf
JOIN tbox_term t ON t.id = sf.term_id
WHERE t.status = 'ACTIVE'
  AND t.term_type = ANY(%s)
  AND (%s::text IS NULL OR starts_with(t.id, %s))
ORDER BY sf.term_id, sf.form_type, sf.lang, sf.surface_text
"""

_COUNT_ACTIVE = """
SELECT count(*) FROM tbox_term WHERE status = 'ACTIVE' AND term_type = ANY(%s)
"""


class TermLookup:
    """In-memory ``(kind, casefolded surface) -> terms`` map, loaded once per run."""

    def __init__(self) -> None:
        self._map: dict[tuple[str, str], list[_Entry]] = {}

    def add(
        self, term_id: str, term_type: str, surface_text: str, lang: str, form_type: str
    ) -> None:
        key = (term_type, surface_text.casefold())
        self._map.setdefault(key, []).append(_Entry(term_id, lang, form_type))

    @classmethod
    def from_rows(cls, rows: Iterable[tuple]) -> TermLookup:
        """Build from ``(term_id, term_type, surface_text, lang, form_type)`` rows."""
        lut = cls()
        for term_id, term_type, surface_text, lang, form_type in rows:
            lut.add(term_id, term_type, surface_text, lang, form_type)
        return lut

    @classmethod
    def load(cls, conn, base_iri: str | None = None) -> TermLookup:
        """Load ACTIVE terms, restricted to ``base_iri``'s namespace when one is given.

        Raises :class:`EmptyTermNamespace` when the namespace filter matches nothing while
        the store does hold ACTIVE terms — that combination means the base IRI is wrong, and
        failing loudly beats a load that resolves nothing and calls itself a success.
        """
        kinds = list(KINDS)
        with conn.cursor() as cur:
            cur.execute(_SQL, (kinds, base_iri, base_iri))
            rows = cur.fetchall()
            if base_iri and not rows:
                cur.execute(_COUNT_ACTIVE, (kinds,))
                if cur.fetchone()[0]:
                    raise EmptyTermNamespace(
                        f"no ACTIVE term lies under {base_iri!r}, though the store has some — "
                        "check the base IRI, or load this document's TBox first"
                    )
            return cls.from_rows(rows)

    def resolve(self, surface: str | None, kind: str) -> Resolved | Unresolved:
        if not surface:
            return Unresolved("miss")
        entries = self._map.get((kind, surface.casefold()))
        if not entries:
            return Unresolved("miss")

        term_ids = {e.term_id for e in entries}
        if len(term_ids) > 1:
            return Unresolved("ambiguous", tuple(sorted(term_ids)))

        # Single term — deterministic lang: prefer a PREF form, then 'en', then alphabetical,
        # so the result never depends on row order.
        pool = [e for e in entries if e.form_type == "PREF"] or entries
        chosen = min(pool, key=lambda e: (0 if e.lang == "en" else 1, e.lang))
        return Resolved(chosen.term_id, chosen.lang)
