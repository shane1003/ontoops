"""Surface-form → IRI mapping table.

Every surface form that is NOT a concept's representative label is recorded here (as JSONL)
so query-time lookups resolve an incoming term to its ontology IRI without bloating the TTL.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.tboxcraft.represent import Concept


@dataclass(frozen=True)
class MappingEntry:
    surface: str
    iri: str
    lang: str
    source: str  # "name" | "correct_term" | "duplicate" | "synonym"


def build_mapping(concepts: list[Concept]) -> list[MappingEntry]:
    entries: list[MappingEntry] = []
    for concept in concepts:
        rep = set(concept.pref_labels.values())
        # (surface, lang, source); first source wins for a given surface within a concept. The
        # language is the member's own — members of one cluster can come from producers that
        # declared different languages, so it cannot be read off the concept.
        surfaces: dict[str, tuple[str, str]] = {}
        for member in concept.members:
            for surface, lang, source in member.surface_candidates():
                if surface and surface not in rep:
                    surfaces.setdefault(surface, (lang, source))
        for surface, (lang, source) in sorted(surfaces.items()):
            entries.append(MappingEntry(surface, concept.iri, lang, source))
    return entries


def scan_surface_collisions(
    concepts: list[Concept], entries: list[MappingEntry]
) -> list[dict]:
    """Strings that do not name exactly one concept — ``[{surface, effect, iris, sources}]``.

    A surface form promises that this string names that concept. Two concepts claiming one string
    breaks the promise, and it breaks in two different ways, because
    :meth:`Repository.resolve_surface` matches ``lower(surface_text)`` over the non-PREF rows with
    ``LIMIT 1`` and no ordering:

    ``ambiguous``
        Two concepts hold the string as a non-PREF surface, so one is returned arbitrarily. This
        is what a name carried by two kinds does — ``hasRole`` as both an object and a data
        property mints two IRIs on purpose and both write ``hasRole``.
    ``shadowed``
        One concept holds it as a non-PREF surface while a *different* concept has it as its
        representative label. The resolver never returns PREF rows, so the string resolves to the
        borrower and the concept actually named that never comes back at all. A single-word name
        equals its own label (``desplit_camel("Population") == "Population"``), so its only row is
        the PREF one — which is exactly the case a curated synonym can take over.

    ``scope`` answers one question: **can naming a kind settle this?** That is what decides
    whether a human has to look at it, so it is computed from the answer rather than from a
    proxy:

    ``cross_kind``
        Every kind has at most one claimant, so a caller that says which kind it wants gets one
        answer. An individual — which can only anchor to a class — has that information.
    ``same_kind``
        Some *one* kind has two or more claimants, so no query-side hint separates them. Either
        they are the same concept and should have merged, or one is misnamed; both are decisions
        above this layer. ``ambiguous_kinds`` names which kinds are in that state.

    Note that these are not "do the claimants span kinds". Three claimants — two data properties
    and an object property — span kinds, yet a resolver asking for a data property still gets two
    answers. Deciding by span marked exactly that case resolvable and skipped it, and the
    individuals naming it were dropped a stage later with nothing queued to explain why.

    Grouped case-insensitively because that is how the resolver reads the column; grouping exactly
    would report fewer collisions than actually occur. PREF labels are grouped in too: two
    concepts whose representative labels differ only in case claim one string, and while
    ``resolve_surface`` returns neither (it excludes PREF), a consumer reading the label — the
    ``Classes`` view exposes ``PrefTerm_en`` — sees both.

    Detection is channel-agnostic on purpose: a ``name``, a ``duplicate`` and a ``synonym`` land in
    the same column, and no producer-side convention can be relied on to keep any of them out.
    """
    kind_of = {concept.iri: concept.kind for concept in concepts}
    # Every claim on a string, from both sides. A PREF label is a claim too: it is how the
    # concept is named, and a consumer that reads labels resolves through it.
    claims: dict[str, dict] = {}

    def claim(text: str) -> dict:
        return claims.setdefault(
            text.lower(), {"surface": text, "non_pref": [], "pref": [], "sources": []}
        )

    for concept in concepts:
        for label in concept.pref_labels.values():
            row = claim(label)
            if concept.iri not in row["pref"]:
                row["pref"].append(concept.iri)
    for entry in entries:
        row = claim(entry.surface)
        row["surface"] = (
            entry.surface
        )  # prefer a written surface's spelling for display
        if entry.iri not in row["non_pref"]:
            row["non_pref"].append(entry.iri)
        if entry.source not in row["sources"]:
            row["sources"].append(entry.source)

    # A string nobody wrote outside PREF is reported only when it names a contest not already
    # visible under a written spelling. `hasRole` and its label `has Role` are one contest
    # between the same two terms; the written one is what a resolver is asked for, so the label
    # would only double the count. A pair that collides *solely* through labels has no written
    # spelling, and that is the case this catches.
    rows = [
        (row, row["non_pref"] + [i for i in row["pref"] if i not in row["non_pref"]])
        for _key, row in sorted(claims.items())
    ]
    written = {
        frozenset(owners) for row, owners in rows if row["non_pref"] and len(owners) > 1
    }

    out: list[dict] = []
    for row, owners in rows:
        if len(owners) < 2:
            continue
        if not row["non_pref"] and frozenset(owners) in written:
            continue
        by_kind: dict[str, list[str]] = {}
        for iri in owners:
            by_kind.setdefault(kind_of.get(iri, "unknown"), []).append(iri)
        ambiguous_kinds = sorted(k for k, v in by_kind.items() if len(v) > 1)
        out.append(
            {
                "surface": row["surface"],
                # One non-PREF claimant beside somebody else's label is the confident-wrong-answer
                # case; anything else with two claimants is a coin toss.
                "effect": (
                    "shadowed"
                    if len(row["non_pref"]) == 1 and len(owners) > 1
                    else "ambiguous"
                ),
                "scope": "same_kind" if ambiguous_kinds else "cross_kind",
                "kinds": sorted(by_kind),
                "ambiguous_kinds": ambiguous_kinds,
                "iris": sorted(owners),
                "sources": sorted(row["sources"]),
            }
        )
    return sorted(out, key=lambda r: r["surface"].lower())


def write_mapping_jsonl(entries: list[MappingEntry], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    return path
