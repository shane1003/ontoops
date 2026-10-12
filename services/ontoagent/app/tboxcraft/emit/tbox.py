"""Author the OWL TBox and serialize it.

Built with rdflib for direct, testable triple assembly from the concept structure. The same
graph serializes to Turtle (primary artifact) and RDF/XML (OOPS! input + owlready2/HermiT
ingestion in validation). Modeling rules: single domain/range per property, a comma-separated
object-property range becomes an ``owl:unionOf`` (never multiple range triples — OOPS! P19),
one ``skos:prefLabel`` per language mirrored to ``rdfs:label``, and an ``owl:Ontology`` header
with ``versionIRI`` + license (avoids OOPS! P38/P41).
"""

from __future__ import annotations

import logging
from pathlib import Path

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD

from app.tboxcraft.models import KIND_CLASS, KIND_DATA_PROPERTY, KIND_OBJECT_PROPERTY
from app.tboxcraft.represent import Concept

logger = logging.getLogger(__name__)


# HermiT reasons only over the OWL 2 datatype map; anything outside it makes the reasoner abort
# ("UnsupportedDatatypeException"), which fails the gate. This used to be a deny-list of the one
# datatype we had hit (xsd:date), so the next unlisted one (xsd:gYear) reproduced the same failure.
# It is an allow-list now: the map is a closed set in the standard, so membership is the question,
# not "have we seen this one break yet".
# https://www.w3.org/TR/owl2-syntax/#Datatype_Maps
_OWL2_DATATYPES = frozenset(
    {
        "decimal",
        "integer",
        "nonNegativeInteger",
        "nonPositiveInteger",
        "positiveInteger",
        "negativeInteger",
        "long",
        "int",
        "short",
        "byte",
        "unsignedLong",
        "unsignedInt",
        "unsignedShort",
        "unsignedByte",
        "double",
        "float",
        "string",
        "normalizedString",
        "token",
        "language",
        "Name",
        "NCName",
        "NMTOKEN",
        "boolean",
        "hexBinary",
        "base64Binary",
        "anyURI",
        "dateTime",
        "dateTimeStamp",
    }
)

# Substitutes for datatypes outside the map. xsd:date -> xsd:dateTime keeps a temporal range
# temporal. The Gregorian fragments and time/duration have no OWL 2 datatype whose value space
# contains theirs, so they fall back to xsd:string: it is the only choice that cannot reject a
# well-formed value of the original type later (xsd:integer would refuse "1985+05:00" or "-0002"
# for gYear). Every substitution loses meaning, which is why it is reported rather than silent.
_XSD_REMAP = {
    "date": "dateTime",
    "gYear": "string",
    "gYearMonth": "string",
    "gMonth": "string",
    "gDay": "string",
    "gMonthDay": "string",
    "time": "string",
    "duration": "string",
}

_FALLBACK_DATATYPE = "string"


def resolve_xsd(name: str) -> tuple[URIRef, str | None]:
    """Resolve an ``xsd:*`` name to a datatype IRI the reasoner accepts.

    Returns ``(iri, substituted_from)`` where ``substituted_from`` is the original local name when
    it is not in the OWL 2 datatype map and had to be replaced — the caller reports that, so a
    downgrade never passes unnoticed.
    """
    local = name.split(":", 1)[-1] if ":" in name else name
    if local in _OWL2_DATATYPES:
        return getattr(XSD, local), None
    replacement = _XSD_REMAP.get(local, _FALLBACK_DATATYPE)
    logger.warning(
        "datatype xsd:%s is outside the OWL 2 datatype map (HermiT cannot reason over it); "
        "emitting xsd:%s instead",
        local,
        replacement,
    )
    return getattr(XSD, replacement, XSD.string), local


def _xsd_ref(name: str) -> URIRef:
    """:func:`resolve_xsd` without the substitution flag, for callers that only build triples."""
    return resolve_xsd(name)[0]


def scan_unresolved_broaders(components) -> list[dict]:
    """Broader names that no component in the batch defines, as ``{name, broader}`` rows.

    ``build_graph`` only emits ``rdfs:subClassOf`` when the broader name resolves to a class it
    knows, so a parent that is merely referenced disappears with no trace — the ontology looks
    complete while the hierarchy is missing edges. Reported so the gap is visible; it is not an
    error, since a batch legitimately references concepts it does not itself define.
    """
    known = {c.name for c in components if c.kind == KIND_CLASS}
    rows: list[dict] = []
    for c in components:
        if c.kind != KIND_CLASS:
            continue
        for broader in c.authoritative_hierarchy().broader:
            if broader not in known and broader != c.name:
                rows.append({"name": c.name, "broader": broader})
    return rows


def _case_variant_map(known_names) -> dict[str, str]:
    """Maps casefold → the spelling that actually exists, to diagnose unresolved reports.

    "References a class this batch does not define" (legitimate: leave it) and "exists
    **under a different spelling**, folded or written otherwise" (a pipeline defect: fix
    it) call for different work from a reviewer. On the cold path all of the batch's
    classes are known; on the warm path only the batch classes come in (stored spellings
    went through slugify in mint and cannot be recovered from the fragment, so they are
    not compared here; in-batch folds are removed up front by fold_case_variants'
    reference carry-over).
    """
    return {n.casefold(): n for n in known_names}


def scan_unresolved_domains(components, resolves, *, case_variants=None) -> list[dict]:
    """Property domain names that do not name a class, as ``{name, unresolved, resolved}``.

    The domain counterpart of :func:`scan_unresolved_ranges`: it must take the same
    ``resolves`` rule to report the same picture as the emission. Unlike ranges it looks
    at **both property kinds**: a data property's domain is a class too (only the range is
    a literal type).

    Reporting rather than raising is the same judgment: a batch referencing a class it
    does not define is legitimate. But it must not be silent: properties lost their
    rdfs:domain without a single log line, and vanished from the graph edge and all,
    with nobody noticing.
    """
    rows: list[dict] = []
    for component in components:
        if component.kind == KIND_CLASS:
            continue
        domains = component.domain_values()
        unresolved = [d for d in domains if not resolves(d)]
        if unresolved:
            rows.append(
                {
                    "name": component.name,
                    "unresolved": unresolved,
                    # Empty means no rdfs:domain went out at all; non-empty means a
                    # **narrower** claim than the input went out, indistinguishable from
                    # outside from a property that had only one domain to begin with.
                    "resolved": [d for d in domains if resolves(d)],
                    # If an unresolved name exists in a spelling differing only by case,
                    # that spelling: "exists differently" rather than "absent" signals a
                    # pipeline defect (_case_variant_map).
                    "case_variants": {
                        d: (case_variants or {}).get(d.casefold())
                        for d in unresolved
                        if (case_variants or {}).get(d.casefold())
                    },
                }
            )
    return rows


def scan_unresolved_ranges(components, resolves, *, case_variants=None) -> list[dict]:
    """Object-property range names that do not name a class, as ``{name, unresolved, resolved}``.

    ``resolves(name) -> bool`` must be the *same* rule emission uses — the cold path resolves
    against the class names this batch defines, the warm path against the accumulated graph — or
    this reports gaps that are not there and misses the ones that are.

    A count alone cannot express what goes wrong here, which is why the resolved names travel too.
    A property whose every range name fails emits no ``rdfs:range`` and is at least visibly
    missing. A property with ``range [A, B]`` where only ``A`` resolves emits ``rdfs:range A`` —
    a *narrower* claim than the input made, and from the outside indistinguishable from a property
    that only ever had one range. Nothing downstream can recover the difference: the union that
    should have been written is simply not there.

    The counterpart of :func:`scan_unresolved_broaders` for ranges, and the same judgment applies —
    reported, not raised. A batch may legitimately reference classes it does not define.

    Object properties only. A data property takes ``ranges[0]`` through the datatype map, which
    :func:`scan_datatype_substitutions` already reports on.
    """
    rows: list[dict] = []
    for component in components:
        if component.kind != KIND_OBJECT_PROPERTY:
            continue
        ranges = component.range_values()
        unresolved = [r for r in ranges if not resolves(r)]
        if unresolved:
            rows.append(
                {
                    "name": component.name,
                    "unresolved": unresolved,
                    # Empty means no rdfs:range at all; non-empty means the emitted range is narrower
                    # than the input asked for, which is the case a count cannot show.
                    "resolved": [r for r in ranges if resolves(r)],
                    # Same diagnosis as scan_unresolved_domains: "absent" vs "exists
                    # under a different spelling".
                    "case_variants": {
                        r: (case_variants or {}).get(r.casefold())
                        for r in unresolved
                        if (case_variants or {}).get(r.casefold())
                    },
                }
            )
    return rows


def scan_datatype_substitutions(components) -> list[dict]:
    """Which data-property ranges will be substituted, as ``{name, requested, emitted}`` rows.

    Derived from the components rather than collected while building the graph, so the pipeline
    can report it without threading a sink through every triple-writing helper.
    """
    rows: list[dict] = []
    for c in components:
        if getattr(c, "kind", None) != KIND_DATA_PROPERTY:
            continue
        ranges = c.range_values()
        if not ranges:
            continue
        _, substituted = resolve_xsd(ranges[0])
        if substituted:
            rows.append(
                {
                    "name": c.name,
                    "requested": f"xsd:{substituted}",
                    "emitted": f"xsd:{_XSD_REMAP.get(substituted, _FALLBACK_DATATYPE)}",
                }
            )
    return rows


def class_iri_by_name(concepts) -> dict[str, str]:
    """Class names (every member name of a class concept) to that concept's IRI.

    Shared rather than rebuilt per caller: this is the rule that decides whether a ``broader`` /
    ``domain`` / ``range`` name resolves to an edge at all, and source attribution has to resolve
    names exactly the way emission did or it attributes provenance to an edge that was never
    emitted (or misses one that was).
    """
    return {m.name: c.iri for c in concepts if c.kind == KIND_CLASS for m in c.members}


def _canonical(concept: Concept):
    return next(m for m in concept.members if m.name == concept.canonical_name)


def build_graph(
    concepts: list[Concept],
    *,
    base_iri: str,
    version_iri: str,
    license_iri: str,
) -> Graph:
    g = Graph()
    # The namespace URI comes from base_iri (config), but the prefix label is a serialization
    # detail with no domain meaning — a fixed neutral "onto" rather than a domain-flavored "mil",
    # so the emitted Turtle reads the same whatever domain the store holds.
    ns = Namespace(base_iri + "/")
    g.bind("skos", SKOS)
    g.bind("owl", OWL)
    g.bind("onto", ns)

    onto = URIRef(base_iri)
    g.add((onto, RDF.type, OWL.Ontology))
    g.add((onto, OWL.versionIRI, URIRef(version_iri)))
    g.add((onto, DCTERMS.license, URIRef(license_iri)))

    class_iri: dict[str, URIRef] = {
        n: URIRef(i) for n, i in class_iri_by_name(concepts).items()
    }

    for concept in concepts:
        iri = URIRef(concept.iri)
        canon = _canonical(concept)

        if concept.kind == KIND_CLASS:
            g.add((iri, RDF.type, OWL.Class))
        elif concept.kind == KIND_OBJECT_PROPERTY:
            g.add((iri, RDF.type, OWL.ObjectProperty))
        else:
            g.add((iri, RDF.type, OWL.DatatypeProperty))

        # One prefLabel/label pair per language the concept carries. Indexing this with a literal
        # "en" (which is what it did) both asserted a language nothing had checked and made a
        # multilingual concept unrepresentable — the second language had nowhere to go.
        for lang, label in sorted(concept.pref_labels.items()):
            g.add((iri, SKOS.prefLabel, Literal(label, lang=lang)))
            g.add((iri, RDFS.label, Literal(label, lang=lang)))
        if canon.dom_notes:
            g.add(
                (iri, RDFS.comment, Literal(canon.dom_notes, lang=canon.comment_lang))
            )

        if concept.kind == KIND_CLASS:
            for broader in canon.authoritative_hierarchy().broader:
                if broader in class_iri and class_iri[broader] != iri:
                    g.add((iri, RDFS.subClassOf, class_iri[broader]))
            continue

        # Property domain: single, or owl:unionOf for a multi-value domain. The producer
        # sends domains as comma-separated multi-values like ranges; while they were
        # compared whole, a multi-value domain matched no class and rdfs:domain silently
        # dropped out. Why not several triples: same as range. In OWL, repeated
        # rdfs:domain means intersection, asserting a constraint no individual can satisfy,
        # the opposite of the producer's "any of these".
        domain_iris = [
            class_iri[d] for d in canon.domain_values() if d in class_iri
        ]
        if len(domain_iris) == 1:
            g.add((iri, RDFS.domain, domain_iris[0]))
        elif len(domain_iris) > 1:
            g.add((iri, RDFS.domain, _class_union_node(g, domain_iris)))

        # Property range: single, or owl:unionOf for a multi-value object-property range.
        ranges = canon.range_values()
        if concept.kind == KIND_DATA_PROPERTY:
            if ranges:
                g.add((iri, RDFS.range, _xsd_ref(ranges[0])))
        else:  # object property
            range_iris = [class_iri[r] for r in ranges if r in class_iri]
            if len(range_iris) == 1:
                g.add((iri, RDFS.range, range_iris[0]))
            elif len(range_iris) > 1:
                g.add((iri, RDFS.range, _class_union_node(g, range_iris)))

    return g


def _class_union_node(g: Graph, member_iris: list) -> BNode:
    """Assembles an anonymous ``owl:unionOf`` class node; domain and range share this code.

    (While only one side had a union, the other dropped multi-values whole; with the
    assembly in one place, the two can never diverge again.)
    """
    union_node = BNode()
    g.add((union_node, RDF.type, OWL.Class))
    members_node = BNode()
    Collection(g, members_node, list(member_iris))
    g.add((union_node, OWL.unionOf, members_node))
    return union_node


def serialize_turtle(graph: Graph) -> str:
    return graph.serialize(format="turtle")


def serialize_rdfxml(graph: Graph) -> str:
    return graph.serialize(format="pretty-xml")


def write_tbox(
    graph: Graph, ttl_path: str | Path, rdfxml_path: str | Path | None = None
) -> Path:
    ttl_path = Path(ttl_path)
    ttl_path.parent.mkdir(parents=True, exist_ok=True)
    ttl_path.write_text(serialize_turtle(graph), encoding="utf-8")
    if rdfxml_path is not None:
        Path(rdfxml_path).write_text(serialize_rdfxml(graph), encoding="utf-8")
    return ttl_path
