"""Build the **TBox** knowledge graph in FalkorDB (OpenCypher over Redis).

Scope is the TBox schema graph — classes, the subclass hierarchy, and object-property
relations between classes. Downstream (OntoRouter) uses it for name/parent lookups and to
ask "which concept relates to which, and how". Data properties (xsd ranges, not classes),
individuals, and data values are NOT written here — those are read from Postgres.

Input is a Turtle file (the canonical TBox snapshot), parsed with rdflib.

One FalkorDB instance holds many *named graphs*, one per document (`ipb` / `adp`);
each run writes to the graph named by ``FALKORDB_GRAPH`` and touches no other.

Graph model
-----------
``(:Class {iri, name})``  — ``name`` is the IRI's last segment, no ``/class/`` prefix
``(:Class)-[:SUBCLASS_OF]->(:Class)``  — child -> parent
``(:Class)-[:OBJECT_PROP {iri, name}]->(:Class)``  — object property, domain -> range
``INDEX ON :Class(iri)``

Object properties use a **single** relationship type ``OBJECT_PROP`` with the property IRI/name
on the edge — NOT one relationship type per property. Cypher can't parameterize relationship
types, and thousands of types make schema introspection useless; consumers filter by ``name``:
``MATCH (a:Class)-[r:OBJECT_PROP {name:$p}]->(b:Class)``. One edge per domain×range pair.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from rdflib import OWL, RDF, RDFS, Graph, URIRef
from rdflib.collection import Collection


@dataclass
class TBox:
    classes: dict[str, str] = field(default_factory=dict)  # iri -> name
    subclass_of: list[tuple[str, str]] = field(default_factory=list)  # (child_iri, parent_iri)
    # (domain_iri, prop_iri, prop_name, range_iri) — one per domain×range of an OBJECT_PROP
    object_edges: list[tuple[str, str, str, str]] = field(default_factory=list)
    # OBJECT_PROP with no class domain or no class range → no edge; counted so the drop shows
    skipped_object_props: int = 0


def _local(iri: str) -> str:
    """Clean name = the IRI's last segment, split on the LAST '#' or '/'.

    Our IRIs look like 'http://.../ns#/class/Weather', so we must split on whichever
    separator comes last — not '#' before '/' — or the name keeps a '/class/' prefix.
    """
    seg = re.split(r"[#/]", iri)[-1]
    return seg or iri


def _add_class(tbox: TBox, o) -> None:
    if isinstance(o, URIRef):
        tbox.classes.setdefault(str(o), _local(str(o)))


def _class_targets(g: Graph, prop, predicate) -> list:
    """Class URIRefs a property points at via ``predicate``, expanding ``owl:unionOf``.

    tboxcraft wraps multiple domains/ranges in ``[ owl:unionOf ( C1 C2 … ) ]`` (a BNode),
    since repeating ``rdfs:domain``/``rdfs:range`` would mean intersection in OWL. A bare
    URIRef is used as-is; a union BNode expands to its class members. Any other BNode form
    (e.g. a Restriction) has no class members, so it yields nothing and the property is
    skip-counted. (While only range was unfolded, a union domain was caught by the BNode filter and
    the whole edge vanished.)
    """
    out = []
    for o in g.objects(prop, predicate):
        if isinstance(o, URIRef):
            out.append(o)
        else:
            col = next(g.objects(o, OWL.unionOf), None)
            if col is not None:
                out.extend(m for m in Collection(g, col) if isinstance(m, URIRef))
    return out


def _tbox_from_graph(g: Graph) -> TBox:
    tbox = TBox()
    for cls in set(g.subjects(RDF.type, OWL.Class)) | set(g.subjects(RDF.type, RDFS.Class)):
        if isinstance(cls, URIRef):
            tbox.classes[str(cls)] = _local(str(cls))
    for sub, sup in g.subject_objects(RDFS.subClassOf):
        if isinstance(sub, URIRef) and isinstance(sup, URIRef):
            tbox.subclass_of.append((str(sub), str(sup)))
            _add_class(tbox, sub)  # every hierarchy endpoint is a named Class node
            _add_class(tbox, sup)
    # Contract: the Class set also includes classes that
    # appear only as an object-property domain/range or a data-property domain. Data-property
    # ranges are xsd datatypes, not classes, so they are excluded.
    #
    # Object properties also become edges: one (:Class)-[:OBJECT_PROP]->(:Class) per
    # domain×range pair (range unions expanded to their members, deduped). A property with no
    # class domain or no class range yields no edge and is counted in ``skipped_object_props``
    # (so the drop is visible, not silent).
    for p in g.subjects(RDF.type, OWL.ObjectProperty):
        domains = sorted(set(_class_targets(g, p, RDFS.domain)), key=str)
        ranges = sorted(set(_class_targets(g, p, RDFS.range)), key=str)
        for o in domains + ranges:
            _add_class(tbox, o)  # every endpoint is a named Class node
        if domains and ranges:
            name = _local(str(p))
            for d in domains:
                for r in ranges:
                    tbox.object_edges.append((str(d), str(p), name, str(r)))
        else:
            tbox.skipped_object_props += 1
    for p in g.subjects(RDF.type, OWL.DatatypeProperty):
        for o in g.objects(p, RDFS.domain):
            _add_class(tbox, o)
    return tbox


def parse_tbox_text(ttl: str) -> TBox:
    g = Graph()
    g.parse(data=ttl, format="turtle")
    return _tbox_from_graph(g)


def parse_tbox_file(path: str | Path) -> TBox:
    g = Graph()
    g.parse(location=str(path), format="turtle")
    return _tbox_from_graph(g)


# ------------------------------------------------------------------ Cypher


def graph_statements(tbox: TBox) -> Iterator[tuple[str, dict]]:
    """Yield (cypher, params) for classes then subclass edges — pure, unit-testable.

    Classes are emitted before edges so the edge MATCH always finds both endpoints. MERGE
    keeps a re-run against the same named graph idempotent (== CREATE on a fresh graph).
    """
    for iri, name in tbox.classes.items():
        yield ("MERGE (c:Class {iri:$iri}) SET c.name = $name", {"iri": iri, "name": name})
    for child, parent in tbox.subclass_of:
        yield (
            ("MATCH (a:Class {iri:$child}), (b:Class {iri:$parent}) MERGE (a)-[:SUBCLASS_OF]->(b)"),
            {"child": child, "parent": parent},
        )
    for dom, prop_iri, prop_name, rng in tbox.object_edges:
        yield (
            (
                "MATCH (a:Class {iri:$dom}), (b:Class {iri:$rng}) "
                "MERGE (a)-[r:OBJECT_PROP {iri:$piri}]->(b) SET r.name = $name"
            ),
            {"dom": dom, "rng": rng, "piri": prop_iri, "name": prop_name},
        )


# ------------------------------------------------------------------ driver


def connect_falkordb(graph: str | None = None):
    """Return the FalkorDB named graph handle for this run.

    ``graph`` selects the named graph for this call; without it the deployment default
    ``FALKORDB_GRAPH`` is used. The handle knows its own name (``.name``), so callers can
    report which graph they actually wrote to.
    """
    from falkordb import FalkorDB

    from ..core.config import get_falkordb_config

    cfg = get_falkordb_config(graph)
    # password=None → the client sends no AUTH, so unauthenticated stacks are unaffected.
    return FalkorDB(host=cfg["host"], port=cfg["port"], password=cfg["password"]).select_graph(
        cfg["graph"]
    )


def _ensure_index(graph) -> None:
    """Create the :Class(iri) index; tolerate an already-present one."""
    try:
        graph.query("CREATE INDEX FOR (c:Class) ON (c.iri)")
    except Exception as exc:
        if "already" in str(exc).lower() or "indexed" in str(exc).lower():
            return
        raise


def load(graph, tbox: TBox) -> dict:
    _ensure_index(graph)
    statements = 0  # data statements only (index is infra) -> classes + subclass + object edges
    for query, params in graph_statements(tbox):
        graph.query(query, params=params)
        statements += 1
    return {
        "classes": len(tbox.classes),
        "subclass_edges": len(tbox.subclass_of),
        "object_edges": len(tbox.object_edges),
        "skipped_props": tbox.skipped_object_props,  # OBJECT_PROP with no edge (missing endpoint)
        "statements": statements,
    }


def run(ttl_path: str | Path, graph) -> dict:
    """Build the TBox graph: parse the TTL and write classes/hierarchy into FalkorDB."""
    return load(graph, parse_tbox_file(ttl_path))
