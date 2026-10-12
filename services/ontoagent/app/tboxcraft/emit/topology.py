"""Relational topology extracted from the emitted TBox graph.

The TBox graph is the source of truth for structure — it is what the syntax/DL/HermiT gate
validated. The relational store mirrors it so consumers can query hierarchy and property
signatures in SQL without parsing Turtle. This module is the single translation point:
graph triples -> ``tbox_hierarchy`` / ``tbox_domain_range`` rows.

Pure and DB-free: it reads a :class:`rdflib.Graph` and returns plain tuples, so the mapping is
testable without a database and the persistence layer stays a thin writer.

Modeling notes it has to undo (see :mod:`tboxcraft.emit.tbox`):

* a multi-class object-property range is emitted as an ``owl:unionOf`` blank node — the relational
  form has no unions, so the members are flattened into one RANGE row each (the union is implied
  by there being more than one row);
* a data-property range is an ``xsd:*`` datatype IRI, which is stored as the ``xsd:name`` text in
  ``xsd_datatype`` rather than as a class reference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rdflib import BNode, Graph, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDFS, XSD

logger = logging.getLogger(__name__)

DOMAIN = "DOMAIN"
RANGE = "RANGE"

# (prop_iri, relation_type, target_class_iri | None, xsd_datatype | None)
DomainRangeRow = tuple[str, str, str | None, str | None]


@dataclass
class Topology:
    """Relational edges mirroring the graph's structural axioms."""

    # (child_iri, parent_iri) from rdfs:subClassOf / rdfs:subPropertyOf. Acyclic by construction.
    hierarchy: list[tuple[str, str]] = field(default_factory=list)
    domain_range: list[DomainRangeRow] = field(default_factory=list)
    # Edges removed from ``hierarchy`` because they closed a cycle — the store cannot hold one.
    cycle_edges_dropped: list[tuple[str, str]] = field(default_factory=list)


def break_cycles(
    edges: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split ``(child, parent)`` edges into ``(acyclic, dropped)``.

    A subClassOf cycle is not an OWL error — it says the classes are equivalent, and HermiT accepts
    it — so the gate passes and the graph keeps it. The relational mirror cannot: a DB trigger
    refuses any insert that closes a multi-hop loop. The schema's own CHECK only catches the
    one-hop case (``child_id <> parent_id``), which is also all this module used to skip, so a
    two-hop loop travelled all the way to the trigger and came back as a 500. Every other thing
    this pipeline cannot store is dropped and counted; this one raised instead.

    Deterministic and order-independent: edges are processed sorted, and an edge is dropped only
    when the edges already kept make its parent a descendant of its child. Which edge of a loop
    survives is therefore a stable function of the IRIs, not of graph iteration order — the same
    batch breaks the same way twice. It is still a choice, not a repair, which is why the dropped
    edges are reported rather than merely counted.
    """
    parents: dict[str, set[str]] = {}
    kept: list[tuple[str, str]] = []
    dropped: list[tuple[str, str]] = []

    def reaches(start: str, goal: str) -> bool:
        """Is ``goal`` an ancestor of ``start`` along the edges kept so far?"""
        seen, stack = {start}, [start]
        while stack:
            for nxt in parents.get(stack.pop(), ()):
                if nxt == goal:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    for child, parent in sorted(set(edges)):
        if child == parent or reaches(parent, child):
            dropped.append((child, parent))
            continue
        parents.setdefault(child, set()).add(parent)
        kept.append((child, parent))
    return kept, dropped


def _xsd_name(ref: URIRef) -> str | None:
    """``URIRef(XSD.string)`` -> ``'xsd:string'``; None when not an XSD datatype."""
    return f"xsd:{str(ref)[len(str(XSD)) :]}" if str(ref).startswith(str(XSD)) else None


def _range_targets(graph: Graph, obj) -> list[URIRef]:
    """Resolve a range object to concrete class IRIs, unwrapping an ``owl:unionOf`` node."""
    if not isinstance(obj, BNode):
        return [obj]
    members: list[URIRef] = []
    for collection_node in graph.objects(obj, OWL.unionOf):
        members.extend(
            m for m in Collection(graph, collection_node) if isinstance(m, URIRef)
        )
    return members


def extract_topology(graph: Graph) -> Topology:
    """Read structural axioms out of an emitted TBox graph.

    Blank nodes are skipped as subjects/parents (only named terms are stored), and deprecated
    tombstones contribute nothing because :func:`~tboxcraft.review._migrate_iri` moves their
    axioms to the renewed IRI before the tombstone triples are added.
    """
    topo = Topology()

    for predicate in (RDFS.subClassOf, RDFS.subPropertyOf):
        for child, parent in graph.subject_objects(predicate):
            if isinstance(child, BNode) or isinstance(parent, BNode) or child == parent:
                continue
            topo.hierarchy.append((str(child), str(parent)))

    for prop, target in graph.subject_objects(RDFS.domain):
        if isinstance(prop, BNode):
            continue
        # Multiple domains arrive as an owl:unionOf BNode, like range (repeating
        # rdfs:domain means intersection, so it cannot be used): unfold into one DOMAIN
        # row per member. A non-union BNode (Restriction etc.) has no members, so as
        # before it is not expressed relationally.
        for member in _range_targets(graph, target):
            topo.domain_range.append((str(prop), DOMAIN, str(member), None))

    for prop, target in graph.subject_objects(RDFS.range):
        if isinstance(prop, BNode):
            continue
        for member in _range_targets(graph, target):
            xsd = _xsd_name(member)
            topo.domain_range.append(
                (str(prop), RANGE, None, xsd)
                if xsd
                else (str(prop), RANGE, str(member), None)
            )

    # Deterministic order so a snapshot's rows are reproducible run to run.
    topo.hierarchy, topo.cycle_edges_dropped = break_cycles(topo.hierarchy)
    if topo.cycle_edges_dropped:
        logger.warning(
            "topology: dropped %d hierarchy edge(s) that closed a cycle — e.g. %s; the graph "
            "keeps them (a subClassOf loop is legal OWL) but the store cannot",
            len(topo.cycle_edges_dropped),
            ", ".join(f"{c} -> {p}" for c, p in topo.cycle_edges_dropped[:3]),
        )
    topo.hierarchy.sort()
    topo.domain_range.sort(key=lambda r: (r[0], r[1], r[2] or "", r[3] or ""))
    return topo
