"""Dangling-orphan scan + garbage-collection enqueue.

A structural safety net. An *orphan* is a declared class/property whose outgoing structural
reference (``rdfs:domain`` / ``rdfs:range`` / ``rdfs:subClassOf`` / ``rdfs:subPropertyOf``, or a
member of a ``owl:unionOf`` range) points at an IRI in our namespace that is no longer declared —
i.e. the referenced target disappeared, leaving the concept dangling.

DELETE has two reachable paths: human-initiated deletion (the primary, reliable trigger,
applied through ``apply_review_decision``'s ``delete`` action) and this scan (a safety net). In a
purely additive attach-only flow orphans rarely arise — hard-remove already cleans references in
both directions — so this scan mainly guards against partial state, future ABox references, or
bugs. It only *proposes*: each orphan is enqueued as a human-gated ``delete`` review item.
"""

from __future__ import annotations

import logging

from rdflib import Graph, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS

from app.tboxcraft.represent import local_of

logger = logging.getLogger(__name__)

_STRUCTURAL_PREDS = (RDFS.domain, RDFS.range, RDFS.subClassOf, RDFS.subPropertyOf)
_KIND_BY_TYPE = {
    str(OWL.Class): "class",
    str(OWL.ObjectProperty): "object_property",
    str(OWL.DatatypeProperty): "data_property",
}


def _reference_targets(graph: Graph, subj: URIRef) -> set[str]:
    """IRIs the concept structurally references (unions expanded to their members)."""
    targets: set[str] = set()
    for pred in _STRUCTURAL_PREDS:
        for o in graph.objects(subj, pred):
            if isinstance(o, URIRef):
                targets.add(str(o))
            else:  # blank node: owl:unionOf collection
                for u in graph.objects(o, OWL.unionOf):
                    targets.update(
                        str(x) for x in Collection(graph, u) if isinstance(x, URIRef)
                    )
    return targets


def scan_orphans(graph: Graph, base_iri: str) -> list[str]:
    """Declared concepts (in ``base_iri``) with a structural reference to a missing target."""
    subjects = list(graph.subjects(RDF.type, None))
    declared = {str(s) for s in subjects}
    orphans: set[str] = set()
    for subj in subjects:
        s = str(subj)
        if not s.startswith(base_iri):
            continue
        for tgt in _reference_targets(graph, subj):
            if tgt.startswith(base_iri) and tgt not in declared:
                orphans.add(s)
                break
    return sorted(orphans)


def _kind_of(graph: Graph, iri: str) -> str:
    for t in graph.objects(URIRef(iri), RDF.type):
        kind = _KIND_BY_TYPE.get(str(t))
        if kind:
            return kind
    return "class"


def enqueue_orphans(repo, graph: Graph, base_iri: str) -> int:
    """Enqueue a human-gated ``delete`` proposal for each dangling orphan. Idempotent per
    ``(candidate_uid, entry_type)``, so re-running never double-enqueues the same concept.
    ``target_iri`` is placed in the payload for ``apply_review_decision``'s delete branch.
    Returns the number of fresh proposals enqueued."""
    enqueued = 0
    for iri in scan_orphans(graph, base_iri):
        kind = _kind_of(graph, iri)
        rid = repo.enqueue_review(
            base_iri=base_iri,
            candidate_uid=iri,
            kind=kind,
            entry_type="delete",
            payload={
                "name": local_of(iri),
                "kind": kind,
                "target_iri": iri,
                "reason": "dangling orphan: structural reference to a missing target",
            },
            nearest=[],
        )
        if rid is not None:
            enqueued += 1
    if enqueued:
        logger.warning(
            "orphan scan: enqueued %d dangling-orphan delete proposal(s)", enqueued
        )
    return enqueued
