"""Kind-filtered retrieval + shared routing helpers for the 4-layer pipeline.

Holds :class:`DbRetriever` (the pgvector k-NN used by attach-only clustering) and the small
graph helpers the DomainTermApproval layer reuses: name-derived-label detection, IRI
existence, and the property domain/range compatibility gate. The old threshold-band
routing engine (``classify`` / ``route_candidate`` / the ``gray_zone`` resolvers) was removed
when clustering + approval replaced it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdflib import Graph, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS, SKOS

from app.tboxcraft.gloss import desplit_camel
from app.tboxcraft.models import Component
from app.tboxcraft.represent import local_of, mint
from app.tboxcraft.similarity import center_query


@dataclass(frozen=True)
class NearestMatch:
    """One retrieved anchor: ``iri`` and its cosine ``score``. ``uid`` mirrors ``iri`` because
    DB concepts have no separate uid."""

    uid: str
    iri: str
    score: float


class DbRetriever:
    """Kind-filtered pgvector k-NN over the persisted ontology.

    Holds the run's model id and whitening mean so each candidate is centered into the same
    space the stored vectors live in before the cosine search. ``mean`` is ``None`` when
    centering already happened upstream (the attach-only path centers once), in which case this
    is a passthrough. Deprecated concepts are excluded so retired IRIs are never anchors.
    """

    def __init__(
        self, repo, model_id: int, mean: np.ndarray | None, *, base_iri: str, k: int = 5
    ):
        self.repo = repo
        self.model_id = model_id
        self.mean = mean
        # One document is one KG, so a candidate may only attach to a term of its own
        # namespace. Unscoped, document B's `Process` merged onto document A's and B
        # never got a term of its own — the merge decision is a similarity score, and
        # the two documents' glosses are similar precisely because they are the same
        # word. Cross-document connection is the query-time federation's job.
        self.base_iri = base_iri
        self.k = k

    def nearest(self, vector: np.ndarray, kind: str) -> list[NearestMatch]:
        centered = center_query(vector, self.mean)
        rows = self.repo.nearest_concepts(
            centered, self.model_id, self.k, kind=kind, base_iri=self.base_iri
        )
        return [NearestMatch(uid=iri, iri=iri, score=score) for iri, score in rows]

    def exact(self, name: str, kind: str) -> str | None:
        """The live term of ``kind`` written exactly ``name``, or None.

        Sits beside the k-NN rather than inside it because it answers a different question. A
        candidate's vector is its gloss, and a gloss is built from the CQs it came with, so the
        same term described by two documents embeds far apart — two terms both named
        ``Activity`` can land well under any usable tau_high. Similarity cannot see that they are
        the same term; the name says so outright.
        """
        return self.repo.resolve_anchor_by_name(name, kind, base_iri=self.base_iri)


def _pref_label(graph: Graph, iri: str) -> str | None:
    lbl = graph.value(URIRef(iri), SKOS.prefLabel)
    return None if lbl is None else str(lbl)


def _iri_exists(graph: Graph, iri: str) -> bool:
    """True if the IRI is already a declared term (has any rdf:type) in the graph."""
    return (URIRef(iri), RDF.type, None) in graph


def _is_name_derived(graph: Graph, iri: str) -> bool:
    """True when the concept's prefLabel was derived from its name (no curated correct_term),
    i.e. it equals the de-split IRI local part. This is the safe UPDATE trigger: a
    candidate with a curated term is only promoted over a label that was never curated."""
    label = _pref_label(graph, iri)
    return label is not None and label == desplit_camel(local_of(iri))


def _range_class_iris(graph: Graph, prop_iri: str) -> set[str]:
    """Range class IRIs of a property, expanding an owl:unionOf into its members."""
    iris: set[str] = set()
    for r in graph.objects(URIRef(prop_iri), RDFS.range):
        if isinstance(r, URIRef):
            iris.add(str(r))
        else:  # blank node: owl:unionOf collection
            for u in graph.objects(r, OWL.unionOf):
                iris.update(
                    str(x) for x in Collection(graph, u) if isinstance(x, URIRef)
                )
    return iris


def _expected_class_iri(base_iri: str, name: str) -> str:
    return mint(base_iri, "class", name)


def _property_compatible(
    component: Component, target_iri: str, graph: Graph, base_iri: str
) -> bool:
    """Domain/range gate: a property may attach to a matched property only when their
    domain/range are compatible. Block on a *positive* mismatch only — if either side declares no
    constraint we cannot prove incompatibility, so we do not block on that axis (a real mismatch
    downgrades to review; an unprovable one is left to the similarity call)."""
    # Domain (single class).
    graph_dom = graph.value(URIRef(target_iri), RDFS.domain)
    if (
        component.domain
        and graph_dom is not None
        and str(graph_dom) != _expected_class_iri(base_iri, component.domain)
    ):
        return False
    # Range (object property: classes, possibly a union; data property: a datatype).
    cand_ranges = component.range_values()
    if cand_ranges and component.kind != "data_property":
        expected = {_expected_class_iri(base_iri, r) for r in cand_ranges}
        graph_ranges = _range_class_iris(graph, target_iri)
        if graph_ranges and expected != graph_ranges:
            return False
    return True
