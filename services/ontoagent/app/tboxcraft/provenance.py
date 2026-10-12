"""Source attribution — which sources assert a term, and which assert an edge.

Provenance used to be CQ-only. That holds for document input and fails for RDB input, where a term
comes from a table row and there is no CQ, so the spine was generalised to ``source`` and the
consuming side references ``source_id`` alone (see ``db/schema/01_tbox_schema.sql``).

Two jobs live here:

* **Term attribution** is easy — a component names its own sources.
* **Edge attribution is collected here rather than read back from the graph**, because by the time
  a ``subClassOf`` triple exists the components are already merged into concepts and the graph
  carries no record of which component contributed which axiom. ``extract_topology`` reads that
  graph, so it cannot attribute anything. Collecting before the merge and resolving names the same
  way emission does is the only way to get this right.

Provenance is asserted-only for now. An edge that appears solely as a consequence of the merge has
no asserting source, and that is the correct state, not a gap to fill: attaching a source to it
would make the provenance chain say something the input never said. Those edges are counted
(``unattributed_edges`` from :meth:`Repository.replace_topology`), not invented.
"""

from __future__ import annotations

from rdflib import URIRef
from rdflib.namespace import OWL, RDF

from app.tboxcraft.emit.tbox import class_iri_by_name, resolve_xsd
from app.tboxcraft.emit.topology import DOMAIN, RANGE
from app.tboxcraft.models import KIND_CLASS, KIND_DATA_PROPERTY
from app.tboxcraft.represent import mint


def cq_source_id(cq_id: str) -> str:
    """The ``source.source_id`` a CQ maps to, per the ingestion contract's deterministic keys.

    Deliberately not hashed: the adapter computes this same string offline, without talking to us,
    and both sides must land on the identical value. A readable natural key is also what makes a
    re-ingest idempotent — the upsert recognises the row. RDB sources use
    ``rdb:<dsn>.<schema>.<table>[#col=<col>|#pk=<pk>]`` and are computed by that adapter, so they
    have no constructor here.
    """
    return f"doc:cq:{cq_id}"


def component_source_ids(component) -> list[str]:
    """Every source a component cites, as the union of both contract fields.

    Union rather than precedence: ``term_source_map`` is keyed ``(term_id, source_id)`` so repeats
    collapse on their own, and an adapter that fills the two fields partially still keeps every
    source. A priority rule would silently drop the loser and is not idempotent under re-ingest.
    """
    seen = dict.fromkeys(component.source_refs)  # ordered set, adapter order preserved
    seen.update(dict.fromkeys(cq_source_id(c) for c in component.source_cq_ids))
    return list(seen)


def collect_term_sources(concepts) -> list[tuple[str, str]]:
    """``(term_iri, source_id)`` rows for every concept, unioned across its members.

    A cluster merges components that may come from different documents, so a minted term carries
    the sources of all of them — the map is m:n for exactly this reason.
    """
    pairs: dict[tuple[str, str], None] = {}
    for concept in concepts:
        for member in concept.members:
            for sid in component_source_ids(member):
                pairs[(concept.iri, sid)] = None
    return list(pairs)


def _edge_sources(claims, resolve_class) -> dict[str, dict]:
    """The one keying rule, shared by both paths — ``claims`` are ``(own_iri, kind, component)``.

    Cold and warm differ in exactly one thing, how a ``broader`` / ``domain`` / ``range`` name
    becomes a class IRI, so that is the only part passed in. Everything else (which axioms a kind
    contributes, the datatype substitution, the union split) has to agree with
    :func:`~tboxcraft.emit.topology.extract_topology` down to the tuple, and a second copy of it
    would be a second thing to keep in step.

    ``resolve_class`` returns the class IRI a name resolves to, or None when it resolves to
    nothing. None means *no key*: the edge was never emitted, and attaching a source to an edge
    that does not exist is the one failure mode this module is built to avoid.
    """
    hierarchy: dict[tuple, list[str]] = {}
    domain_range: dict[tuple, list[str]] = {}

    def add(bucket, key, sources):
        seen = bucket.setdefault(key, [])
        seen.extend(s for s in sources if s not in seen)

    for own_iri, kind, component in claims:
        sources = component_source_ids(component)
        if not sources or own_iri is None:
            continue

        if kind == KIND_CLASS:
            # Only classes get subClassOf; emission writes no property hierarchy.
            for broader in component.authoritative_hierarchy().broader:
                parent = resolve_class(broader)
                if parent is not None and parent != own_iri:
                    add(hierarchy, (own_iri, parent), sources)
            continue

        target = resolve_class(component.domain) if component.domain else None
        if target is not None:
            add(domain_range, (own_iri, DOMAIN, target, None), sources)

        ranges = component.range_values()
        if kind == KIND_DATA_PROPERTY:
            # Only the first range is emitted for a data property, and it goes through the OWL 2
            # datatype substitution — key on the substituted name or nothing will match.
            if ranges:
                iri, _ = resolve_xsd(ranges[0])
                add(
                    domain_range,
                    (own_iri, RANGE, None, f"xsd:{str(iri).rsplit('#', 1)[-1]}"),
                    sources,
                )
        else:
            # A multi-class range is emitted as owl:unionOf and read back as one row per member,
            # so each resolved class is its own key.
            for value in ranges:
                target = resolve_class(value)
                if target is not None:
                    add(domain_range, (own_iri, RANGE, target, None), sources)

    return {"hierarchy": hierarchy, "domain_range": domain_range}


def collect_edge_sources(concepts) -> dict[str, dict]:
    """Bootstrap edge provenance, keyed the way :func:`extract_topology` keys the stored rows.

    Collected from **every** member of every concept, not just each cluster's canonical member:
    when two members declare the same broader, both genuinely assert that edge and both sources
    are kept. Keys a member declares that emission did not produce (a broader only a non-canonical
    member names) simply never match a stored edge and are ignored — harmless, and cheaper than
    replicating the canonical-member rule a second time.

    Names resolve against the batch alone, because on the cold path the batch is the whole
    ontology. :func:`collect_edge_sources_warm` is the same collection against a store that
    already holds concepts.

    Returns the ``edge_sources`` shape :meth:`Repository.replace_topology` consumes.
    """
    iri_by_uid = {m.uid: c.iri for c in concepts for m in c.members}
    kind_by_iri = {c.iri: c.kind for c in concepts}
    class_iri = class_iri_by_name(concepts)

    claims = (
        (iri_by_uid.get(m.uid), kind_by_iri.get(iri_by_uid.get(m.uid)), m)
        for c in concepts
        for m in c.members
    )
    return _edge_sources(claims, class_iri.get)


def collect_edge_sources_warm(
    outcomes, iri_by_outcome_id, graph, base_iri
) -> dict[str, dict]:
    """The same collection for an update batch, resolving names against the mutated graph.

    A warm ``broader`` may name a class this batch never sent — it is already in the store — so
    :func:`class_iri_by_name` cannot resolve it and every warm edge came back unattributed. The
    graph passed here is the store's snapshot with this batch's concepts already added, so it
    *is* the store-aware index: reconstructing ``{base_iri}/{segment}/{slug}`` and asking whether
    the graph declares it a class is character-for-character what ``_add_edges`` did when it
    decided to emit the edge.

    Not a name lookup against the store (``resolve_anchor_by_name`` and friends), tempting as that
    is. It resolves strictly *more* names than emission does — an ALTER surface, a name whose slug
    differs — so it would key edges emission never wrote, and where such an edge happens to exist
    because a different document asserted it, this batch's sources would be attached to a claim it
    never made. Matching emission is the requirement; a broader name that resolves to nothing must
    stay unattributed. It is also free of a query per name inside the persist transaction.

    ``iri_by_outcome_id`` maps ``id(outcome)`` to the concept that outcome's claims belong to, and
    it is what decides who may be attributed at all: a concept this batch minted, or the live one
    a candidate was confidently merged into. Anything still awaiting a human — a relabel proposal,
    a gray — is absent from it, because until someone decides, "this candidate is that concept" is
    the very thing in question and its sources would be filed against a merge nobody approved.

    Deliberately not built from ``target_iri``. On a GRAY that field is the anchor the candidate
    was *proposed* against, or the live IRI its slug collided with, so keying on it files this
    batch's sources under another concept's edges.

    A confident attach is in the map even though this batch emitted none of its edges: it merged
    into a concept whose edges the store already holds, and a second document re-asserting an
    existing ``broader`` genuinely asserts it. The key simply matches an edge that is already
    there. This is the case that would otherwise vanish twice over — the claim unrecorded, and
    invisible in ``unattributed_edges`` too, since the edge does have a source from the document
    that first asserted it.
    """

    def resolve_class(name: str) -> str | None:
        cand = URIRef(mint(base_iri, KIND_CLASS, name))
        return str(cand) if (cand, RDF.type, OWL.Class) in graph else None

    claims = (
        (iri_by_outcome_id.get(id(o)), o.representative.kind, m)
        for o in outcomes
        if id(o) in iri_by_outcome_id
        for m in o.members
    )
    return _edge_sources(claims, resolve_class)
