"""Update-mode orchestrator: route a later batch, mutate the graph, gate, persist.

load -> gloss -> embed -> center -> calibrate -> classify -> mutate one graph
(new concepts add axioms; merges are surface-only; gray/relabel enqueue) -> validate
(syntax + HermiT) -> commit snapshot + resolution rows + queue in one transaction.

The batch is all-or-nothing: the mutated graph passes the validation gate once, and a failure
rolls the whole batch back. Deletion never appears here — it is human-only.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

from app.tboxcraft.calibration import calibrate
from app.tboxcraft.cluster import attach_and_cluster
from app.tboxcraft.config import Config
from app.tboxcraft.db.repository import StaleOntologyError
from app.tboxcraft.embed import make_provider
from app.tboxcraft.embed.base import EmbeddingProvider
from app.tboxcraft.emit.tbox import (
    _class_union_node,
    _xsd_ref,
    scan_datatype_substitutions,
    scan_unresolved_broaders,
    _case_variant_map,
    scan_unresolved_domains,
    scan_unresolved_ranges,
    serialize_turtle,
)
from app.tboxcraft.emit.topology import extract_topology
from app.tboxcraft.gloss import build_gloss
from app.tboxcraft.io import load_components, parse_components
from app.tboxcraft.io.components import (
    kind_conflict_reason,
    demotable_kind_conflict_iris,
    fold_case_variants,
    scan_cross_batch_kind_conflicts,
    scan_cross_batch_slug_collisions,
    scan_kind_conflicts,
    scan_non_english_labels,
    scan_overridden_lang_declarations,
    scan_slug_collisions,
    withheld_kind_conflict_uids,
    withheld_slug_collision_uids,
)
from app.tboxcraft.io.cq import resolve_cqs
from app.tboxcraft.models import KIND_CLASS, KIND_DATA_PROPERTY, KIND_OBJECT_PROPERTY
from app.tboxcraft.pipeline.approval import (
    OUTCOME_ADD,
    OUTCOME_GRAY,
    OUTCOME_SYNONYM_ADD,
    OUTCOME_UPDATE,
    approve,
)
from app.tboxcraft.pipeline.generate import GenerateResult, generate
from app.tboxcraft.pipeline.update import DbRetriever, _iri_exists
from app.tboxcraft.pipeline.verdict import flagged_reason, route_by_verdict
from app.tboxcraft.provenance import collect_edge_sources_warm, component_source_ids
from app.tboxcraft.represent import (
    assign_pref_labels,
    build_concept,
    is_hash_namespace,
    mint,
    pref_label_of,
    pref_lang_for,
    representative_label,
)
from app.tboxcraft.similarity import center_query
from app.core.tracing import traced_pipeline
from app.tboxcraft.validate import gate, summarize, validate_tbox

if TYPE_CHECKING:  # annotation only — no runtime import cycle
    from app.tboxcraft.db import Repository

logger = logging.getLogger(__name__)


@dataclass
class UpdateResult:
    ok: bool
    gate_reasons: list[str] = field(default_factory=list)
    # How this batch passed, not just that it did. ok=true says the gate found no reason to
    # refuse; it does not say consistency was proven — a run with the reasoner switched off has
    # `consistent` None and nothing to refuse, so it passes unchecked. A consumer that treats PASS
    # as "verified" needs to be able to tell those apart, and until now the report existed only as
    # a file on the server.
    validation: dict = field(default_factory=dict)
    n_candidates: int = 0
    n_new: int = 0
    n_merged: int = 0
    n_relabel: int = 0
    n_gray: int = 0
    # Of the gray candidates, how many were also minted so their anchors resolve, and how
    # many could not be (their slug already names a live concept). n_gray still counts every
    # one — being minted does not settle it — so these two say what the queue cost.
    n_gray_minted: int = 0
    n_gray_unmintable: int = 0
    # AlterTerm surfaces not attached to a new concept because the string already resolves to a
    # different ACTIVE term (surface uniqueness A). Reported because it is the direct measure of
    # how much individual loss the rule prevented — each would otherwise have made a class name
    # ambiguous and dropped the individuals naming it.
    n_surface_forms_skipped: int = 0
    # Terms this batch minted with no provenance link at all (term_source_map empty). The same
    # invariant generate reports, computed on the update path too: the harness calls only /update,
    # so /generate's counter is never observed in that configuration. 0 is the invariant — a term
    # citing no source, or one whose only CQ was never registered, otherwise lands unattributed and
    # nothing downstream shows it (a dropped component is not even in the DB to count).
    n_terms_without_source: int = 0
    version_id: str | None = None
    ttl: str | None = None
    # Row-level resolution for this batch: one entry per merged/new candidate,
    # raw name -> canonical IRI, all kinds (classes + properties).
    mapping_delta: list[dict] = field(default_factory=list)
    # Candidates routed to human review. A gray one is now minted as well as queued
    # (see mintable_gray), so being here no longer means unresolvable — it means undecided, and
    # a minted gray is in mapping_delta AND here. Coverage is therefore a UNION, not a sum:
    #
    #     {r["candidate_uid"] for r in mapping_delta} | {p["candidate_uid"] for p in pending}
    #     has exactly n_candidates entries.
    #
    # The old ``len(mapping_delta) + len(pending) == n_candidates`` double-counted every minted
    # gray; adding n_gray_unmintable instead undercounts by the relabels and by any multi-member
    # gray, since that field is per outcome, not per member. The union holds in every case.
    pending: list[dict] = field(default_factory=list)
    # Verifier health for this run: how many judged candidates came back usable vs unavailable.
    # A down/misconfigured LLM degrades to Verdict.unavailable per candidate by design (a dead
    # verifier must not block human review), which otherwise leaves an ok=true run with a valid
    # TTL and no sign that verification never happened — only a log line. Surfacing the counts
    # makes that visible to a caller that never reads the logs.
    n_verified: int = 0
    n_verifier_unavailable: int = 0
    # What the verification cost. Only a cold /update fills these — it bootstraps through
    # generate(), which is where candidates are batched; the warm path judges one candidate per
    # decision and leaves them at 0.
    n_clusters: int = 0
    n_verify_calls: int = 0
    # [{kind, tau_high, tau_low, source, n_positives}] — the merge threshold in force per kind.
    # A cold /update bootstraps through generate, which is the path this was invisible on.
    calibration: list[dict] = field(default_factory=list)
    # Merges refused because the two sides were not similar to each other. A threshold that is
    # too low shows up here before it shows up as a wrong concept count.
    n_chained_refused: int = 0
    # [{surface, iris, sources}] — strings naming more than one concept; neither resolves
    # reliably, and the individual that goes missing does so a stage later.
    surface_collisions: list[dict] = field(default_factory=list)
    # [{name, kinds, sides}] — one name minted as two kinds; under a slash namespace both stay
    # and the pair is queued, because which reading is right is a modelling call.
    kind_conflicts: list[dict] = field(default_factory=list)
    queued_kind_conflicts: list[dict] = field(default_factory=list)
    # Under a HASH namespace the two readings would collapse onto one IRI, which OWL 2 DL forbids
    # (object and data properties are disjoint), so neither side is minted. Counted and listed
    # rather than only counted: the whole point is to hand upstream "these names, these two
    # readings, these ranges — fix them", and a bare number cannot be acted on.
    # Counts the contested NAMES, so it matches len(withheld_kind_conflicts) the way
    # every other n_* in this result matches its list. The number of TERMS not minted
    # is a different question — one name withholds both readings — and gets its own
    # field rather than overloading this one, because a gate asserting on "how many
    # terms are missing from the TBox" and one asserting on "how many names are
    # contested" want different numbers.
    n_kind_conflicts_withheld: int = 0
    n_terms_withheld: int = 0
    withheld_kind_conflicts: list[dict] = field(default_factory=list)
    # Slug collisions — see GenerateResult. Both halves are counted here (the cold path has no
    # cross-batch half), and the cross-batch one is the reason this exists on the warm path at all:
    # an unverifiable store hit is read as a re-ingest and merges two names onto one IRI in silence.
    # [{uid, kind, name, demoted_term, demoted_lang, label, label_lang, label_is_english}] —
    # curated terms that could not be the representative label (it is one per term and English),
    # so they stayed on the term as alternative surfaces and the label fell back to the de-split
    # name. Not an error and not a stop: a producer sending Korean official terms is doing its
    # job. Listed rather than only counted, because
    # "the curator's term is not this term's displayed name" is a fact somebody has to be able to
    # see — and because `label_is_english` False marks the rarer, worse shape where the fallback
    # is not English either and the label had to be tagged `und`.
    non_english_labels: list[dict] = field(default_factory=list)
    # [{uid, kind, name, surface, source, declared, written}] — surfaces whose `en` declaration
    # was refused because the string has no Latin letter, and the count of them. Writing the
    # declaration down would make this pipeline the laundering step for the same false `@en` claim;
    # refusing it silently would hide whether the producer ever stopped sending it.
    n_lang_declarations_overridden: int = 0
    overridden_lang_declarations: list[dict] = field(default_factory=list)
    n_slug_collisions_withheld: int = 0
    n_slug_terms_withheld: int = 0
    withheld_slug_collisions: list[dict] = field(default_factory=list)
    # Terms an EARLIER batch minted that this one moved ACTIVE -> DRAFT, because this batch sends
    # the same name under the other kind. A write to previously persisted rows, so it is reported
    # as its own count with the rows that moved — not folded into the withheld list above, which
    # is about terms that were never created.
    n_demoted_to_draft: int = 0
    demoted_to_draft: list[dict] = field(default_factory=list)
    # [{kind, label, kept, relabelled, fallback, rule}] and which of them reached the merge queue.
    # See GenerateResult. Filled on a cold /update, which routes inside generate() — carried over
    # rather than left at 0, because a default 0 cannot be told from a measured one.
    pref_label_collisions: list[dict] = field(default_factory=list)
    queued_pref_label_collisions: list[dict] = field(default_factory=list)
    # [{surface, kind, alias_owner, claimed_by, form_type, rows}] — aliases removed from a term
    # that already held the string, because this batch minted a concept whose representative label
    # is that string. PREF outranks a synonym, and the string keeps resolving (to the term it names)
    # instead of resolving to neither, which is what two ACTIVE claimants means for the loader.
    # Reported because it edits a term this batch did not create — usually one from an earlier
    # document, but the check reads the open transaction, so it also covers a concept minted moments
    # earlier in this same batch (whichever of the two is written first, the alias loses).
    demoted_store_aliases: list[dict] = field(default_factory=list)
    # Which of those are in the queue — [{surface, iris, review_id, already_queued}]. See
    # GenerateResult. On this path every detected store collision is queued, so the two lists have
    # the same length once already_queued ones are counted in.
    queued_surface_collisions: list[dict] = field(default_factory=list)
    # [{kind, kept, folded, ...}] — same-kind names differing only in case, merged before
    # minting. Reported because it decides an IRI.
    folded_names: list[dict] = field(default_factory=list)
    # [{a, b, kind, reason}] — curated 'same concept' claims the property gate refused;
    # both sides minted and the pair queued for review.
    refused_duplicates: list[dict] = field(default_factory=list)
    # Surfaces written as HiddenTerms rather than standard synonyms.
    n_hidden_surfaces: int = 0
    # --- what this run silently dropped or altered ---------------------------------------------
    # Reported, not raised: none of these should fail an ingest, but a caller seeing ok=true and a
    # valid TTL otherwise has no way to tell the output is incomplete. Each was previously only a
    # log line, and a missing CQ link in particular only surfaces one stage later, as a foreign-key
    # violation in the ABox loader.
    n_cq_linked: int = 0
    n_cq_missing: int = 0
    n_hierarchy_skipped: int = 0
    # [{child, parent}] — see GenerateResult.
    hierarchy_cycles_broken: list[dict] = field(default_factory=list)
    n_domain_range_skipped: int = 0
    # Stored edges no source asserted, and links naming a source the registry does not hold. Same
    # pair GenerateResult reports, and needed here more than there: replace_topology rewrites the
    # whole topology every warm run, so the edge->source maps are rebuilt from scratch each time and
    # "nobody collected this" would otherwise be indistinguishable from a clean load.
    # n_unattributed_edges is expected to be non-zero (a merge-only edge has no asserting source,
    # and provenance attributes only what this batch emitted); n_edge_sources_skipped at 0 is the
    # one worth pinning.
    n_unattributed_edges: int = 0
    n_edge_sources_skipped: int = 0
    # [{name, requested, emitted}] — data-property ranges outside the OWL 2 datatype map.
    datatype_substitutions: list[dict] = field(default_factory=list)
    # [{name, broader}] — subClassOf edges dropped because the parent is not defined in the batch.
    unresolved_broaders: list[dict] = field(default_factory=list)
    # [{name, unresolved, resolved}] — object-property range names that do not name a class, so
    # the emitted rdfs:range is missing (resolved empty) or NARROWER than the input asked for
    # (resolved non-empty). The second shape is why the resolved names are carried: from the TTL
    # alone a union that lost a member is indistinguishable from a property that only ever had one
    # range, and a count cannot say which happened.
    unresolved_ranges: list[dict] = field(default_factory=list)
    # The domain counterpart of ranges: makes visible the loss where multi-value domains
    # were silently dropped.
    unresolved_domains: list[dict] = field(default_factory=list)
    # Upstream TBox/ABox verdict tally. All three are minted (see route_by_verdict for why an
    # 'abox' component is still minted); the flagged ones are queued for review. Without these the
    # classifier could degrade badly and the pipeline would look identical.
    n_tbox: int = 0
    n_conflict: int = 0
    n_abox: int = 0
    n_verdict_queued: int = 0
    # Review rows this run actually inserted, from every source: verdict routing, refused
    # duplicates, surface collisions, and the GRAY/RELABEL outcomes. n_gray and n_relabel count
    # outcomes while the queue takes one row per member, and n_verdict_queued covers only the
    # verdict sources — so no combination of the existing numbers said how much the queue grew.
    # Counted from what enqueue_review returned, so an idempotent re-enqueue (which returns None)
    # is not counted as growth.
    n_review_rows: int = 0
    # Attaches decided by an exact name match rather than by cosine. Reported because it is the
    # measure of whether a second document merges into the first at all: it was 0 by construction
    # before the rule existed, and a run where it stays 0 while gray climbs is the same failure.
    n_attached_by_name: int = 0
    # The gray/new boundary, kept checkable after the run. tau_low is a per-pair quantile compared
    # against a maximum over the whole store, so it drifts as the store grows — the gray share
    # climbed steeply over a few documents with the threshold unchanged. These say where the
    # boundary actually fell, so the same drift cannot recur unseen.
    nearest_score_histogram: dict = field(default_factory=dict)
    n_attached_by_margin: int = 0
    n_new_background: int = 0
    # Every gray outcome by why it is gray, e.g. {"low-confidence attach": n,
    # "slug collides with existing IRI": m, ...}. The attach-band counters above cover only the
    # first of four sources — a merge-band attach the property gate rejects, a new cluster whose
    # slug collides, and one that is not cohesive are all gray too, and all three grow with the
    # store. Keyed on the reason string the outcome already carries, so a source added later
    # appears here without anyone remembering to count it; that unexplained remainder is exactly
    # what went unnoticed while it grew document after document.
    gray_reasons: dict = field(default_factory=dict)

    @property
    def verifier_degraded(self) -> bool:
        """True when the verifier was asked but never answered — verification was effectively off."""
        return self.n_verifier_unavailable > 0 and self.n_verified == 0

    @property
    def cq_provenance_degraded(self) -> bool:
        """True when CQ links were requested but none resolved — every term lost its provenance."""
        return self.n_cq_missing > 0 and self.n_cq_linked == 0


def _declare_concept(graph: Graph, component, base_iri: str, labels=None) -> URIRef:
    """Pass 1: add the type triple + labels (so cross-references resolve in pass 2)."""
    iri = URIRef(mint(base_iri, component.kind, component.name))
    rdf_type = {
        KIND_CLASS: OWL.Class,
        KIND_OBJECT_PROPERTY: OWL.ObjectProperty,
        KIND_DATA_PROPERTY: OWL.DatatypeProperty,
    }[component.kind]
    graph.add((iri, RDF.type, rdf_type))
    for lang, label in sorted(_pref_labels_for(component, labels).items()):
        graph.add((iri, SKOS.prefLabel, Literal(label, lang=lang)))
        graph.add((iri, RDFS.label, Literal(label, lang=lang)))
    if component.dom_notes:
        graph.add(
            (
                iri,
                RDFS.comment,
                Literal(component.dom_notes, lang=component.comment_lang),
            )
        )
    return iri


def _pref_labels_for(component, labels) -> dict[str, str]:
    """``{lang: label}`` for one component — the settled labels when the contest moved any, else
    the component's own.

    The single joint between :func:`resolve_pref_labels` and the two places that write a label
    (this graph builder and :func:`_mint_concept`). It exists so the warm path cannot end up
    writing one label into the TTL and a different one into ``surface_form``, which is the shape
    the cold path's ``dedupe_pref_labels`` avoids by mutating the concept in place — the warm path
    has no Concept object to mutate, so the fallback has to be stated once instead of twice.
    """
    return (labels or {}).get(component.uid) or dict([pref_label_of(component)])


def class_resolver(graph: Graph, base_iri: str):
    """``name -> URIRef | None`` against the accumulated graph — the warm path's name resolution.

    Module-level and shared rather than a closure inside the one caller, because whether a name
    resolves is exactly what decides if an edge is emitted, and the scan that reports the names
    that did NOT resolve has to ask the identical question. Two copies of this rule would drift
    and the report would describe a graph nobody built (the mistake `class_iri_by_name` exists to
    prevent on the cold path).
    """

    def resolve(name: str) -> URIRef | None:
        cand = URIRef(mint(base_iri, KIND_CLASS, name))
        return cand if (cand, RDF.type, OWL.Class) in graph else None

    return resolve


def _add_edges(graph: Graph, component, base_iri: str) -> None:
    """Pass 2: hierarchy / domain / range, resolving names against the whole graph."""
    iri = URIRef(mint(base_iri, component.kind, component.name))
    resolve_class = class_resolver(graph, base_iri)

    if component.kind == KIND_CLASS:
        for broader in component.authoritative_hierarchy().broader:
            b_iri = resolve_class(broader)
            if b_iri is not None and b_iri != iri:
                graph.add((iri, RDFS.subClassOf, b_iri))
        return

    # Domains are comma-separated multi-values like ranges. Comparing them whole (the old
    # way) silently drops multi-values, so this working graph, which the gates and the
    # reasoner see, and the output graph (emit/tbox) would use different domain rules.
    # Assembly goes through emit's single _class_union_node (so they cannot diverge).
    domain_iris = [
        d for d in (resolve_class(x) for x in component.domain_values()) if d is not None
    ]
    if len(domain_iris) == 1:
        graph.add((iri, RDFS.domain, domain_iris[0]))
    elif len(domain_iris) > 1:
        graph.add((iri, RDFS.domain, _class_union_node(graph, domain_iris)))

    ranges = component.range_values()
    if component.kind == KIND_DATA_PROPERTY:
        if ranges:
            graph.add((iri, RDFS.range, _xsd_ref(ranges[0])))
        return
    range_iris = [r for r in (resolve_class(x) for x in ranges) if r is not None]
    if len(range_iris) == 1:
        graph.add((iri, RDFS.range, range_iris[0]))
    elif len(range_iris) > 1:
        graph.add((iri, RDFS.range, _class_union_node(graph, range_iris)))


def _add_new_concepts(
    graph: Graph, new_components: list, base_iri: str, labels=None
) -> None:
    """Add axioms for the batch's new concepts in two passes so intra-batch references resolve.

    ``labels`` overrides a component's representative label where resolve_pref_labels had to move
    it off a string another concept owns. Threaded rather than applied afterwards because this is
    what writes skos:prefLabel into the graph the gate validates.
    """
    for c in new_components:
        _declare_concept(graph, c, base_iri, labels)
    for c in new_components:
        _add_edges(graph, c, base_iri)


def _surface_entries(component) -> list[tuple[str, str, str]]:
    """``(surface, lang, source)`` rows for a concept, straight from the shared collector.

    Kept as a named indirection rather than inlined because ``review.py`` writes surfaces on the
    apply path and this is the sentence that says the two agree. It used to add ``"en"`` to every
    row here, which is how Korean surfaces from a non-English producer came to be tagged
    as English; the language now belongs to the collector, per channel.
    """
    return component.surface_candidates()


def _component_payload(component, gloss: str) -> dict:
    """Enough of a component to apply a queued decision later without the batch file."""
    return {
        "name": component.name,
        "kind": component.kind,
        "correct_term": component.correct_term,
        "domain": component.domain,
        "range": component.range_raw,
        "dom_notes": component.dom_notes,
        # The languages travel with the surfaces for the same reason the surfaces do: a human
        # merging this candidate later rebuilds the component from the payload alone. Without
        # them the rebuild silently re-answers "we were not told" for a component whose producer
        # did tell us, and the applied term lands with `und` on strings that arrived tagged.
        "correct_term_en": component.correct_term_en,
        "lang": component.lang,
        "notes_lang": component.notes_lang,
        # Alternative surfaces must survive the queue: a human merging this candidate later
        # rebuilds the component from the payload alone, and without these they are lost.
        "duplicates": list(component.dom_duplicates),
        "synonyms": list(component.dom_synonyms),
        "broader": list(component.authoritative_hierarchy().broader),
        "source_cq_ids": list(component.source_cq_ids),
        "gloss": gloss,
    }


@traced_pipeline("update")
def update(
    components_path,
    config: Config,
    *,
    repo,
    provider: EmbeddingProvider | None = None,
    llm=None,
    verify: bool = True,
    run_reasoner: bool = True,
    run_oops: bool = False,
    source_uri: str | None = None,
    components_doc: dict | None = None,
    sources: dict | None = None,
) -> UpdateResult:
    """Route a later batch against the accumulated ontology."""
    provider = provider or make_provider(config)
    model_name = getattr(provider, "model_name", provider.name)
    revision = getattr(provider, "revision", "unknown")

    # Components may arrive inline (service call) or from a file; CQs come from the DB.
    if components_doc is not None:
        components = parse_components(components_doc)
    else:
        components = load_components(components_path)
    # Same fold generate does, on the warm path too: two spellings of one name would
    # otherwise attach or mint separately and leave the string ambiguous again. The cold
    # path bootstraps through generate and gets its folded_names from there.
    components, folded_names = fold_case_variants(components)
    if folded_names:
        logger.warning(
            "update: folded %d name(s) differing only in case — %s",
            len(folded_names),
            "; ".join(f"{r['folded']} -> {r['kept']}" for r in folded_names[:5]),
        )
    # Same scan the bootstrap runs. It was added to generate only, so a warm run detected none
    # of these: after the first document every later batch's kind conflicts went unqueued.
    #
    # Batch-local by construction — it compares the components in hand. A candidate that
    # conflicts with a term minted by an *earlier* batch is not seen here; that needs a store
    # lookup and the stored side's range, and is not attempted yet.
    kind_conflicts = scan_kind_conflicts(components)
    if kind_conflicts:
        logger.warning(
            "update: %d name(s) judged as two kinds within this batch — both minted, "
            "queued for review: %s",
            len(kind_conflicts),
            "; ".join(
                f"{k['name']}({'/'.join(k['kinds'])})" for k in kind_conflicts[:5]
            ),
        )
    # Read the upstream TBox/ABox verdict. Nothing is filtered out — see route_by_verdict.
    flagged_verdicts, verdict_tally = route_by_verdict(components, stage="update")
    cqs = resolve_cqs(repo)
    logger.info(
        "update: received %d components (%s)",
        len(components),
        "inline" if components_doc is not None else "file",
    )

    latest = repo.get_latest_ontology(config.base_iri)
    if latest is None:
        # Cold start: no ontology yet, so the first batch must be *generated*. /update is
        # the single ingest entry point — bootstrap transparently instead of erroring.
        logger.info("update: no ontology present -> bootstrapping via generate")
        gr = generate(
            components_path,
            config,
            provider=provider,
            run_reasoner=run_reasoner,
            run_oops=run_oops,
            persist_db=True,
            repo=repo,
            source_uri=source_uri,
            # Adapter-declared sources have to travel with the batch. They reach _persist_update
            # on the warm path, and the cold branch does not go through it — so without this the
            # bootstrap registered none of them: no source rows for all the adapter sent, and the
            # failure surfaced two stages later as an abox_individual.source_id FK violation that
            # names aboxmapper. An RDB store always starts empty, so for that path it was every
            # run, not an edge case.
            components_doc=components_doc,
            sources=sources,
        )
        return _bootstrap_result(gr, repo, config.base_iri)

    # Warm path only: a name this batch judges as one kind may already be minted under another by
    # an earlier document. The in-batch scan above cannot see that side, and the individuals do
    # not go missing (each kind has its own anchor) — but the contradiction is never reviewed,
    # while the in-batch case is queued loudly. Look up the stored side and queue it the same way.
    stored_by_name = repo.stored_terms_by_local_name(
        [c.name for c in components], config.base_iri
    )
    cross = scan_cross_batch_kind_conflicts(components, stored_by_name)
    if cross:
        logger.warning(
            "update: %d name(s) already minted under a different kind by an earlier batch — "
            "queued for review: %s",
            len(cross),
            "; ".join(f"{k['name']}({'/'.join(k['kinds'])})" for k in cross[:5]),
        )
    kind_conflicts = kind_conflicts + cross

    # Kind conflicts. Only under a hash namespace: there the two readings collapse onto one IRI and
    # OWL 2 DL refuses it, so minting is not an option. A slash namespace still has the kind
    # segment, the two IRIs stay distinct and legal, and the "mint both, queue the pair" rule
    # remains correct — withholding there would take away terms nothing is wrong with.
    withheld_conflicts: list[dict] = []
    withheld_components: list = []
    n_withheld = n_terms = 0
    demoted_rows: list[dict] = []
    n_demoted = 0
    if is_hash_namespace(config.base_iri) and kind_conflicts:
        withheld_uids = withheld_kind_conflict_uids(kind_conflicts)
        demotable = demotable_kind_conflict_iris(kind_conflicts)
        # The stored side of a cross-batch conflict cannot be "not minted" — it already exists —
        # so it is demoted instead. Done before the batch is routed so the rest of this run sees
        # the same store a fresh run would: a demoted term is not an attach anchor.
        if demotable:
            moved = repo.demote_to_draft([d["iri"] for d in demotable])
            n_demoted, demoted_rows = moved, demotable
            logger.warning(
                "update: %d term(s) demoted ACTIVE -> DRAFT — this batch sends the same name under "
                "the other kind, and a hash namespace has one IRI for both: %s",
                moved,
                "; ".join(f"{d['name']}({d['kind']})" for d in demotable[:5]),
            )
        if withheld_uids:
            # Removed from minting and routing only: the objects are restored below as
            # queueing material (vectors, glosses) (same reason as the cold path: deleted,
            # withheld conflicts could not reach the queue).
            withheld_components = [c for c in components if c.uid in withheld_uids]
            components = [c for c in components if c.uid not in withheld_uids]
            n_withheld, n_terms, withheld_conflicts = (
                len(kind_conflicts),
                len(withheld_uids),
                kind_conflicts,
            )
            logger.warning(
                "update: %d component(s) withheld — a hash namespace cannot separate an object "
                "property from a data property of the same name, so neither side is minted and "
                "both stay queued: %s",
                len(withheld_uids),
                "; ".join(
                    f"{k['name']}({'/'.join(k['kinds'])})" for k in kind_conflicts[:5]
                ),
            )

    # The representative label is one per term and English; a curated term that cannot be goes to
    # an alternative surface instead of being translated or asserted. Reported, never fatal.
    # A producer's `en` over a string with no Latin letter is refused, not written (see
    # Component.declared_lang_for). Counted so "they fixed it" and "we are still covering" differ.
    overridden_langs = scan_overridden_lang_declarations(components)
    if overridden_langs:
        logger.warning(
            "update: %d surface(s) declared `en` by their producer carry no Latin letter — written "
            "as und instead; the declaration is upstream's to fix: %s",
            len(overridden_langs),
            "; ".join(
                f"{r['name']}/{r['source']} {r['surface']!r}"
                for r in overridden_langs[:5]
            ),
        )

    non_english_labels = scan_non_english_labels(components)
    if non_english_labels:
        _no_english = [r for r in non_english_labels if not r["label_is_english"]]
        logger.info(
            "update: %d curated term(s) kept as an alternative surface — the representative "
            "label is English and these are not, so the label came from the name instead%s: %s",
            len(non_english_labels),
            f" ({len(_no_english)} had no English name either, tagged und)"
            if _no_english
            else "",
            "; ".join(
                f"{r['name']}: {r['demoted_term']!r} -> {r['label']!r}"
                for r in non_english_labels[:5]
            ),
        )

    # Slug collisions. Both halves here, unlike the cold path: a warm batch has a store to collide
    # with, and the cross-batch half is the one that actually corrupts — an unverifiable hit reads
    # as a re-ingest and `upsert_concept` merges the new name onto the old term's IRI.
    slug_collisions = scan_slug_collisions(
        components
    ) + scan_cross_batch_slug_collisions(components, stored_by_name)
    n_slug_terms_withheld = 0
    if slug_collisions:
        withheld_slugs = withheld_slug_collision_uids(slug_collisions)
        components = [c for c in components if c.uid not in withheld_slugs]
        n_slug_terms_withheld = len(withheld_slugs)
        logger.warning(
            "update: %d component(s) withheld — %d IRI(s) would name more than one term because "
            "the slug is not reversible, and an identity cannot be un-merged later: %s",
            n_slug_terms_withheld,
            len(slug_collisions),
            "; ".join(
                f"{'/'.join(c['names'])} -> {c['slug']}" for c in slug_collisions[:5]
            ),
        )

    model_id = repo.get_model_id(model_name, revision)
    if model_id is None:
        # An ontology exists but this provider's model isn't registered — a config mismatch,
        # NOT a cold start. Do not silently re-bootstrap over the existing ontology.
        logger.warning(
            "update: embedding model %s@%s not registered for the existing ontology",
            model_name,
            revision,
        )
        return UpdateResult(
            ok=False,
            gate_reasons=[
                (
                    "embedding model not registered for the existing ontology; "
                    "check TBOXCRAFT_EMBEDDING_* matches the bootstrap model"
                )
            ],
        )

    base_id, ttl = latest
    graph = Graph()
    graph.parse(data=ttl, format="turtle")
    logger.info(
        "update: loaded ontology snapshot v%s (%d triples)", base_id, len(graph)
    )

    ordered = sorted(components, key=lambda c: c.uid)
    glosses = {
        c.uid: build_gloss(c, cqs, max_cq_sample=config.max_cq_sample) for c in ordered
    }

    logger.info(
        "update: embedding %d candidate glosses via %s...", len(ordered), provider.name
    )
    matrix = provider.embed([glosses[c.uid] for c in ordered])
    mean = repo.get_model_mean(model_name, revision)
    # Center once into the stored centered space; reuse for both calibration and routing,
    # so the retriever must not re-center (mean=None).
    centered = {c.uid: center_query(matrix[i], mean) for i, c in enumerate(ordered)}
    # Restore the withheld vectors and glosses (excluded from minting ≠ evidence deleted).
    # They are centered separately with the stored mean, so the main batch's vectors and
    # calibration are unaffected; they serve only as material the kind_conflict queue row
    # freezes.
    if withheld_components:
        withheld_ordered = sorted(withheld_components, key=lambda c: c.uid)
        for comp in withheld_ordered:
            glosses[comp.uid] = build_gloss(
                comp, cqs, max_cq_sample=config.max_cq_sample
            )
        withheld_matrix = provider.embed([glosses[c.uid] for c in withheld_ordered])
        for comp, row in zip(withheld_ordered, withheld_matrix):
            centered[comp.uid] = center_query(row, mean)

    thresholds = calibrate(
        components,
        centered,
        provider.name,
        seed_high=config.tau_high_seed,
        seed_low=config.tau_low_seed,
    )
    retriever = DbRetriever(
        repo, model_id, mean=None, base_iri=config.base_iri, k=config.gray_zone_k
    )
    persist_vectors = provider.dimension == 1024
    components_by_uid = {c.uid: c for c in ordered}

    # Attach-only clustering: existing concepts are frozen anchors; new components attach
    # or self-cluster. Then DomainTermApproval judges each cluster — compute-only, no DB
    # writes; enqueue happens later inside the post-gate transaction.
    plan = attach_and_cluster(
        ordered, centered, retriever, thresholds, attach_margin=config.attach_margin
    )
    n_by_name = sum(1 for a in plan.attached if a.matched_on == "name")
    if n_by_name:
        logger.info(
            "update: %d of %d attaches were decided by an exact name match, not by cosine",
            n_by_name,
            len(plan.attached),
        )
    outcomes = approve(
        plan,
        components_by_uid,
        centered,
        graph,
        config,
        thresholds,
        llm,
        verify=verify,
        glosses=glosses,
    )

    tally = Counter(o.kind for o in outcomes)
    logger.info(
        "update: routed %d candidates -> add=%d synonym=%d update=%d gray=%d",
        len(ordered),
        tally.get(OUTCOME_ADD, 0),
        tally.get(OUTCOME_SYNONYM_ADD, 0),
        tally.get(OUTCOME_UPDATE, 0),
        tally.get("gray", 0),
    )

    # Verifier health: an unreachable/misconfigured LLM yields Verdict.unavailable per candidate
    # and the run still succeeds, so count it and report it out rather than only logging.
    judged = [o.verdict for o in outcomes if o.verdict is not None]
    n_unavailable = sum(1 for v in judged if v.status == "unavailable")
    n_verified = len(judged) - n_unavailable
    if verify and judged and n_verified == 0:
        logger.error(
            "update: the verifier answered for NONE of %d judged candidates — verification was "
            "effectively disabled for this run (check TBOXCRAFT_LLM_URL / model name)",
            len(judged),
        )

    # TboxMatcher: ADD clusters mutate the graph — one IRI per cluster minted from the
    # representative (non-representatives become surface forms in persist); SynonymADD is
    # surface-only; UPDATE/GRAY enqueue only.
    add_reps = [o.representative for o in outcomes if o.kind == OUTCOME_ADD]
    # A GRAY candidate is minted too, when it can be. Leaving it out of the store is what
    # `pending` used to mean, and it costs every individual naming it its anchor — dropped a
    # stage later as a `miss`, with no record of which ones. That is the trade
    # `route_by_verdict` already refused on the bootstrap path: a duplicate concept is visible
    # and a reviewer can merge it, a missing anchor is neither visible nor recoverable.
    #
    # "When it can be" is the exception that made some of these gray in the first place: a
    # candidate whose slug already names a live concept cannot be minted without clobbering a
    # stable IRI. Those stay pending-only, and `n_gray_unmintable` says how many.
    gray_mints, n_gray_unmintable = mintable_gray(
        outcomes,
        graph,
        config.base_iri,
        claimed={
            o.target_iri for o in outcomes if o.kind == OUTCOME_ADD and o.target_iri
        },
    )
    minted_reps = add_reps + [o.representative for o in outcomes if id(o) in gray_mints]
    # Before the graph: the label lands in the TTL the gate and HermiT read, so a decision taken at
    # the DB write would leave the validated snapshot and the store disagreeing about the concept's
    # name. Same reason dedupe_pref_labels runs where it does on the cold path.
    iri_of = {
        id(o): o.target_iri for o in outcomes if o.kind == OUTCOME_ADD and o.target_iri
    }
    iri_of.update(gray_mints or {})
    pref_labels, pref_label_collisions = resolve_pref_labels(
        [(o.representative, iri_of[id(o)]) for o in outcomes if id(o) in iri_of],
        repo,
        base_iri=config.base_iri,
    )
    if pref_label_collisions:
        logger.warning(
            "update: %d representative label(s) already claimed — relabelled to the concept's own "
            "de-split name: %s",
            len(pref_label_collisions),
            "; ".join(
                f"{r['label']!r} kept by {r['kept'].rsplit('/', 1)[-1]} "
                f"({r['rule']}), {r['relabelled'].rsplit('/', 1)[-1]} -> {r['fallback']!r}"
                for r in pref_label_collisions[:5]
            ),
        )
    _add_new_concepts(graph, minted_reps, config.base_iri, pref_labels)
    if gray_mints:
        logger.info(
            "update: minting %d of %d gray candidate(s) so their anchors resolve; every one "
            "stays queued for review",
            len(gray_mints),
            tally.get(OUTCOME_GRAY, 0),
        )

    logger.info(
        "update: validating mutated TBox (reasoner=%s, oops=%s)...",
        run_reasoner,
        run_oops,
    )
    report = validate_tbox(
        graph=graph,
        run_reasoner=run_reasoner,
        run_oops=run_oops,
        java_exe=config.java_exe,
        oops_endpoint=config.oops_endpoint,
    )
    _warm_resolve = class_resolver(graph, config.base_iri)
    passed, reasons = gate(report)

    result = UpdateResult(
        ok=passed,
        gate_reasons=reasons,
        n_candidates=len(ordered),
        validation=summarize(report),
        n_verified=n_verified,
        n_verifier_unavailable=n_unavailable,
        n_gray_unmintable=n_gray_unmintable,
        folded_names=folded_names,
        # Kind conflicts, decided before routing (a withheld component never reached
        # `ordered`, and the demotion already happened against the store).
        n_kind_conflicts_withheld=n_withheld,
        n_terms_withheld=n_terms,
        withheld_kind_conflicts=withheld_conflicts,
        n_slug_collisions_withheld=len(slug_collisions),
        n_slug_terms_withheld=n_slug_terms_withheld,
        withheld_slug_collisions=slug_collisions,
        non_english_labels=non_english_labels,
        n_lang_declarations_overridden=len(overridden_langs),
        overridden_lang_declarations=overridden_langs,
        n_demoted_to_draft=n_demoted,
        demoted_to_draft=demoted_rows,
        # Ranges outside the OWL 2 datatype map were substituted so HermiT could
        # run; the emitted range no longer says what the source asked for.
        datatype_substitutions=scan_datatype_substitutions(ordered),
        unresolved_broaders=scan_unresolved_broaders(ordered),
        # Resolved against the mutated graph — the same rule
        # _add_edges emitted with, via class_resolver.
        unresolved_ranges=scan_unresolved_ranges(
            ordered,
            lambda n: _warm_resolve(n) is not None,
            case_variants=_case_variant_map(
                c.name for c in ordered if c.kind == KIND_CLASS
            ),
        ),
        unresolved_domains=scan_unresolved_domains(
            ordered,
            lambda n: _warm_resolve(n) is not None,
            case_variants=_case_variant_map(
                c.name for c in ordered if c.kind == KIND_CLASS
            ),
        ),
        n_tbox=verdict_tally["tbox"],
        n_conflict=verdict_tally["conflict"],
        n_abox=verdict_tally["abox"],
        n_attached_by_name=n_by_name,
        nearest_score_histogram=plan.nearest_score_histogram,
        n_attached_by_margin=plan.n_attached_by_margin,
        n_new_background=plan.n_new_background,
        gray_reasons=dict(
            sorted(
                Counter(o.reason for o in outcomes if o.kind == OUTCOME_GRAY).items()
            )
        ),
    )
    if not passed:
        # The kind-conflict demotion is the one write that already landed: it runs before routing so
        # the rest of this run sees the store a fresh run would, which puts it ~250 lines and one
        # HermiT pass ahead of this decision. Its queue row does not — enqueueing is on the persist
        # path below, which a failed gate never reaches. Left alone, a rejected batch retires an
        # earlier ingest's terms with nothing recorded to say why: invisible to every consumer (all
        # of them filter status='ACTIVE') and waiting on a decision nobody was asked to make. Undone
        # here so the two halves agree — either the demotion and its review row, or neither.
        if demoted_rows:
            restored = repo.restore_draft_to_active([d["iri"] for d in demoted_rows])
            logger.warning(
                "update: gate FAILED -> restored %d term(s) this run had demoted to DRAFT; "
                "the kind conflict is real but goes unqueued with the batch: %s",
                restored,
                "; ".join(f"{d['name']}({d['kind']})" for d in demoted_rows[:5]),
            )
            result.n_demoted_to_draft = 0
            result.demoted_to_draft = []
        logger.warning(
            "update: gate FAILED -> rolling back, nothing committed: %s",
            "; ".join(reasons),
        )
        return result  # nothing commits on a gate failure
    logger.info("update: validation gate PASSED")

    new_ttl = serialize_turtle(graph)
    try:
        version_id = _persist_update(
            repo,
            outcomes,
            new_ttl,
            base_id,
            model_id,
            persist_vectors,
            centered,
            glosses,
            cqs,
            result,
            graph,
            flagged_verdicts,
            kind_conflicts,
            gray_mints,
            config.base_iri,
            sources=sources,
            pref_labels=pref_labels,
            pref_label_collisions=pref_label_collisions,
        )
        if result.cq_provenance_degraded:
            logger.error(
                "update: NONE of %d requested CQ links resolved — terms were stored without "
                "provenance, and an ABox insert citing those CQ ids will fail on the foreign key",
                result.n_cq_missing,
            )
        elif result.n_cq_missing:
            logger.warning(
                "update: %d CQ links dropped (CQ not in the store)", result.n_cq_missing
            )
    except StaleOntologyError as exc:
        logger.warning("update: concurrent-update conflict, rolled back: %s", exc)
        return UpdateResult(
            ok=False,
            n_candidates=len(ordered),
            gate_reasons=[f"concurrent update conflict: {exc}"],
        )

    result.version_id = version_id
    result.ttl = new_ttl
    logger.info(
        "update: committed snapshot v%s (new=%d merged=%d gray=%d relabel=%d)",
        version_id,
        result.n_new,
        result.n_merged,
        result.n_gray,
        result.n_relabel,
    )

    # Dangling-orphan safety net: scan the committed graph and enqueue human-gated delete
    # proposals for any concept left referencing a missing target. Post-commit and best-effort —
    # a scan failure must never roll back a committed batch.
    try:
        from app.tboxcraft.orphan import enqueue_orphans

        # The return value is the count of fresh proposals — dropping it made n_review_rows
        # under-report by exactly the orphans a run found, which is the shape of miscount this
        # counter exists to prevent.
        result.n_review_rows += enqueue_orphans(repo, graph, config.base_iri)
    except Exception as exc:  # noqa: BLE001 - the orphan scan never fails a committed update
        logger.warning("update: orphan scan failed (ignored): %s", exc)

    return result


def resolve_pref_labels(
    candidates, repo: Repository, *, base_iri: str, conn=None
) -> tuple[dict[str, dict[str, str]], list[dict]]:
    """Settle every representative label this batch is about to write, before the graph is built.

    ``candidates`` are ``(component, iri)`` for the concepts about to be minted. Returns
    ``({uid: {lang: label}}, rows)``: the labels each one should carry — the same values unless it
    had to give way — and one report row per contest.

    The assignment is :func:`~tboxcraft.represent.assign_pref_labels`, shared with the cold path
    (``dedupe_pref_labels``) so the same contest resolves the same way whichever path a batch took.
    What this adds is the store: a probe that reports the live terms already holding a string as
    PREF, which makes them immovable claimants. Inside the batch nothing is written yet and the
    tiered rule decides on merit; against the store the batch gives way even when the tier says it
    should have won, because taking the label renames a concept that already exists and an IRI
    renewal is a human-gated event (``review.py``) rather than something an ingest does in passing.
    Those rows carry ``tier_favoured_batch``. In practice the tier usually agrees with deferring,
    which is the data's shape rather than luck: documents spell a concept out early and
    abbreviate later, so the expansion is minted first.

    Only PREF claimants are consulted. A stored *alias* on the same string is settled the other
    way, at write time, by dropping the alias — PREF outranks a synonym, and that needs no label
    change and so no pre-graph decision.
    """
    seen: dict[tuple[str, str], list[dict]] = {}

    def stored_prefs(kind, label):
        # Cached per (kind, folded): a ladder re-probes a string each time a claim reaches it, and
        # the store cannot change inside this transaction.
        key = (kind, label.casefold())
        if key not in seen:
            seen[key] = [
                s
                for s in repo.surface_claimants(
                    label, kind, base_iri=base_iri, conn=conn
                )
                if s["form_type"] == "PREF"
            ]
        return seen[key]

    uid_of = {iri: c.uid for c, iri in candidates}
    labels, rows = assign_pref_labels(
        [
            {
                "iri": iri,
                "kind": c.kind,
                "name": c.name,
                "label": representative_label(c),
            }
            for c, iri in candidates
        ],
        stored_prefs=stored_prefs,
    )
    # Re-tagged from the settled string, not from the candidate: a concept that gave way falls
    # back to its own de-split name, which may sit on the other side of the English test than the
    # label it lost. `pref_lang_for` is the same rule `pref_label_of` used, asked again.
    return {
        uid_of[iri]: {pref_lang_for(label): label} for iri, label in labels.items()
    }, rows


def mintable_gray(
    outcomes, graph, base_iri, *, claimed=None
) -> tuple[dict[int, str], int]:
    """Which GRAY outcomes can still be minted, as ``({id(outcome): iri}, n_unmintable)``.

    A queued candidate used to be left out of the store entirely. That reads as caution but is
    not: every individual naming it loses its anchor and is dropped a stage later as a `miss`,
    with no record of which ones. Minting it *and* queueing it is the trade `route_by_verdict`
    already made on the bootstrap path — a duplicate concept is visible and a reviewer can merge
    it, a missing anchor is neither visible nor recoverable.

    The exception is the case that made some of these gray to begin with: a candidate whose slug
    already names a live concept cannot be minted without clobbering a stable IRI.
    A re-ingested name lands here and needs nothing — its anchor already exists from the first
    ingest — so what stays unminted is exactly what has nowhere to go.

    ``claimed`` carries the IRIs this same batch is about to mint for its ADDs. ``graph`` is the
    pre-mutation snapshot, so without it a gray whose slug collides with an in-batch ADD (slugify
    folds "on order" and "on-order" onto one segment) looks free and both concepts collapse onto
    one IRI — the clobber the exception above exists to refuse, just sourced from this batch
    rather than the store.
    """
    mints: dict[int, str] = {}
    unmintable = 0
    claimed = claimed or frozenset()
    for o in outcomes:
        if o.kind != OUTCOME_GRAY:
            continue
        own_iri = build_concept(o.members, base_iri).iri
        if (
            _iri_exists(graph, own_iri)
            or own_iri in mints.values()
            or own_iri in claimed
        ):
            unmintable += 1
            continue
        mints[id(o)] = own_iri
    return mints, unmintable


def _settle_stored_claims_on_label(
    repo: Repository, label, rep, iri, cid, result, *, base_iri: str, conn
) -> None:
    """Resolve the live terms already holding ``label``, for one label string this batch is
    writing as a PREF. Split out of :func:`_mint_concept` only so it can be called per language
    without nesting the whole body a level deeper."""
    for claimant in repo.surface_claimants(
        label, rep.kind, base_iri=base_iri, exclude_iri=cid, conn=conn
    ):
        if claimant["form_type"] == "PREF":
            # Two representative labels. Deliberately not resolved here: taking the batch's side
            # would rename a live concept, and an IRI renewal is a human-gated event (review.py),
            # not something an ingest does in passing. Left as it is and reported; the store-wide
            # scan below queues it in this same run.
            #
            # resolve_pref_labels settles this before the graph is built, so reaching here means it
            # could not: the candidate lost both rungs of its ladder — the label it asked for and
            # its own name — to live terms. That is the one shape that stays unresolved, and this
            # is where it says so, after the pre-graph rows have already said the label moved.
            result.pref_label_collisions.append(
                {
                    "kind": rep.kind,
                    "label": label,
                    "kept": claimant["iri"],
                    "relabelled": None,
                    "fallback": None,
                    "rule": "store keeps its representative label; renaming is human-gated",
                }
            )
            continue
        dropped = repo.drop_surface_form(claimant["iri"], label, conn=conn)
        if dropped:
            result.demoted_store_aliases.append(
                {
                    "surface": label,
                    "kind": rep.kind,
                    "alias_owner": claimant["iri"],
                    "claimed_by": iri,
                    "form_type": claimant["form_type"],
                    "rows": dropped,
                }
            )


def _mint_concept(
    repo: Repository,
    o,
    iri,
    model_id,
    centered,
    glosses,
    cqs,
    cq_db_id,
    version_id,
    result,
    flagged_verdicts,
    conn,
    *,
    base_iri,
    labels=None,
) -> None:
    """Create one concept from an outcome: term row, surfaces by role, CQ links, delta rows.

    Shared by ADD and by a GRAY that could still be minted. A queued candidate used to be left
    out of the store entirely, which reads as caution but is not: every individual naming it
    loses its anchor and is dropped a stage later, with no record of which ones. Minting it and
    queueing it is the same trade `route_by_verdict` already made on the bootstrap path — a
    duplicate concept is visible and mergeable, a missing anchor is neither.
    """
    rep = o.representative
    # The vector is supplied at creation: the store's vector column is NOT NULL, so a term
    # cannot be inserted now and embedded afterwards.
    vec = centered.get(rep.uid)
    if vec is None:
        raise ValueError(f"no vector for new concept {iri} (uid {rep.uid})")
    pref = _pref_labels_for(rep, labels)
    cid = repo.upsert_concept(
        iri,
        rep.kind,
        pref,
        rep.name,
        embedding_model_id=model_id,
        embedding_val=vec,
        snapshot_id=version_id,
        conn=conn,
    )
    # Representative-label uniqueness across batches — the mirror image of surface uniqueness (A).
    # A says "do not attach an ALTER to a string another ACTIVE term already resolves to"; this is
    # the same string contested from the other end and later in time: the batch is writing that
    # string as a PREF, and an alias somebody attached earlier is already on it. Whoever is second
    # does not decide it — PREF does. A name outranks a synonym.
    #
    # Nothing is lost by dropping the alias, by the same argument A makes for refusing one: the
    # string still resolves, now to the term it names. Today it resolves to neither, because two
    # ACTIVE terms hold it and the loader drops every individual that uses it. The cross-batch
    # cases seen in practice are this shape, the store side ALTER or HIDDEN every time. A HIDDEN
    # one earns its keep when it is a surface's only way to resolve (many are); one that collides
    # with a PREF is by definition not that.
    #
    # Once per distinct label *string*, not once per language: the question is who owns the
    # string, and ``surface_claimants`` matches ``lower(surface_text)`` filtering no language —
    # exactly like the resolver whose behaviour this protects. A term whose ko and en labels are
    # the same string asks once.
    for label in sorted(set(pref.values())):
        _settle_stored_claims_on_label(
            repo, label, rep, iri, cid, result, base_iri=base_iri, conn=conn
        )
    # Every member's surfaces attach to this one concept (one IRI per cluster). Members
    # can share a surface (a member's name is another's curated duplicate), so resolve each
    # distinct surface's role BEFORE inserting rather than letting insert order decide it: a form
    # that is a standard AlterTerm for ANY member stays alter (never bury a synonym some member —
    # or a human-curated duplicate — treats as valid); only a form hidden by every member that
    # claims it becomes a HiddenTerm.
    role_by_surface: dict[str, tuple[str, str, str]] = {}
    for m in o.members:
        tt = o.term_type_for(m.uid)
        for s, lang, src in _surface_entries(m):
            cur = role_by_surface.get(s)
            if cur is None or (cur[2] == "hidden" and tt == "alter"):
                role_by_surface[s] = (lang, src, tt)
    for role in ("alter", "hidden"):
        entries = [
            (s, lang, src) for s, (lang, src, r) in role_by_surface.items() if r == role
        ]
        if role == "alter":
            # Surface uniqueness (A): an AlterTerm string that already resolves to a different
            # ACTIVE term is not a synonym this concept may claim — attaching it makes that string
            # ambiguous, and the ABox loader then drops every individual naming it rather than
            # resolving to one. Not a loss: the string keeps resolving to the term that already
            # owns it, which is the only term an individual could have meant. HiddenTerms are
            # exempt (they are document artifacts, never a resolution target). The concept's own
            # name is a PREF handled by upsert_concept and is not in this alter set.
            kept = []
            for s, lang, src in entries:
                if (
                    repo.active_surface_owner(
                        s, rep.kind, base_iri=base_iri, exclude_iri=cid, conn=conn
                    )
                    is not None
                ):
                    result.n_surface_forms_skipped += 1
                else:
                    kept.append((s, lang, src))
            entries = kept
        repo.insert_surface_forms(cid, entries, term_type=role, conn=conn)
    for m in o.members:
        linked, missing = _link_cqs(repo, cid, m, cqs, cq_db_id, conn)
        result.n_cq_linked += linked
        result.n_cq_missing += missing
        result.mapping_delta.append(_delta_row(m, "new", iri))
        # A conflict component is minted (other components reference it) but must not pass as
        # settled: queue it with the per-chunk disagreement as the evidence.
        flagged = (flagged_verdicts or {}).get(m.uid)
        if flagged:
            rid = repo.enqueue_review(
                base_iri=base_iri,
                candidate_uid=m.uid,
                kind=m.kind,
                entry_type="gray",
                gloss=glosses.get(m.uid),
                candidate_vector=centered.get(m.uid),
                nearest=[],
                payload={
                    **_component_payload(m, glosses.get(m.uid, "")),
                    "target_iri": iri,
                    "reason": flagged_reason(flagged),
                    "upstream_verdict": flagged,
                    "disagreement_flags": list(m.disagreement_flags),
                },
                conn=conn,
            )
            if rid is not None:
                result.n_verdict_queued += 1
                result.n_review_rows += 1


def _persist_update(
    repo: Repository,
    outcomes,
    new_ttl,
    base_id,
    model_id,
    persist_vectors,
    centered,
    glosses,
    cqs,
    result,
    graph=None,
    flagged_verdicts=None,
    kind_conflicts=None,
    gray_mints=None,
    base_iri=None,
    sources=None,
    pref_labels=None,
    pref_label_collisions=None,
) -> int:
    """Commit the batch atomically: snapshot + concept/surface rows + risky-item enqueue with
    verdicts, all inside one transaction so a rollback discards queue rows too."""
    result.pref_label_collisions.extend(pref_label_collisions or [])
    with repo.transaction() as conn:
        version_id = repo.save_ontology_version(
            new_ttl, base_iri=base_iri, note="update", base_id=base_id, conn=conn
        )
        # Once per transaction, not per component: _link_cqs runs inside the outcome loop and its
        # link_term_sources call needs every document source already present — which is also why
        # the citations are collected here rather than reusing cq_db_id, built later in that loop.
        # Scoped to the documents this batch cites: a shared store holds other documents' CQs.
        repo.sync_document_sources(
            cited_cq_ids={
                q
                for o in outcomes
                for m in getattr(o, "members", ())
                for q in m.source_cq_ids
            },
            conn=conn,
        )
        # Adapter-declared sources (kind='rdb' etc.) register at the same point as the document
        # spine and BEFORE any link — link_term_sources silently drops pairs naming a source not
        # yet in the registry, so a later call would lose this batch's RDB provenance. The document
        # path derives its rows; RDB has no derivation source here, so the adapter must send them.
        # Adapter-declared rdb sources arrive as {database, sources:[...]}; register the leaf
        # tables (rdb_database/table/ref) and the source rows in FK order, before any term link.
        if sources and sources.get("sources"):
            repo.register_rdb_sources(sources, conn=conn)
        cq_db_id: dict[str, int] = {}

        for o in outcomes:
            if o.kind == OUTCOME_ADD:
                _mint_concept(
                    repo,
                    o,
                    o.target_iri,
                    model_id,
                    centered,
                    glosses,
                    cqs,
                    cq_db_id,
                    version_id,
                    result,
                    flagged_verdicts,
                    conn,
                    base_iri=base_iri,
                    labels=pref_labels,
                )
                result.n_new += 1
            elif o.kind == OUTCOME_SYNONYM_ADD:
                comp = o.representative
                cid = repo.get_concept_id(o.target_iri)
                if cid is not None:
                    term_type = o.term_type_for(comp.uid)
                    entries = _surface_entries(comp)
                    if term_type == "alter":
                        # Surface uniqueness (A) on the third write path. It guarded both minting
                        # paths and not this one, which is the one that runs most: an attach is the
                        # normal warm outcome, so every later document re-created the collisions
                        # minting had just refused. A cold batch would drop the ALTER that clashed
                        # with another concept's PREF, then the next document's candidate attached
                        # and put it straight back — the same string owned by two ACTIVE terms
                        # again, and every individual naming it dropped by the ABox loader.
                        #
                        # Same rule and the same non-loss as at mint time: the string keeps
                        # resolving to the term that already owns it. exclude_iri is the anchor
                        # itself, so a form it already carries (this candidate re-sending a
                        # synonym the first ingest attached) does not read as a rival to itself.
                        kept = []
                        for s, lang, src in entries:
                            if (
                                repo.active_surface_owner(
                                    s,
                                    comp.kind,
                                    base_iri=base_iri,
                                    exclude_iri=cid,
                                    conn=conn,
                                )
                                is not None
                            ):
                                result.n_surface_forms_skipped += 1
                            else:
                                kept.append((s, lang, src))
                        entries = kept
                    repo.insert_surface_forms(
                        cid, entries, term_type=term_type, conn=conn
                    )
                    # The concept absorbed this candidate, so it absorbs the questions the
                    # candidate came from. Linking only on the ADD path left every merged
                    # candidate's provenance behind — and because the count lives inside
                    # _link_cqs, a path that never called it reported nothing missing.
                    linked, missing = _link_cqs(repo, cid, comp, cqs, cq_db_id, conn)
                    result.n_cq_linked += linked
                    result.n_cq_missing += missing
                result.n_merged += 1
                result.mapping_delta.append(_delta_row(comp, "merge", o.target_iri))
            else:  # OUTCOME_UPDATE or OUTCOME_GRAY -> enqueue (post-gate, in txn) with verdict
                entry_type = "relabel" if o.kind == OUTCOME_UPDATE else "gray"
                # Enqueue one review row PER MEMBER, not just the representative: a multi-member
                # GRAY cluster (borderline cohesion / slug collision) must leave every distinct
                # candidate individually reviewable, or non-representative members are silently
                # lost. Attaches have a single member, so this is one row there.
                verdict_dict = o.verdict.to_dict() if o.verdict is not None else None
                for m in o.members:
                    # The queue freezes the candidate's vector, so an item without one cannot be
                    # decided later — surface that here instead of enqueuing an undecidable row.
                    cand_vec = centered.get(m.uid)
                    if cand_vec is None:
                        raise ValueError(f"no vector for review candidate {m.uid}")
                    rid = repo.enqueue_review(
                        base_iri=base_iri,
                        candidate_uid=m.uid,
                        kind=m.kind,
                        entry_type=entry_type,
                        gloss=glosses.get(m.uid),
                        candidate_vector=cand_vec,
                        nearest=o.nearest,
                        payload={
                            **_component_payload(m, glosses.get(m.uid, "")),
                            "target_iri": o.target_iri,
                            "reason": o.reason,
                        },
                        conn=conn,
                    )
                    # Idempotent re-enqueue returns None; only attach a verdict to a fresh row.
                    if rid is not None:
                        result.n_review_rows += 1
                        if verdict_dict is not None:
                            repo.set_review_verdict(rid, verdict_dict, conn=conn)
                    result.pending.append(
                        {
                            "candidate_uid": m.uid,
                            "name": m.name,
                            "kind": m.kind,
                            "entry_type": entry_type,
                            "nearest": o.nearest,
                        }
                    )
                own_iri = (gray_mints or {}).get(id(o))
                if own_iri is not None:
                    # Minted *and* queued: the anchor resolves now, and the reviewer still gets
                    # to decide whether it should have merged into its near neighbour instead.
                    _mint_concept(
                        repo,
                        o,
                        own_iri,
                        model_id,
                        centered,
                        glosses,
                        cqs,
                        cq_db_id,
                        version_id,
                        result,
                        flagged_verdicts,
                        conn,
                        base_iri=base_iri,
                        labels=pref_labels,
                    )
                    result.n_gray_minted += 1
                # A candidate that stays in review is not minted, so its CQ links are not made
                # yet — but that is deferral, not loss, and must not be counted as missing.
                # n_cq_missing means "a cited CQ is not in the store" (the FK-violation risk the
                # ABox loader hits later); it is not "a term for this link does not exist yet".
                # An earlier version added the latter here and inflated the count roughly a
                # hundredfold — the cited CQs were in the store, on other terms, and this
                # candidate's own term will link them once a reviewer decides. A slug-collision
                # gray in particular already has a live term from the first ingest carrying them.
                if o.kind == OUTCOME_UPDATE:
                    result.n_relabel += 1
                else:
                    result.n_gray += 1

        # Same queuing the bootstrap does. Both sides of a kind conflict are minted or attached
        # by the loop above — separate IRIs are the design — so what is queued is the
        # contradiction, with both ranges as the evidence.
        #
        # Its own entry_type, so a candidate already queued as GRAY for a different reason does
        # not swallow it: the queue is unique per (candidate_uid, entry_type).
        for conflict in kind_conflicts or []:
            # The anchor is a batch side — only those have a uid and a frozen vector. A
            # cross-batch conflict's stored side is already a term; it rides in the payload as
            # evidence, not as the queued candidate.
            batch_sides = [s for s in conflict["sides"] if "uid" in s]
            anchor = min((s["uid"] for s in batch_sides), default=None)
            vec = centered.get(anchor) if anchor is not None else None
            if vec is None:
                # Only conflicts among stored terms, with no batch-side claim at all, remain
                # here; a batch side filtered out by withholding gets its vector restored
                # above and is queued. (The old wording lumped the withheld-deletion case
                # and this one into one sentence, hiding the cause.)
                logger.warning(
                    "update: kind conflict on %r has no batch-side claimant to anchor "
                    "a queue row (stored-term conflict); reported without queueing",
                    conflict["name"],
                )
                continue
            anchor_kind = next(s["kind"] for s in batch_sides if s["uid"] == anchor)
            # Minting state: the batch side from this run's mapping_delta, the stored side
            # from its own iri. In the withheld branch it is [None, None], and reason
            # states exactly that.
            minted_by_uid = {
                r["candidate_uid"]: r["iri"] for r in result.mapping_delta
            }
            minted = [
                side.get("iri") or minted_by_uid.get(side.get("uid"))
                for side in conflict["sides"]
            ]
            rid = repo.enqueue_review(
                base_iri=base_iri,
                candidate_uid=anchor,
                kind=anchor_kind,
                entry_type="kind_conflict",
                gloss=glosses.get(anchor),
                candidate_vector=vec,
                nearest=[],
                payload={
                    "name": conflict["name"],
                    "names": conflict["names"],
                    "kind": anchor_kind,
                    # A sentence computed from the actual minting state; the same function
                    # as cold, so the wording cannot diverge again (the "one copy each for
                    # cold/warm" that two duplicated f-strings suffered from).
                    "reason": kind_conflict_reason(conflict, minted),
                    "kinds": conflict["kinds"],
                    # Same shape as cold: without this key there is no structured field
                    # to refute wrong prose.
                    "minted_iris": minted,
                    "sides": [
                        {
                            **side,
                            "unresolved_cq_ids": [
                                q
                                for q in side.get("source_cq_ids", [])
                                if cqs.get(q) is None
                            ],
                        }
                        for side in conflict["sides"]
                    ],
                },
                conn=conn,
            )
            if rid is not None:
                result.n_review_rows += 1
            # review_id None means the queue already held it, not that the write failed — see
            # the identical note in generate. The uid is `kind:name`, so a name two documents each
            # judge as two kinds files once and no-ops after.
            result.queued_kind_conflicts.append(
                {
                    "name": conflict["name"],
                    "review_id": rid,
                    "already_queued": rid is None,
                }
            )
        result.kind_conflicts = list(kind_conflicts or [])

        # Mirror the validated graph's structure into the relational tables, in the same
        # transaction so the stored edges match the snapshot being committed.
        if graph is not None:
            # Which concept each outcome's edge claims belong to. Settled outcomes only: a
            # minted concept (ADD, or a gray mintable_gray could still mint) and a confident
            # attach, whose candidate really did merge into the anchor and so re-asserts that
            # concept's stored edges. UPDATE and the unminted grays are queued, not applied —
            # whether the candidate is that concept is the open question, so attributing its
            # claims to the anchor would record a merge no human approved.
            #
            # Built from target_iri only for the kinds where it is this concept's own IRI. On a
            # GRAY it is the anchor it was proposed against, or the live IRI its slug collided
            # with; there, only mintable_gray's answer is.
            attributable = {
                id(o): o.target_iri
                for o in outcomes
                if o.kind in (OUTCOME_ADD, OUTCOME_SYNONYM_ADD) and o.target_iri
            }
            attributable.update(gray_mints or {})
            topo = repo.replace_topology(
                extract_topology(graph),
                base_iri=base_iri,
                edge_sources=collect_edge_sources_warm(
                    outcomes, attributable, graph, base_iri
                ),
                conn=conn,
            )
            result.n_hierarchy_skipped = topo.get("skipped_hierarchy", 0)
            result.hierarchy_cycles_broken = topo.get("cycle_edges_dropped", [])
            result.n_domain_range_skipped = topo.get("skipped_domain_range", 0)
            result.n_unattributed_edges = topo.get("unattributed_edges", 0)
            result.n_edge_sources_skipped = topo.get("skipped_edge_sources", 0)

        # Warm-path collision detection (B): A stops NEW splits, but concepts already split across
        # earlier documents remain — the same string on two ACTIVE terms (Country / NationState /
        # State). The warm path never recomputed collisions against the accumulated store, so most
        # went unqueued and invisible. Scan the store and queue each so a human can merge; the queue
        # dedups per (candidate_uid, entry_type), so a collision already raised is a no-op.
        for coll in repo.find_ambiguous_active_surfaces(base_iri=base_iri, conn=conn):
            # Record the detection too, not only the queueing: the warm path reported
            # surface_collisions=0 while queueing some, because this scan fed the queue but not the
            # detection list. They are one axis — detected vs acted-on — so a collision counted in
            # queued_surface_collisions must also appear here. scope='store' distinguishes an
            # accumulated-store collision from generate's in-batch ones.
            result.surface_collisions.append(
                {
                    "surface": coll["surface"],
                    "effect": "ambiguous",
                    "scope": "store",
                    "kinds": [coll["kind"]],
                    "iris": coll["iris"],
                }
            )
            rid = repo.enqueue_review(
                base_iri=base_iri,
                # Casefolded, to dedup on what resolution collides on — see the same key in
                # generate's in-batch scan. The two paths report different spellings of one
                # string, so an un-folded key put the same decision in the queue twice.
                candidate_uid=f"surface:{coll['kind']}:{coll['surface'].lower()}",
                kind=coll["kind"],
                entry_type="gray",
                candidate_vector=coll["vector"],
                nearest=[],
                payload={
                    "surface": coll["surface"],
                    "kind": coll["kind"],
                    "minted_iris": coll["iris"],
                    "reason": f"surface {coll['surface']!r} resolves to "
                    f"{len(coll['iris'])} ACTIVE terms of the same kind; the loader "
                    f"returns an arbitrary one and drops the rest",
                },
                conn=conn,
            )
            if rid is not None:
                result.n_review_rows += 1
            # Appended either way, with which one it was. Only fresh rows used to be recorded, so
            # a collision the queue already held counted as detected-but-not-queued and
            # surface_collisions stayed permanently one ahead of queued — read as "one is missing
            # from the queue" when nothing was missing. n_review_rows still counts growth only.
            result.queued_surface_collisions.append(
                {
                    "surface": coll["surface"],
                    "iris": coll["iris"],
                    "review_id": rid,
                    "already_queued": rid is None,
                }
            )

        # Read back which of this batch's minted terms landed with no provenance — the same table
        # check generate does, over the IRIs this path touched. link_term_sources has already run
        # (per component, above), so term_source_map reflects what actually stuck.
        # Newly minted terms only (ADD and minted GRAY) — a SYNONYM_ADD extends a term that
        # already existed and carried its own provenance, which is not this batch's to answer for.
        minted = {
            o.target_iri for o in outcomes if o.target_iri and o.kind == OUTCOME_ADD
        }
        minted |= set((gray_mints or {}).values())
        if minted:
            result.n_terms_without_source = conn.execute(
                "SELECT count(*) FROM tbox_term t WHERE t.id = ANY(%s) AND NOT EXISTS "
                "(SELECT 1 FROM term_source_map m WHERE m.term_id = t.id)",
                (list(minted),),
            ).fetchone()[0]

    return version_id


def _bootstrap_result(gr: GenerateResult, repo, base_iri: str) -> UpdateResult:
    """Translate a bootstrap generate() into the /update contract (all concepts are 'new').

    A cold /update routes inside generate(), and every counter generate() fills has to
    travel — a carried-over 0 is a measurement, a dropped one is a silent lie. This used to
    be a hand-written field list and twice something quietly fell off it (four counters,
    then unresolved_domains), so the shared fields are now copied mechanically by name and
    only the fields where copying would be WRONG are spelled out, each with its reason.
    """
    latest = repo.get_latest_ontology(base_iri) if gr.ok else None
    delta = gr.mapping_delta if gr.ok else []
    overrides = {
        "ok": gr.ok,
        # Latest snapshot from the repo — a different thing than GenerateResult.ttl.
        "ttl": latest[1] if latest else None,
        # On gate failure nothing was persisted, so the delta must be empty too.
        "mapping_delta": delta,
        # /update-only fields (no GenerateResult counterpart), filled with bootstrap meaning.
        "version_id": latest[0] if latest else None,
        "n_candidates": gr.n_components,
        "n_new": sum(1 for r in delta if r["outcome"] == "new"),
        "n_merged": sum(1 for r in delta if r["outcome"] == "merge"),
        "pending": [],
    }
    shared = {f.name for f in fields(GenerateResult)} & {
        f.name for f in fields(UpdateResult)
    }
    carried = {name: getattr(gr, name) for name in shared if name not in overrides}
    return UpdateResult(**carried, **overrides)


def _delta_row(component, outcome: str, iri: str) -> dict:
    """One row-level resolution: this candidate's raw name -> canonical IRI."""
    return {
        "candidate_uid": component.uid,
        "name": component.name,
        "kind": component.kind,
        "outcome": outcome,  # "merge" | "new"
        "iri": iri,
    }


def _link_cqs(repo, cid, component, cqs, cq_db_id, conn) -> tuple[int, int]:
    """Link a term to the CQs it came from. Returns ``(linked, missing)``.

    A cited CQ that is not in the store cannot be linked, and that loss is invisible here — it
    resurfaces one stage later as a foreign-key violation when the ABox loader inserts an
    individual carrying the same cq id. So it is counted and reported rather than skipped quietly.
    """
    linked = missing = 0
    for uuid in sorted(set(component.source_cq_ids)):
        if uuid not in cq_db_id:
            cq = cqs.get(uuid)
            if cq is None:
                missing += 1
                continue
            cq_db_id[uuid] = uuid  # already in the store; upstream owns cq
        repo.link_concept_cq(cid, cq_db_id[uuid], conn=conn)
        linked += 1
    # The source-neutral rows, written from the component's own union so an RDB-sourced component
    # (no cq_id at all) still lands provenance. link_term_sources drops unregistered sources
    # itself, so a component citing a CQ that was never registered costs its provenance link, not
    # the batch.
    repo.link_term_sources(
        [(cid, s) for s in component_source_ids(component)],
        conn=conn,
    )
    return linked, missing
