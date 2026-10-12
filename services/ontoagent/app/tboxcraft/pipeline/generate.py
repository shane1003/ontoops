"""Generate-mode pipeline.

load -> gloss -> embed -> calibrate -> cluster -> represent -> emit -> validate. The TBox is
returned only when the validation gate passes (syntax + DL + HermiT); OOPS! is advisory. A
human-reviewable merge report records every embedding-driven merge (Option A has no review
queue, so this is the spot-check).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from app.tboxcraft.calibration import calibrate
from app.tboxcraft.cluster import cluster_components
from app.tboxcraft.config import Config
from app.tboxcraft.embed import make_provider
from app.tboxcraft.embed.base import EmbeddingProvider
from app.tboxcraft.emit.tbox import (
    build_graph,
    class_iri_by_name,
    scan_datatype_substitutions,
    scan_unresolved_broaders,
    _case_variant_map,
    scan_unresolved_domains,
    scan_unresolved_ranges,
    serialize_turtle,
    write_tbox,
)
from app.tboxcraft.emit.topology import extract_topology
from app.tboxcraft.gloss import build_gloss
from app.tboxcraft.io import parse_components
from app.tboxcraft.io.components import (
    fold_case_variants,
    kind_conflict_reason,
    scan_kind_conflicts,
    scan_non_english_labels,
    scan_overridden_lang_declarations,
    scan_slug_collisions,
    withheld_kind_conflict_uids,
    withheld_slug_collision_uids,
)
from app.tboxcraft.io.cq import resolve_cqs
from app.tboxcraft.mapping import (
    build_mapping,
    scan_surface_collisions,
    write_mapping_jsonl,
)
from app.tboxcraft.pipeline.verdict import flagged_reason, route_by_verdict, surface_role
from app.tboxcraft.provenance import collect_edge_sources, collect_term_sources
from app.tboxcraft.represent import build_concepts, dedupe_pref_labels, is_hash_namespace
from app.tboxcraft.similarity import center_query, mean_center
from app.core.tracing import traced_pipeline
from app.tboxcraft.validate import gate, summarize, validate_tbox

if TYPE_CHECKING:  # annotation only — no runtime import cycle
    from app.tboxcraft.db import Repository

logger = logging.getLogger(__name__)


@dataclass
class GenerateResult:
    ok: bool
    gate_reasons: list[str] = field(default_factory=list)
    ttl_path: Path | None = None
    mapping_path: Path | None = None
    report_path: Path | None = None
    # How this batch passed, not just that it did. ok=true says the gate found no reason to
    # refuse; it does not say consistency was proven — a run with the reasoner switched off has
    # `consistent` None and nothing to refuse, so it passes unchecked. A consumer that treats PASS
    # as "verified" needs to be able to tell those apart, and until now the report existed only as
    # a file on the server.
    validation: dict = field(default_factory=dict)
    merge_report_path: Path | None = None
    n_components: int = 0
    n_concepts: int = 0
    persisted: bool = False
    # Verifier health. Counted by outcome, not by attempt: annotation stops early once the LLM
    # stops answering (rather than hammering it once per concept), so "attempts" would hide how
    # many concepts actually ended up with no verdict. A dead verifier still yields ok=true and a
    # valid TTL by design, so without these a caller has only the logs to notice.
    n_verified: int = 0
    n_verifier_unavailable: int = 0
    # What the verification cost. n_verified says how many concepts got a verdict;
    # n_verify_calls says how many LLM round trips that took, and is the whole point of batching
    # — un-batched the two are equal, and the gap between them is the saving. Kept as counters
    # rather than a log line because a caller comparing two runs cannot assert on a log line.
    n_clusters: int = 0
    n_verify_calls: int = 0
    # [{kind, tau_high, tau_low, source, n_positives}] — the merge threshold actually in force,
    # per kind. Per kind and not a single figure because calibration is kind-partitioned and the
    # failure it exists to expose was kind-local: once the class partition crossed MIN_POSITIVES,
    # it calibrated tau_high far too low off a handful of positives and collapsed every class into
    # one concept while the two property partitions, still on the seed, looked entirely normal.
    # A flat tau_high would have reported one of the healthy kinds.
    # `source` is the part that has to survive any retuning of the calibration itself: raising
    # MIN_POSITIVES moves the day this happens, it does not remove it, and "we started trusting
    # the computed threshold" is the event to watch for.
    calibration: list[dict] = field(default_factory=list)
    # Merges above tau_high that were refused because the two sides were not similar to each
    # other. Reads with `calibration`: a threshold that is too low shows up here first, as a large
    # number of refusals, before it shows up as a wrong concept count.
    n_chained_refused: int = 0
    # [{surface, iris, sources}] — strings that name more than one concept. Neither term resolves
    # reliably afterwards: the store returns one arbitrarily and an ABox mapper seeing both drops
    # the individual as ambiguous, so the loss lands two stages away from the cause with nothing
    # connecting them. Reported, not raised — a batch legitimately containing `hasRole` as both an
    # object and a data property is not an error, it just cannot be resolved by that string alone.
    surface_collisions: list[dict] = field(default_factory=list)
    # Which of those collisions are in the queue — [{surface, iris, review_id, already_queued}].
    # Detection alone does not say whether anything was done about it, and a caller cannot
    # reconcile "found 3" with "queued 3" from one list; refused_duplicates already separates the
    # two the same way. ``already_queued`` (review_id None) is a collision the queue held from an
    # earlier run — enqueue_review is idempotent per (candidate_uid, entry_type). It is in the
    # list because it IS queued; leaving it out made detected exceed queued forever and read as a
    # queue gap. The remaining gap is real: a collision whose every claimant is outside this batch
    # has no vector to freeze and is reported without being queued.
    queued_surface_collisions: list[dict] = field(default_factory=list)
    # [{kind, label, kept, relabelled, fallback, rule}] — two concepts claimed one representative
    # label, so the loser was moved onto its own de-split name before the graph was built. Its own
    # list rather than folded into surface_collisions: those are strings that ARE ambiguous, these
    # are strings that would have been and are not, and the row is a merge question rather than a
    # defect. `fallback` None means the de-split name was taken too, so that concept's label was
    # left alone — the one case here that is still a live collision.
    pref_label_collisions: list[dict] = field(default_factory=list)
    # Which of those reached the merge queue — [{label, kept, relabelled, review_id,
    # already_queued}]. Same detected-vs-acted-on split as the two lists above.
    queued_pref_label_collisions: list[dict] = field(default_factory=list)
    # [{kind, kept, folded, ...}] — same-kind names differing only in case, merged into one
    # term before minting. This changes an IRI, so it is reported rather than assumed: which
    # spelling survives is a convention (the canonical_member rule), not a judgment that one
    # spelling is better, and the row carries what the folded side brought with it.
    folded_names: list[dict] = field(default_factory=list)
    # [{name, kinds, sides}] — one name judged as two kinds (usesWeapon as both an object and a
    # data property). Both are minted: separate IRIs are the design and the collision
    # detector only fires on an exact IRI match, so the contradiction otherwise sits in the
    # ontology with nothing raised. `sides` carries both ranges, which is what the decision turns
    # on — a class range on one side and xsd:string on the other are not the same claim.
    kind_conflicts: list[dict] = field(default_factory=list)
    # Which of those were acted on — [{name, review_id}]. Same split as the two lists above:
    # detection does not say whether anything was done about it.
    queued_kind_conflicts: list[dict] = field(default_factory=list)
    # Under a HASH namespace neither reading is minted: the two would collapse onto one IRI, and
    # OWL 2 DL keeps object and data properties disjoint. Cold start reaches this too — every
    # namespace's FIRST document bootstraps through here, so leaving the withholding to the warm
    # path alone would let exactly the batch that establishes a namespace mint the contested pair.
    # Counts the contested NAMES, so it matches len(withheld_kind_conflicts) the way
    # every other n_* in this result matches its list. The number of TERMS not minted
    # is a different question — one name withholds both readings — and gets its own
    # field rather than overloading this one, because a gate asserting on "how many
    # terms are missing from the TBox" and one asserting on "how many names are
    # contested" want different numbers.
    n_kind_conflicts_withheld: int = 0
    n_terms_withheld: int = 0
    withheld_kind_conflicts: list[dict] = field(default_factory=list)
    # Slug collisions — distinct names whose slugs collide, so one IRI would name several terms.
    # Counted apart from the kind-conflict withholding above rather than added into it: both
    # withhold, but one says "we do not know which kind this name is" and the other says "we cannot
    # tell these names apart once minted", and a reader deciding whether a batch is safe to accept
    # needs to know which. Expected to stay 0 — see scan_slug_collisions on why it is wired anyway.
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
    # [{a, b, kind, reason}] — curated 'same concept' claims the property gate refused.
    # Both sides were minted, so nothing is missing from the ontology; what is missing is
    # the curator's intent, and these are queued for review rather than skipped.
    refused_duplicates: list[dict] = field(default_factory=list)
    # Surfaces written as HiddenTerms — non-standard forms kept for ABox mapping but not
    # offered as synonyms. 0 on every bootstrap before this was wired: the verifier's
    # is_standard_form was recorded and then ignored, so hidden_confidence_floor did
    # nothing on the cold path.
    n_hidden_surfaces: int = 0
    # Upstream TBox/ABox verdict tally. All three are minted (route_by_verdict says why an 'abox'
    # component is still minted); the flagged ones are queued for review. A cold /update
    # bootstraps through generate, so this is the path most batches are routed on.
    n_tbox: int = 0
    n_conflict: int = 0
    n_abox: int = 0
    n_verdict_queued: int = 0
    # Review rows this run actually inserted, from every source. n_verdict_queued covers only the
    # verdict-derived ones, so a bootstrap that queued refused duplicates or surface collisions
    # reported fewer rows than it wrote. Same field as UpdateResult so a cold and a warm run
    # reconcile against the queue the same way.
    n_review_rows: int = 0
    # Minted terms that cite no source at all, and stored edges no source asserted. The spine's
    # "every artefact traces to an origin" invariant is a write-path promise the DDL cannot make,
    # and neither can a run whose input omits provenance — inventing a source would make the chain
    # assert something the input never said. So both are reported. n_unattributed_edges is
    # expected to be non-zero (an edge that exists only because of the merge has no asserting
    # source); n_terms_without_source at 0 is the one worth pinning.
    n_terms_without_source: int = 0
    # AlterTerm surfaces not attached because the string already resolves to a different ACTIVE
    # term — including one minted earlier in THIS batch (surface uniqueness A on the cold path).
    # 0 before this was wired, which is why cold batches created the splits A/B later cleaned up.
    n_surface_forms_skipped: int = 0
    n_unattributed_edges: int = 0
    # Edge sources the adapter named that are not in the registry — the batch drops them and stores
    # the edge without provenance. update fills this from replace_topology; generate did not, though
    # the same dict carries it. An invariant: 0 unless the adapter cites an unregistered source.
    n_edge_sources_skipped: int = 0
    # --- what this run silently dropped or altered ---------------------------------------------
    # Each of these was previously only a log line (or not even that), so a caller saw ok=true and
    # a valid TTL with no way to tell the output was incomplete. They are reported, not raised:
    # none of them should fail an ingest, but none should pass unnoticed either.
    n_cq_linked: int = 0
    n_cq_missing: int = 0
    n_hierarchy_skipped: int = 0
    # [{child, parent}] — subClassOf edges dropped because they closed a cycle. Legal OWL (the
    # classes are equivalent) that the relational mirror cannot hold, so it is a report, not a gate.
    hierarchy_cycles_broken: list[dict] = field(default_factory=list)
    n_domain_range_skipped: int = 0
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
    # Per-component resolution (raw name -> minted IRI): the canonical member is "new",
    # cluster-mates that folded into it are "merge". Feeds /update's bootstrap mapping_delta.
    mapping_delta: list[dict] = field(default_factory=list)

    @property
    def verifier_degraded(self) -> bool:
        """True when verification was asked for but nothing came back — effectively disabled."""
        return self.n_verifier_unavailable > 0 and self.n_verified == 0

    @property
    def cq_provenance_degraded(self) -> bool:
        """True when CQ links were requested but none resolved — every term lost its provenance."""
        return self.n_cq_missing > 0 and self.n_cq_linked == 0


def _build_merge_report(merge_edges) -> list[dict]:
    return [asdict(e) for e in merge_edges]


def _bootstrap_delta(concepts) -> list[dict]:
    """Per-component raw name -> concept IRI for the initial ontology (one row per member)."""
    rows: list[dict] = []
    for concept in concepts:
        for member in concept.members:
            rows.append(
                {
                    "candidate_uid": member.uid,
                    "name": member.name,
                    "kind": member.kind,
                    "outcome": "new"
                    if member.name == concept.canonical_name
                    else "merge",
                    "iri": concept.iri,
                }
            )
    return rows


@traced_pipeline("generate")
def generate(
    components_path: str | Path,
    config: Config,
    *,
    provider: EmbeddingProvider | None = None,
    run_reasoner: bool = True,
    run_oops: bool = False,
    persist_db: bool = False,
    repo=None,
    source_uri: str | None = None,
    components_doc: dict | None = None,
    verify: bool = True,
    llm=None,
    sources: dict | None = None,
) -> GenerateResult:
    provider = provider or make_provider(config)
    # Components may arrive inline (service-to-service) or from a file. CQs come from the DB
    # `cq` table; absent -> an empty index (the gloss is viable from name + notes
    # alone).
    if components_doc is not None:
        components = parse_components(components_doc)
        doc_sources = components_doc.get("sources") or {}
    else:
        raw_doc = json.loads(Path(components_path).read_text(encoding="utf-8"))
        components = parse_components(raw_doc)
        # Adapter-declared sources ride in the same document as the components (GenerateRequest
        # takes a path, so there is no separate field for them as /update has).
        doc_sources = raw_doc.get("sources") or {}
    # An explicit argument as well, because /update carries sources in a field of its own and its
    # cold branch delegates here: without a parameter to hand them to, they had nowhere to go and
    # the in-document key stayed a slot nobody filled. Argument wins over the document; either
    # way both paths must see the same value.
    sources = sources or doc_sources
    # One term written two ways in one kind is an identity fact, not a similarity question;
    # folded before anything keys on the name, so only one IRI is minted for it.
    components, folded_names = fold_case_variants(components)
    if folded_names:
        logger.warning(
            "generate: folded %d name(s) differing only in case — %s",
            len(folded_names),
            "; ".join(f"{r['folded']} -> {r['kept']}" for r in folded_names[:5]),
        )
    # One name judged as two kinds. Read from the components, not from the minted concepts or
    # the surface table: a concept's local name follows its cluster representative (so a
    # conflicting member whose cluster is named after somebody else stops being findable by
    # name), and a single-word name is written only as a PREF row (so it never reaches the
    # surface scan at all). Both cases are present in real batches.
    kind_conflicts = scan_kind_conflicts(components)
    withheld_conflicts: list[dict] = []
    withheld_components: list = []
    n_withheld = n_terms = 0
    if kind_conflicts and is_hash_namespace(config.base_iri):
        # Kind conflicts under a hash namespace. No cross-batch half here: a cold start is by
        # definition a namespace with nothing in it, so there is no earlier term to demote — every
        # contested side is in this batch and neither is minted.
        withheld_uids = withheld_kind_conflict_uids(kind_conflicts)
        # Removed from minting only: the component objects are used below as queueing
        # material (vectors, glosses). Previously they were deleted from the list here, so
        # the vectors the queue freezes were never computed and every withheld conflict
        # skipped the queue as "reported without queueing" (the one justification for
        # withholding, "it goes to a human", did not hold).
        withheld_components = [c for c in components if c.uid in withheld_uids]
        components = [c for c in components if c.uid not in withheld_uids]
        n_withheld, n_terms, withheld_conflicts = (
            len(kind_conflicts),
            len(withheld_uids),
            kind_conflicts,
        )
        logger.warning(
            "generate: %d component(s) withheld — a hash namespace cannot separate an object "
            "property from a data property of the same name, so neither side is minted and both "
            "stay queued: %s",
            n_terms,
            "; ".join(
                f"{k['name']}({'/'.join(k['kinds'])})" for k in kind_conflicts[:5]
            ),
        )
    elif kind_conflicts:
        logger.warning(
            "generate: %d name(s) judged as two kinds — both minted, queued for review: %s",
            len(kind_conflicts),
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
            "generate: %d surface(s) declared `en` by their producer carry no Latin letter — written "
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
            "generate: %d curated term(s) kept as an alternative surface — the representative "
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

    # Distinct names that slug onto one IRI. Not a namespace-form question like the kind conflicts
    # above — the local part is the slug under both forms — and no cross-batch half, for the same
    # reason the kind conflicts have none here: a cold start has nothing already minted to collide
    # with.
    slug_collisions = scan_slug_collisions(components)
    n_slug_terms_withheld = 0
    if slug_collisions:
        withheld_slugs = withheld_slug_collision_uids(slug_collisions)
        components = [c for c in components if c.uid not in withheld_slugs]
        n_slug_terms_withheld = len(withheld_slugs)
        logger.warning(
            "generate: %d component(s) withheld — %d IRI(s) would name more than one term because "
            "the slug is not reversible, and an identity cannot be un-merged later: %s",
            n_slug_terms_withheld,
            len(slug_collisions),
            "; ".join(
                f"{'/'.join(c['names'])} -> {c['slug']}" for c in slug_collisions[:5]
            ),
        )

    # Read the upstream TBox/ABox verdict. Nothing is filtered out — see route_by_verdict.
    flagged_verdicts, verdict_tally = route_by_verdict(components, stage="generate")
    cqs = resolve_cqs(repo)
    logger.info(
        "generate: loaded %d components (%d CQs from %s)",
        len(components),
        len(cqs),
        "db" if repo is not None else "none",
    )

    # Gloss + embed, keyed by uid (kind-qualified; dual-kind names never collide).
    ordered = sorted(components, key=lambda c: c.uid)
    glosses = [build_gloss(c, cqs, max_cq_sample=config.max_cq_sample) for c in ordered]
    logger.info("generate: embedding %d glosses via %s...", len(glosses), provider.name)
    matrix = provider.embed(glosses)
    # Whiten (mean-center) the batch so compressed e5 cosines become discriminative for
    # clustering; the mean is persisted so query vectors can be centered into the same space.
    centered, embed_mean = mean_center(matrix)
    vectors = {c.uid: centered[i] for i, c in enumerate(ordered)}
    # Withholding excludes from minting; it does not delete evidence. The vectors
    # and glosses the review queue freezes are restored here. Why not embed them
    # **together** with the main batch: the mean_center mean would then include the
    # withheld ones and shift every vector of the batch slightly (existing output must not
    # change). The withheld ones are centered separately with the already fixed mean and
    # merged in.
    withheld_gloss_by_uid: dict[str, str] = {}
    if withheld_components:
        withheld_ordered = sorted(withheld_components, key=lambda c: c.uid)
        withheld_glosses = [
            build_gloss(c, cqs, max_cq_sample=config.max_cq_sample)
            for c in withheld_ordered
        ]
        withheld_matrix = provider.embed(withheld_glosses)
        for comp, gloss, row in zip(
            withheld_ordered, withheld_glosses, withheld_matrix
        ):
            vectors[comp.uid] = center_query(row, embed_mean)
            withheld_gloss_by_uid[comp.uid] = gloss
    logger.info(
        "generate: embedded + mean-centered %d vectors (dim=%d)",
        len(vectors),
        provider.dimension,
    )

    thresholds = calibrate(
        components,
        vectors,
        provider.name,
        seed_high=config.tau_high_seed,
        seed_low=config.tau_low_seed,
    )
    calibration_rows = [
        {
            "kind": kind,
            "tau_high": round(t.tau_high, 4),
            "tau_low": round(t.tau_low, 4),
            "source": t.source,
            "n_positives": t.n_positives,
            # Names the `duplicates` entries the producer got wrong. n_positives alone cannot: a
            # sample that shrank because a claim was rejected reads the same as one that was small.
            "excluded_positives": list(t.excluded_positives),
        }
        for kind, t in sorted(thresholds.items())
    ]
    logger.info("generate: thresholds %s", calibration_rows)
    for row in calibration_rows:
        if row["source"] != "seed":
            # The threshold stopped being the vetted default. Whether that is right depends on a
            # sample this pipeline cannot judge, so say it out loud rather than only in a dict.
            logger.warning(
                "generate: %s merge threshold is %s at %.4f from %d curated positive(s)",
                row["kind"],
                row["source"],
                row["tau_high"],
                row["n_positives"],
            )
        if row["excluded_positives"]:
            logger.warning(
                "generate: %s calibration rejected %d curated duplicate pair(s) as miscurated: %s",
                row["kind"],
                len(row["excluded_positives"]),
                "; ".join(
                    f"{r['a']}={r['b']} at {r['cosine']}"
                    for r in row["excluded_positives"][:3]
                ),
            )
    clustered = cluster_components(components, vectors, thresholds)
    concepts = build_concepts(clustered.clusters, config.base_iri)
    logger.info(
        "generate: clustered %d components -> %d concepts (%d embedding merges, %d chains refused)",
        len(components),
        len(concepts),
        sum(1 for e in clustered.merge_edges if e.source == "embedding"),
        clustered.n_chained_refused,
    )
    if clustered.refused_duplicates:
        logger.warning(
            "generate: %d curated duplicate claim(s) blocked by the property gate — "
            "both sides minted, queued for review: %s",
            len(clustered.refused_duplicates),
            "; ".join(f"{r['a']}={r['b']}" for r in clustered.refused_duplicates[:5]),
        )

    # Before the graph, because the label goes into the TTL the gate and HermiT read. Two concepts
    # holding one representative label is one string naming two terms, and resolution does not
    # filter form_type, so the loader drops every individual that uses it.
    pref_label_collisions = dedupe_pref_labels(concepts)
    if pref_label_collisions:
        logger.warning(
            "generate: %d concept(s) claimed a representative label another already holds — "
            "relabelled to their own de-split name and queued for a merge decision: %s",
            len(pref_label_collisions),
            "; ".join(
                f"{r['label']!r} kept by {r['kept'].rsplit('/', 1)[-1]}, "
                f"{r['relabelled'].rsplit('/', 1)[-1]} -> {r['fallback']!r}"
                for r in pref_label_collisions[:5]
            ),
        )

    graph = build_graph(
        concepts,
        base_iri=config.base_iri,
        version_iri=config.version_iri,
        license_iri=config.license_iri,
    )

    logger.info(
        "generate: validating emitted TBox (reasoner=%s, oops=%s)...",
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
    passed, reasons = gate(report)
    validation = summarize(report)

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Report + merge report are always written (even on gate failure).
    report_path = out / "validation_report.json"
    report_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    merge_report_path = out / "merge_report.json"
    merge_report_path.write_text(
        json.dumps(
            _build_merge_report(clustered.merge_edges), ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    batch_classes = class_iri_by_name(concepts)
    result = GenerateResult(
        ok=passed,
        validation=validation,
        gate_reasons=reasons,
        report_path=report_path,
        merge_report_path=merge_report_path,
        n_components=len(components),
        n_concepts=len(concepts),
        n_tbox=verdict_tally["tbox"],
        n_conflict=verdict_tally["conflict"],
        n_abox=verdict_tally["abox"],
        mapping_delta=_bootstrap_delta(concepts),
        # Ranges the OWL 2 datatype map cannot hold were substituted to keep HermiT runnable;
        # report which, since the emitted range no longer says what the source asked for.
        datatype_substitutions=scan_datatype_substitutions(components),
        unresolved_broaders=scan_unresolved_broaders(components),
        # Same rule build_graph resolved with, so the report describes the graph
        # that was actually emitted.
        unresolved_ranges=scan_unresolved_ranges(
            components,
            batch_classes.__contains__,
            case_variants=_case_variant_map(batch_classes),
        ),
        unresolved_domains=scan_unresolved_domains(
            components,
            batch_classes.__contains__,
            case_variants=_case_variant_map(batch_classes),
        ),
        calibration=calibration_rows,
        folded_names=folded_names,
        # Scanned before the gate, so report them even when the gate fails or the run does not
        # persist — the contradiction exists in the input either way. Only queued_kind_conflicts
        # is persist-only, since only that path can enqueue.
        kind_conflicts=kind_conflicts,
        n_kind_conflicts_withheld=n_withheld,
        n_terms_withheld=n_terms,
        withheld_kind_conflicts=withheld_conflicts,
        n_slug_collisions_withheld=len(slug_collisions),
        n_slug_terms_withheld=n_slug_terms_withheld,
        withheld_slug_collisions=slug_collisions,
        non_english_labels=non_english_labels,
        n_lang_declarations_overridden=len(overridden_langs),
        overridden_lang_declarations=overridden_langs,
        refused_duplicates=clustered.refused_duplicates,
        n_chained_refused=clustered.n_chained_refused,
    )

    if not passed:
        logger.warning("generate: gate FAILED -> %s", "; ".join(reasons))
        return result  # do not emit an invalid TBox
    logger.info("generate: validation gate PASSED")

    ttl_path = write_tbox(graph, out / "tbox.ttl", out / "tbox.owl")
    result.ttl_path = ttl_path

    mapping_entries = build_mapping(concepts)
    result.pref_label_collisions = pref_label_collisions
    result.surface_collisions = scan_surface_collisions(concepts, mapping_entries)
    if result.surface_collisions:
        # Neither side resolves reliably from here on, and the individual that goes
        # missing does so in a later stage that cannot see this.
        logger.warning(
            "generate: %d surface form(s) name more than one concept — e.g. %s; "
            "resolving those strings returns an arbitrary one of them",
            len(result.surface_collisions),
            ", ".join(r["surface"] for r in result.surface_collisions[:5]),
        )
    if persist_db and repo is not None:
        gloss_by_uid = {c.uid: g for c, g in zip(ordered, glosses)}
        # Merge in the withheld glosses: material the kind_conflict queue row shows the
        # reviewer.
        gloss_by_uid.update(withheld_gloss_by_uid)
        # Judge before writing: the verdict decides whether a member's surface goes in as a
        # standard AlterTerm or a HiddenTerm, and that has to be settled before the row exists.
        # Still outside the write transaction — a hanging verifier must not hold one open.
        verdicts: dict = {}
        if verify:
            if llm is None:
                try:
                    from app.tboxcraft.llm import make_llm

                    llm = make_llm(config)
                except Exception as exc:  # noqa: BLE001 - annotation is optional
                    logger.warning(
                        "generate: verifier unavailable, skipping annotation: %s", exc
                    )
            if llm is not None:
                judged_at = time.perf_counter()
                verdicts, result.n_clusters, result.n_verify_calls = _judge_members(
                    concepts, config, llm, vectors, glosses=gloss_by_uid
                )
                # Its own line, because judging happens *between* the gate and the persist log.
                # Anyone timing those two lines from outside reads the whole verifier fan-out as
                # database time — that has already happened once, and it pointed a performance
                # investigation at the wrong component.
                logger.info(
                    "generate: judged %d member(s) in %d LLM call(s) over %d batch(es) in %.1fs "
                    "(verify_concurrency=%d, verify_cluster_size=%d)",
                    len(verdicts),
                    result.n_verify_calls,
                    result.n_clusters,
                    time.perf_counter() - judged_at,
                    config.verify_concurrency,
                    config.verify_cluster_size,
                )
        roles = _surface_roles(concepts, verdicts, config.hidden_confidence_floor)
        n_hidden = sum(1 for r in roles.values() if r == "hidden")
        if n_hidden:
            logger.info(
                "generate: %d surface(s) buried as HiddenTerms — non-standard forms kept for ABox "
                "mapping, not offered as synonyms",
                n_hidden,
            )
        persist_at = time.perf_counter()
        (
            cq_linked,
            cq_missing,
            topo,
            n_queued,
            n_rows,
            queued_collisions,
            queued_kind_conflicts,
            n_no_source,
            n_surface_skipped,
            queued_pref_labels,
        ) = _persist_to_db(
            repo,
            concepts,
            mapping_entries,
            vectors,
            cqs,
            provider,
            embed_mean,
            base_iri=config.base_iri,
            ttl=serialize_turtle(graph),
            topology=extract_topology(graph),
            flagged_verdicts=flagged_verdicts,
            glosses=gloss_by_uid,
            refused_duplicates=clustered.refused_duplicates,
            roles=roles,
            surface_collisions=result.surface_collisions,
            kind_conflicts=kind_conflicts,
            sources=sources,
            pref_label_collisions=pref_label_collisions,
        )
        result.n_hidden_surfaces = n_hidden
        result.persisted = True
        result.n_cq_linked, result.n_cq_missing = cq_linked, cq_missing
        result.n_verdict_queued = n_queued
        result.n_review_rows = n_rows
        result.queued_surface_collisions = queued_collisions
        result.queued_pref_label_collisions = queued_pref_labels
        result.queued_kind_conflicts = queued_kind_conflicts
        result.n_terms_without_source = n_no_source
        result.n_surface_forms_skipped = n_surface_skipped
        result.n_unattributed_edges = topo.get("unattributed_edges", 0)
        result.n_edge_sources_skipped = topo.get("skipped_edge_sources", 0)
        result.n_hierarchy_skipped = topo.get("skipped_hierarchy", 0)
        result.hierarchy_cycles_broken = topo.get("cycle_edges_dropped", [])
        result.n_domain_range_skipped = topo.get("skipped_domain_range", 0)
        logger.info(
            "generate: persisted %d concepts + snapshot to DB in %.1fs",
            len(concepts),
            time.perf_counter() - persist_at,
        )
        if result.cq_provenance_degraded:
            logger.error(
                "generate: NONE of %d requested CQ links resolved — every term was stored without "
                "provenance, and an ABox insert citing those CQ ids will fail on the foreign key",
                cq_missing,
            )
        elif cq_missing:
            logger.warning(
                "generate: %d CQ links dropped (CQ not in the store)", cq_missing
            )
        if verify:
            result.n_verified, result.n_verifier_unavailable = _record_verdicts(
                repo, concepts, verdicts, config
            )
            if result.verifier_degraded:
                logger.error(
                    "generate: the verifier answered for NONE of %d concepts — verification was "
                    "effectively disabled for this run (check TBOXCRAFT_LLM_URL / model name)",
                    result.n_verifier_unavailable,
                )
    else:
        # File mode: JSONL mapping is retired once DB persistence is the default.
        result.mapping_path = write_mapping_jsonl(
            mapping_entries, out / "mapping.jsonl"
        )
        logger.info("generate: wrote TTL %s (file mode)", ttl_path)
    return result


def _nearest_concepts_by_uid(concepts, vectors, k: int) -> dict[str, list[dict]]:
    """``{member uid: [{iri, label, score}]}`` — the k nearest OTHER concepts, per member.

    What the update path shows the verifier is the anchor a candidate landed near, with its score.
    At bootstrap there is no accumulated ontology to be near, so the neighbourhood is the rest of
    this batch — the same question ("what does this term sit beside?"), asked about the only
    concepts that exist yet.

    Own concept excluded: a member is trivially nearest to the concept it is already in, and
    saying so answers nothing. Cosine on the mean-centered vectors, i.e. the same space clustering
    routed on, so a score here means what a score there meant.
    """
    if not vectors:
        return {}
    canon = [(c, f"{c.kind}:{c.canonical_name}") for c in concepts]
    known = [(c, uid) for c, uid in canon if vectors.get(uid) is not None]
    if len(known) < 2:
        return {}
    matrix = np.asarray([vectors[uid] for _, uid in known], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0, 1.0, norms)
    index_of_iri = {c.iri: i for i, (c, _) in enumerate(known)}

    out: dict[str, list[dict]] = {}
    for concept in concepts:
        row = index_of_iri.get(concept.iri)
        if row is None:
            continue
        scores = matrix @ matrix[row]
        scores[row] = -np.inf  # the concept it is already in says nothing
        top = np.argsort(-scores)[:k]
        neighbours = [
            {
                "iri": known[j][0].iri,
                "label": known[j][0].sole_label or known[j][0].canonical_name,
                "score": float(scores[j]),
            }
            for j in top
            if np.isfinite(scores[j])
        ]
        for member in concept.members:
            out[member.uid] = neighbours
    return out


def _judge_members(
    concepts, config, llm, vectors=None, *, glosses=None, n_nearest: int = 3
) -> tuple[dict, int, int]:
    """Judge every member of every concept. Returns ``({uid: Verdict}, n_batches, n_calls)``.

    Runs before persistence, and outside any transaction: what the verifier says decides whether a
    member's surface is a standard AlterTerm or a HiddenTerm kept only for ABox mapping, and that
    has to be known when the row is written. Doing it afterwards would mean writing every surface
    as 'alter' and then moving some — a second pass whose only honest form is the one reserved for
    a human overriding the machine.

    Every member is judged, not just the one that names the concept. The update path already does
    this (``approval`` splits each non-representative member into alter/hidden); the bootstrap
    judged only the representative, which is why ``hidden_confidence_floor`` had no effect on a
    first ingest at all.

    Two things shrink the wall clock, deliberately separate:

    * **Fan-out.** Batches are judged over ``config.verify_concurrency`` threads.
    * **Batching.** ``config.verify_cluster_size`` candidates share one call, and the reply is
      keyed per candidate, so a verdict still answers a question about one term.

    A verifier that stops answering must not be hammered once per member. The stop takes a *streak*
    of batches that came back with nothing usable rather than a single failure: with N calls in
    flight the first failure arrives next to N-1 others that may well have answered. Batches
    already running are drained rather than abandoned — they are paid for either way, and reading
    them keeps both the verdicts and the call count honest.
    """
    from app.tboxcraft.verify_batch import group_for_verification, run_batched_judgment

    nearest_by_uid = _nearest_concepts_by_uid(concepts, vectors, n_nearest)

    # One entry per member. Members of a concept sit in the same batch group as their concept, so
    # a cluster is judged against its own neighbourhood rather than scattered across calls.
    members_of = {c.iri: list(c.members) for c in concepts}
    ordered = [
        m
        for group in group_for_verification(
            concepts, vectors, config.verify_cluster_size
        )
        for concept in group
        for m in members_of[concept.iri]
    ]
    return run_batched_judgment(
        ordered,
        llm,
        cluster_size=config.verify_cluster_size,
        concurrency=config.verify_concurrency,
        glosses=glosses,
        nearest_by_uid=nearest_by_uid,
        domain=config.domain_description,
        stage="generate",
    )


def _surface_roles(concepts, verdicts, floor: float) -> dict[str, str]:
    """``{uid: 'alter' | 'hidden'}``. The member that names the concept carries the PrefTerm and
    stays a standard form; the rest are split by what the verifier said about their form."""
    from app.tboxcraft.verify_batch import canonical_member

    roles: dict[str, str] = {}
    for concept in concepts:
        rep = canonical_member(concept)
        for member in concept.members:
            roles[member.uid] = (
                "alter"
                if member.uid == rep.uid
                else surface_role(verdicts.get(member.uid), floor)
            )
    return roles


def _record_verdicts(repo, concepts, verdicts, config) -> tuple[int, int]:
    """Attach each concept's verdict — the one about the member that names it. Advisory:
    recorded, never gates.

    Returns ``(n_verified, n_unavailable)`` counted by *outcome*: every concept left without a
    verdict counts as unavailable, including ones the early stop cancelled before they were tried.
    Counting attempts instead would report a single failure and hide that the rest of the batch
    went unverified.
    """
    from app.tboxcraft.verify_batch import canonical_member

    total = len(concepts)
    flagged = verified = 0
    for concept in concepts:
        cid = repo.get_concept_id(concept.iri)
        verdict = verdicts.get(canonical_member(concept).uid)
        if cid is None or verdict is None:
            continue
        # The model is recorded with the verdict: opinions from different verifiers are not
        # comparable, and the store refuses a verdict without that provenance.
        repo.set_concept_verdict(cid, verdict.to_dict(), model=config.llm_model)
        verified += 1
        if not verdict.is_official_term or verdict.confidence < 0.5:
            flagged += 1
    unavailable = total - verified
    if unavailable:
        logger.warning(
            "generate: verifier answered for %d of %d concepts — %d left unverified",
            verified,
            total,
            unavailable,
        )
    logger.info(
        "generate: annotated %d concepts (%d flagged for later review)",
        verified,
        flagged,
    )
    return verified, unavailable


def _persist_to_db(
    repo: Repository,
    concepts,
    mapping_entries,
    vectors,
    cqs,
    provider,
    embed_mean,
    *,
    base_iri,
    ttl=None,
    topology=None,
    flagged_verdicts=None,
    glosses=None,
    refused_duplicates=None,
    roles=None,
    surface_collisions=None,
    kind_conflicts=None,
    sources=None,
    pref_label_collisions=None,
):
    """Persist concepts, mapping, representative vectors, source/CQ provenance, and the
    canonical TBox snapshot.

    Returns ``(cq_linked, cq_missing, topology_counts, n_verdict_queued, n_review_rows,
    queued_collisions)`` so
    the caller can report what the write could not carry over — see :class:`GenerateResult`.
    """
    cq_linked = cq_missing = n_queued = n_rows = n_surface_skipped = 0
    queued_collisions: list[dict] = []
    queued_pref_labels: list[dict] = []
    queued_kind_conflicts: list[dict] = []
    topo_counts: dict = {}
    model_name = getattr(provider, "model_name", provider.name)
    revision = getattr(provider, "revision", "unknown")
    dim = provider.dimension
    persist_vectors = (
        dim == 1024
    )  # the vector column is fixed at 1024 (e5); others are file-only

    # Surfaces grouped by concept and then by role. A MappingEntry no longer knows which member
    # produced the string, so the members are walked again.
    #
    # A surface that IS a member's name takes that member's role, even when another member also
    # offers the string. The two are different claims: the member named "the order that was
    # issued" is the messy form the verifier judged, while a sibling listing that string among its
    # `duplicates` is asserting a merge, not vouching for the spelling — letting the sibling's role
    # win would un-bury every form that any other member happened to reference.
    #
    # For the rest — correct_terms, synonyms, duplicates naming nothing in this concept — the role
    # follows whoever contributed it, and 'alter' wins a disagreement: a form one member uses as a
    # standard synonym must not be buried because another did not.
    role_by_surface: dict[str, dict[str, str]] = {}
    for concept in concepts:
        per_concept = role_by_surface.setdefault(concept.iri, {})
        named_by_member = {m.name for m in concept.members}
        for member in concept.members:
            per_concept[member.name] = (roles or {}).get(member.uid, "alter")
        for member in concept.members:
            role = (roles or {}).get(member.uid, "alter")
            for surface, _lang, _source in member.surface_candidates():
                if surface in named_by_member:
                    continue
                if per_concept.get(surface) != "alter":
                    per_concept[surface] = role

    by_iri: dict[str, dict[str, list]] = {}
    for entry in mapping_entries:
        role = role_by_surface.get(entry.iri, {}).get(entry.surface, "alter")
        by_iri.setdefault(entry.iri, {}).setdefault(role, []).append(
            (entry.surface, entry.lang, entry.source)
        )

    if not persist_vectors:
        # A term row cannot exist without a vector, so a provider whose dimension the store cannot
        # hold has nothing to persist — fail here rather than part-way through the transaction.
        raise ValueError(
            f"cannot persist: the store holds 1024-dim vectors, provider gives {dim}"
        )

    # Who ends up owning each representative label, decided for the whole batch before any row is
    # written. The ALTER filter below needs it: active_surface_owner sees only what this loop has
    # already inserted, so whether an alias clashing with another concept's PREF is refused came
    # down to which of the two was written first. `MDMP` (its name is a surface candidate, so it
    # offers 'MDMP' as an ALTER) sorts before `MdmpProcess` (which holds 'MDMP' as its label), so
    # the alias always won and the string kept naming two terms even after dedupe_pref_labels had
    # picked one owner for the label.
    #
    # Keyed by kind and label string with no language in it, matching `assign_pref_labels`'s
    # ownership key and `resolve_surface`'s query: a string is taken or it is not, whatever tag
    # the row that took it carries. A concept holding labels in several languages owns each of
    # those strings, so every one of them goes in.
    pref_owner = {
        (c.kind, label.casefold()): c.iri
        for c in concepts
        for label in c.pref_labels.values()
        if label
    }

    # One transaction for the whole run — a mid-persist failure rolls back entirely.
    with repo.transaction() as conn:
        snapshot_id = None
        if ttl is not None:
            snapshot_id = repo.save_ontology_version(
                ttl, base_iri=base_iri, note="generate", conn=conn
            )
        model_id = repo.upsert_embedding_model(
            model_name, revision, dim, mean=embed_mean, conn=conn
        )

        cq_db_id: dict[str, str] = {}
        cq_links: list[tuple[str, str]] = []
        for concept in concepts:
            # The vector is part of creating the term, not a follow-up update: the column is
            # NOT NULL, so an insert without it would fail.
            canon_uid = f"{concept.kind}:{concept.canonical_name}"
            vec = vectors.get(canon_uid)
            if vec is None:
                raise ValueError(f"no vector for {concept.iri} (canonical {canon_uid})")
            cid = repo.upsert_concept(
                concept.iri,
                concept.kind,
                concept.pref_labels,
                concept.canonical_name,
                embedding_model_id=model_id,
                embedding_val=vec,
                snapshot_id=snapshot_id,
                conn=conn,
            )
            for role, entries in sorted(by_iri.get(concept.iri, {}).items()):
                if role == "alter":
                    # Surface uniqueness inside the batch (A on the cold path): the warm path filters
                    # via active_surface_owner, but generate did not, so two concepts minted in one
                    # cold batch could split the same AlterTerm and there was no store yet to catch
                    # it. active_surface_owner reads the caller's transaction, so a term this loop
                    # inserted earlier already counts — the first concept keeps the string, the rest
                    # skip it, and it resolves to one term instead of dropping the individuals.
                    kept = []
                    for s_, lang_, src_ in entries:
                        # Another concept in this batch owns the string as its label, whether or not
                        # it has been written yet — the store query cannot see the ones still to
                        # come, and PREF outranks an alias regardless of who got there first.
                        claimed = pref_owner.get((concept.kind, s_.casefold()))
                        if (claimed is not None and claimed != concept.iri) or (
                            repo.active_surface_owner(
                                s_,
                                concept.kind,
                                base_iri=base_iri,
                                exclude_iri=cid,
                                conn=conn,
                            )
                            is not None
                        ):
                            n_surface_skipped += 1
                        else:
                            kept.append((s_, lang_, src_))
                    entries = kept
                repo.insert_surface_forms(cid, entries, term_type=role, conn=conn)

            # Minted (the anchor must resolve) but not settled: queue the flagged members so the
            # modelling can be corrected with the individuals still attached. Per member, not per
            # concept — a cluster can hold one 'abox' member beside settled ones, and reviewing
            # the representative would not surface it.
            for member in concept.members:
                flagged = (flagged_verdicts or {}).get(member.uid)
                if not flagged:
                    continue
                rid = repo.enqueue_review(
                    base_iri=base_iri,
                    candidate_uid=member.uid,
                    kind=member.kind,
                    entry_type="gray",
                    gloss=(glosses or {}).get(member.uid),
                    candidate_vector=vectors.get(member.uid),
                    nearest=[],
                    payload={
                        "name": member.name,
                        "kind": member.kind,
                        "target_iri": concept.iri,
                        "reason": flagged_reason(flagged),
                        "upstream_verdict": flagged,
                        # Which chunk said what — the evidence for the reviewer.
                        "disagreement_flags": list(member.disagreement_flags),
                    },
                    conn=conn,
                )
                if rid is not None:
                    n_queued += 1
                    n_rows += 1

            cq_uuids: set[str] = set()
            for member in concept.members:
                cq_uuids.update(member.source_cq_ids)
            for uuid in sorted(cq_uuids):
                if uuid not in cq_db_id:
                    cq = cqs.get(uuid)
                    if cq is None:
                        # The component cites a CQ that is not in the store, so its provenance
                        # link is lost. Counted, not skipped in silence: a downstream ABox insert
                        # referencing this cq_id fails on the FK a whole stage later.
                        cq_missing += 1
                        continue
                    # Present in the store (that is where `cqs` was read from), so the id is the
                    # link key as-is. Writing it back was a store-to-store no-op left over from
                    # when CQs arrived as a CSV; the upstream ingestion model owns that table now.
                    cq_db_id[uuid] = uuid
                # Collected and written once after the loop: this is the highest-volume table in
                # a bootstrap, and one round trip per link was the bulk of the per-row cost.
                cq_links.append((cid, cq_db_id[uuid]))
                cq_linked += 1

        repo.link_concept_cqs(cq_links, conn=conn)
        # The source-neutral successor, written alongside rather than instead of the CQ links:
        # a component may cite RDB sources that have no cq_id at all, and term_cq_map is still the
        # column the ABox loader joins until it moves over too.
        # Derive the document-kind spine rows before anything links to them: link_term_sources
        # drops pairs naming a source that is not there yet, so the wrong order costs the batch its
        # provenance silently. Upstream owns `cq` and does not write `source`.
        # Scoped to the documents this batch cites — a shared store holds other documents' CQs, and
        # registering those too makes each document's source count the store-wide total.
        repo.sync_document_sources(
            cited_cq_ids={
                q for c in concepts for m in c.members for q in m.source_cq_ids
            },
            conn=conn,
        )
        # Adapter-declared sources (kind='rdb' etc.) register at the same point, before the link.
        # link_term_sources drops pairs naming an unregistered source, so a later call would lose
        # this batch's RDB provenance. The document spine is derived above; RDB has no derivation.
        # See update_apply: rdb sources arrive as {database, sources:[...]} and register their
        # leaf tables + source rows in FK order, before link_term_sources.
        if sources and sources.get("sources"):
            repo.register_rdb_sources(sources, conn=conn)
        repo.link_term_sources(collect_term_sources(concepts), conn=conn)
        # The "no orphan terms" invariant is a write-path promise the DDL cannot make, and this run
        # cannot keep it either: a component citing no source leaves nothing to link, and one whose
        # only CQ was never registered has its link dropped. Inventing a source would make the
        # chain assert something the input never said, so the gap is counted instead — visible
        # rather than indistinguishable from "provenance was never collected". Read back from the
        # table so it reflects what actually landed, not what was offered.
        iris = [c.iri for c in concepts]
        n_no_source = conn.execute(
            "SELECT count(*) FROM tbox_term t WHERE t.id = ANY(%s) AND NOT EXISTS "
            "(SELECT 1 FROM term_source_map m WHERE m.term_id = t.id)",
            (iris,),
        ).fetchone()[0]

        # A curated "these are the same concept" the property gate would not let through. Both
        # sides were minted, so the ontology is usable, but the contradiction between what the
        # curator asserted and what the gate allowed is a decision — and the update path already
        # routes exactly this to review. Queued per pair, on the lexicographically first side, so
        # a rerun of the same batch does not stack duplicates (the queue dedups per candidate).
        iri_by_uid = {m.uid: c.iri for c in concepts for m in c.members}
        for refused in refused_duplicates or []:
            rid = repo.enqueue_review(
                base_iri=base_iri,
                candidate_uid=refused["a_uid"],
                kind=refused["kind"],
                entry_type="gray",
                gloss=(glosses or {}).get(refused["a_uid"]),
                candidate_vector=vectors.get(refused["a_uid"]),
                nearest=[],
                payload={
                    "name": refused["a"],
                    "kind": refused["kind"],
                    "target_iri": iri_by_uid.get(refused["b_uid"]),
                    "reason": refused["reason"],
                    # Both sides, so the reviewer sees the claim and the block together.
                    "duplicate_of": refused["b"],
                    "minted_iris": [
                        iri_by_uid.get(refused["a_uid"]),
                        iri_by_uid.get(refused["b_uid"]),
                    ],
                },
                conn=conn,
            )
            if rid is not None:
                n_rows += 1

        # A string that names two concepts of one kind. Unlike the cross-kind case, no query-side
        # hint separates them — the caller cannot say "the class one" when both are classes — so
        # the resolver returns an arbitrary side and the individuals naming it are split or lost.
        # Both concepts stay ACTIVE: the producer asserted both, and refusing either destroys an
        # anchor. What is queued is the contradiction itself, so a human picks the owner.
        #
        # Keyed by the surface, not by a claimant. The queue dedups per (candidate_uid,
        # entry_type), and a claimant that is already queued for another reason would swallow this
        # one silently — the collision would then be detected and reported as unqueued forever.
        canon_uid_by_iri = {c.iri: f"{c.kind}:{c.canonical_name}" for c in concepts}

        # Two concepts claimed one representative label. dedupe_pref_labels already moved the loser
        # off it, so nothing is ambiguous any more and this is not a defect report — it is the merge
        # question the fallback deliberately did not answer. Claiming the same label is a strong
        # signal of one concept, but not proof (ElectronicIntelligenceWarfare may be inside
        # ElectronicWarfare rather than equal to it), and most such pairs fall far
        # below tau_high because these concepts carry no gloss — the embedding sees only the surface
        # string, so an abbreviation and its expansion look unrelated. Similarity cannot settle it;
        # a human can.
        for row in pref_label_collisions or []:
            anchor = canon_uid_by_iri.get(row["relabelled"])
            vec = vectors.get(anchor) if anchor else None
            if vec is None:
                logger.warning(
                    "generate: pref-label collision on %r has no vector in this batch; "
                    "reported without queueing",
                    row["label"],
                )
                continue
            rid = repo.enqueue_review(
                base_iri=base_iri,
                candidate_uid=f"pref:{row['kind']}:{(row['label'] or '').lower()}",
                kind=row["kind"],
                entry_type="pref_conflict",
                gloss=(glosses or {}).get(anchor),
                candidate_vector=vec,
                nearest=[],
                payload={
                    "label": row["label"],
                    "kind": row["kind"],
                    "kept": row["kept"],
                    "relabelled": row["relabelled"],
                    "fallback": row["fallback"],
                    "rule": row["rule"],
                    "reason": f"{row['relabelled']} and {row['kept']} both claimed the "
                    f"representative label {row['label']!r}; the label stays with "
                    f"{row['kept']} ({row['rule']}) and the other fell back to "
                    f"{row['fallback']!r}. Merge them if they are one concept.",
                },
                conn=conn,
            )
            if rid is not None:
                n_rows += 1
            queued_pref_labels.append(
                {
                    "label": row["label"],
                    "kept": row["kept"],
                    "relabelled": row["relabelled"],
                    "review_id": rid,
                    "already_queued": rid is None,
                }
            )

        # A label collision is also seen by scan_surface_collisions, which reads what the batch
        # claimed rather than what the write path accepted: 'Loc' still offers 'Location' as an
        # ALTER (its correct_term is a surface candidate too), so the string still looks contested
        # even though surface A refuses that row and the store ends with one owner. Queueing it
        # again would put the same merge question in front of a human twice under two keys.
        pref_queued = {
            (r["kind"], (r["label"] or "").lower()) for r in pref_label_collisions or []
        }
        for collision in surface_collisions or []:
            if collision.get("scope") != "same_kind":
                continue
            if (collision["kinds"][0], collision["surface"].lower()) in pref_queued:
                continue
            anchor = next(
                (
                    canon_uid_by_iri[i]
                    for i in collision["iris"]
                    if i in canon_uid_by_iri
                ),
                None,
            )
            vec = vectors.get(anchor) if anchor else None
            if vec is None:
                # Every claimant is outside this batch (an /update attach): nothing to freeze, and
                # the queue refuses a vectorless candidate. Reported but not queued, rather than
                # dropped — the caller sees it in surface_collisions without it.
                logger.warning(
                    "generate: surface %r collides but no claimant is in this batch; "
                    "reported without queueing",
                    collision["surface"],
                )
                continue
            rid = repo.enqueue_review(
                base_iri=base_iri,
                # Casefolded, because the queue key has to dedup on the same thing resolution
                # collides on: (kind, casefold(surface)). Keyed on the written spelling, one
                # collision sat in the queue twice — this scan prefers a written surface's
                # spelling for display while the warm scan reports whatever the store holds, so
                # 'Civil Considerations' and 'civil considerations' were two review rows for one
                # decision (and n_review_rows inflated to match).
                candidate_uid=f"surface:{collision['kinds'][0]}:{collision['surface'].lower()}",
                kind=collision["kinds"][0],
                entry_type="gray",
                gloss=(glosses or {}).get(anchor),
                candidate_vector=vec,
                nearest=[],
                payload={
                    "name": collision["surface"],
                    "kind": collision["kinds"][0],
                    "surface": collision["surface"],
                    "reason": f"surface {collision['surface']!r} names "
                    f"{len(collision['iris'])} concepts of the same kind "
                    f"({collision['effect']}); resolving it returns an arbitrary one",
                    "effect": collision["effect"],
                    "sources": collision["sources"],
                    "minted_iris": collision["iris"],
                },
                conn=conn,
            )
            if rid is not None:
                n_rows += 1
            # review_id None means the queue already held this one (enqueue_review is idempotent
            # per (candidate_uid, entry_type)); said outright so it does not read as a failed
            # write. n_review_rows counts growth, this list counts what is in the queue.
            queued_collisions.append(
                {
                    "surface": collision["surface"],
                    "iris": collision["iris"],
                    "review_id": rid,
                    "already_queued": rid is None,
                }
            )

        # One name minted as two kinds. Both stay: they are separate IRIs by design, and
        # a resolver told which kind it wants finds the right one. What is wrong is the modelling —
        # usesWeapon cannot be both a relation to a Weapon and a string attribute — and nothing in
        # the pipeline can settle that, because either reading is legitimate for some term
        # (hasLegitimacyStatus really may be data). So the contradiction itself is queued.
        #
        # Its own entry_type rather than another GRAY: the question is different ("relation or
        # attribute", not "same concept as that one"), and the queue's unique key is
        # (candidate_uid, entry_type) — filing it as GRAY would collide with a real gray item on
        # the same candidate and one of the two would vanish.
        for conflict in kind_conflicts or []:
            # The lexicographically first side owns the row so a rerun does not stack duplicates;
            # both sides travel in the payload, so which one holds the key carries no meaning.
            anchor = min(side["uid"] for side in conflict["sides"])
            vec = vectors.get(anchor)
            if vec is None:
                # On the cold path every conflict side comes from this batch and the
                # withheld vectors are restored above, so this defensive branch is
                # unreachable normally. Reaching it means queueing material was lost; the
                # old wording ("no claimant in this batch") also covered the case the
                # withholding filter deleted, hiding the real cause in the log.
                logger.warning(
                    "generate: kind conflict on %r has no frozen vector to queue with; "
                    "reported without queueing (unexpected on the cold path)",
                    conflict["name"],
                )
                continue
            anchor_kind = next(
                s["kind"] for s in conflict["sides"] if s["uid"] == anchor
            )
            # Compute the minting state once so reason and minted_iris come from the same
            # value ([None, None] in the withheld branch; the prose must not contradict
            # it).
            minted = [iri_by_uid.get(s["uid"]) for s in conflict["sides"]]
            rid = repo.enqueue_review(
                base_iri=base_iri,
                candidate_uid=anchor,
                kind=anchor_kind,
                entry_type="kind_conflict",
                gloss=(glosses or {}).get(anchor),
                candidate_vector=vec,
                nearest=[],
                payload={
                    "name": conflict["name"],
                    "names": conflict["names"],
                    "kind": anchor_kind,
                    # A sentence computed from the actual minting state
                    # (kind_conflict_reason); while it was the literal "both are minted",
                    # the withheld branch's payload contradicted itself.
                    "reason": kind_conflict_reason(conflict, minted),
                    "kinds": conflict["kinds"],
                    # Both ranges: this is what the decision turns on. Demoting an object
                    # property to data turns "uses a weapon" into "a weapon is a string".
                    #
                    # `unresolved_cq_ids` marks the cited questions that are not in the
                    # store, so looking one up returns nothing. The citation itself is kept
                    # — that the upstream judged on that question is part of the evidence —
                    # but a reader has to be able to tell "no such question here" from "the
                    # lookup failed". A batch ingested before its CQs are registered puts
                    # every id in this list (n_cq_missing counts the same event in total).
                    "sides": [
                        {
                            **side,
                            "iri": iri_by_uid.get(side["uid"]),
                            "unresolved_cq_ids": [
                                q for q in side["source_cq_ids"] if cqs.get(q) is None
                            ],
                        }
                        for side in conflict["sides"]
                    ],
                    "minted_iris": minted,
                },
                conn=conn,
            )
            if rid is not None:
                n_rows += 1
            # Same shape as queued_collisions and queued_pref_labels above, for the same reason:
            # review_id None here means the queue already held this contradiction, not that the
            # write failed. A component's uid is `kind:name`, so the anchor is the same candidate
            # in every document — a term two documents each describe two ways files once and is an
            # idempotent no-op after, which without this flag reads as a conflict that never
            # reached a human. This list was the one of the three that never got the flag.
            queued_kind_conflicts.append(
                {
                    "name": conflict["name"],
                    "review_id": rid,
                    "already_queued": rid is None,
                }
            )

        # Mirror the graph's structure into the relational tables, inside the same transaction so
        # the stored edges match the snapshot being committed.
        if topology is not None:
            topo_counts = repo.replace_topology(
                topology,
                base_iri=base_iri,
                edge_sources=collect_edge_sources(concepts),
                conn=conn,
            )

    return (
        cq_linked,
        cq_missing,
        topo_counts,
        n_queued,
        n_rows,
        queued_collisions,
        queued_kind_conflicts,
        n_no_source,
        n_surface_skipped,
        queued_pref_labels,
    )
