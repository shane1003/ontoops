"""DomainTermApproval layer.

A pure computation step: given the attach-only :class:`~tboxcraft.cluster.ClusterPlan`, judge
each cluster with the LLM verifier and classify it into a concrete outcome — ADD, SynonymADD,
UPDATE (IRI-renewal proposal), or GRAY (routed to human review). The verifier runs once per
attach and once per new cluster's representative; a multi-member ADD cluster additionally judges
each non-representative member to split it into AlterTerm (standard) vs HiddenTerm (non-standard).
This module performs **no DB writes**; enqueue + verdict-attach happen inside the
TboxMatcher's single post-gate transaction. Auto outcomes (ADD, confident SynonymADD) apply;
risky outcomes (UPDATE, DELETE, low-confidence attach) go to review with the verdict attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdflib import Graph

from app.tboxcraft.calibration import Thresholds, collect_duplicate_pairs
from app.tboxcraft.cluster import ClusterPlan
from app.tboxcraft.config import Config
from app.tboxcraft.llm.base import LLMProvider, Verdict
from app.tboxcraft.models import Component
from app.tboxcraft.pipeline.update import (
    _iri_exists,
    _is_name_derived,
    _pref_label,
    _property_compatible,
)
from app.tboxcraft.pipeline.verdict import surface_role as _role_for
from app.tboxcraft.represent import build_concept, canonical_member
from app.tboxcraft.similarity import cosine
from app.tboxcraft.verify_batch import run_batched_judgment

# Outcome kinds. DELETE is intentionally absent — it is human-initiated or produced by the
# orphan scan, never by approval routing.
OUTCOME_ADD = "add"
OUTCOME_SYNONYM_ADD = "synonym_add"
OUTCOME_UPDATE = "update"
OUTCOME_GRAY = "gray"


@dataclass
class ApprovedOutcome:
    """One routed cluster. ``representative`` is the concept-bearing member; ``members`` is the
    whole cluster for ADD (non-representatives become surface forms) or ``[component]`` for an
    attach. ``target_iri`` is the minted IRI (ADD) or the anchor IRI (attach)."""

    kind: str  # add | synonym_add | update | gray
    members: list[Component]
    representative: Component
    target_iri: str | None
    verdict: Verdict | None = None
    nearest: list[dict] = field(default_factory=list)
    band: str | None = None
    reason: str = ""
    # Per-member surface role decided by the verifier: uid -> 'alter' | 'hidden'. A standard term
    # form is an AlterTerm; a non-standard document artifact (right concept, messy form) is a
    # HiddenTerm kept for ABox mapping. Absent uid defaults to 'alter'.
    term_type_by_uid: dict[str, str] = field(default_factory=dict)

    @property
    def is_risky(self) -> bool:
        """Risky outcomes are enqueued for human review rather than auto-applied."""
        return self.kind in (OUTCOME_UPDATE, OUTCOME_GRAY)

    def term_type_for(self, uid: str) -> str:
        return self.term_type_by_uid.get(uid, "alter")


def _connected_by_duplicates(uids: list[str], duplicate_pairs: set) -> bool:
    """True when the curated duplicate edges alone connect every member into one component."""
    parent = {u: u for u in uids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    uset = set(uids)
    for a, b in duplicate_pairs:
        if a in uset and b in uset:
            parent[find(a)] = find(b)
    return len({find(u) for u in uids}) == 1


def _cluster_cohesive(
    cluster: list[Component], centered: dict, tau_high: float, duplicate_pairs: set
) -> bool:
    """A multi-member self-cluster is cohesive when it is trustworthy as one concept. Curated
    duplicates are authoritative: if they alone connect the whole cluster, trust it regardless of
    cosine (a curator asserting A=B=C need not be pairwise-similar in embedding space). Otherwise
    every member pair must be a curated duplicate or directly similar (cosine >= tau_high) — this
    catches an embedding over-merge where union-find chained A-B-C but cosine(A,C) < tau_high,
    which must go to review so a distinct concept is not silently absorbed under one IRI."""
    if len(cluster) < 2:
        return True
    uids = [c.uid for c in cluster]
    if _connected_by_duplicates(uids, duplicate_pairs):
        return True
    for i in range(len(uids)):
        for j in range(i + 1, len(uids)):
            a, b = uids[i], uids[j]
            if tuple(sorted((a, b))) in duplicate_pairs:
                continue
            va, vb = centered.get(a), centered.get(b)
            if va is None or vb is None or cosine(va, vb) < tau_high:
                return False
    return True


def approve(
    plan: ClusterPlan,
    components_by_uid: dict[str, Component],
    centered: dict,
    graph: Graph,
    config: Config,
    thresholds: dict[str, Thresholds],
    provider: LLMProvider | None,
    *,
    verify: bool = True,
    glosses: dict | None = None,
) -> list[ApprovedOutcome]:
    """Classify every attach and new cluster into an outcome (compute-only, no DB writes)."""
    outcomes: list[ApprovedOutcome] = []
    duplicate_pairs = set(collect_duplicate_pairs(list(components_by_uid.values())))

    # Judge everything first, then assemble. This used to call the verifier inline inside the loop
    # below, one candidate per call with nothing in flight beside it, so the whole update path ran
    # at concurrency 1 while the bootstrap ran at verify_concurrency — a small corpus ingested one
    # document at a time cost as long as a far larger one in one shot.
    #
    # Every participating candidate is judged, including the few whose cluster turns out to collide
    # on its slug and is routed to review without a verdict. Predicting that here would mean
    # duplicating the minting logic below, which is a worse trade than a rare spare judgment.
    verdicts: dict = {}
    if verify and provider is not None:
        nearest_by_uid = {
            att.uid: [
                {"uid": att.anchor_iri, "iri": att.anchor_iri, "score": att.score}
            ]
            for att in plan.attached
        }
        to_judge = [
            components_by_uid[att.uid]
            for att in plan.attached
            if att.uid in components_by_uid
        ]
        seen = {c.uid for c in to_judge}
        for cluster in plan.new_clusters:
            for member in cluster:
                if member.uid not in seen:
                    seen.add(member.uid)
                    to_judge.append(member)
        verdicts, _, _ = run_batched_judgment(
            to_judge,
            provider,
            cluster_size=config.verify_cluster_size,
            concurrency=config.verify_concurrency,
            glosses=glosses,
            nearest_by_uid=nearest_by_uid,
            domain=config.domain_description,
            stage="update",
        )

    def judge(component: Component) -> Verdict:
        if not verify or provider is None:
            return Verdict.unavailable("verifier disabled")
        # A candidate the verifier never answered for stays unavailable, exactly as the inline
        # call left it — the caller's degrade accounting reads that status.
        return verdicts.get(component.uid) or Verdict.unavailable(
            "verifier did not answer"
        )

    # Attaches to existing (live) anchors.
    for att in plan.attached:
        comp = components_by_uid[att.uid]
        anchor = att.anchor_iri
        nearest = [{"uid": anchor, "iri": anchor, "score": att.score}]
        verdict = judge(comp)

        if att.band == "gray":
            outcomes.append(
                ApprovedOutcome(
                    OUTCOME_GRAY,
                    [comp],
                    comp,
                    anchor,
                    verdict,
                    nearest,
                    "gray",
                    "low-confidence attach",
                )
            )
            continue

        # Confident attach (merge band): property gate, then UPDATE-vs-SynonymADD.
        if comp.is_property and not _property_compatible(
            comp, anchor, graph, config.base_iri
        ):
            outcomes.append(
                ApprovedOutcome(
                    OUTCOME_GRAY,
                    [comp],
                    comp,
                    anchor,
                    verdict,
                    nearest,
                    "gray",
                    "property domain/range mismatch",
                )
            )
            continue

        if (
            comp.correct_term
            and _is_name_derived(graph, anchor)
            and comp.correct_term != _pref_label(graph, anchor)
            and att.score >= config.tau_major
        ):
            outcomes.append(
                ApprovedOutcome(
                    OUTCOME_UPDATE,
                    [comp],
                    comp,
                    anchor,
                    verdict,
                    nearest,
                    "merge",
                    "candidate term more appropriate; IRI-renewal proposal",
                )
            )
        else:
            # The verifier splits a confident attach into AlterTerm (standard form) vs HiddenTerm
            # (non-standard document artifact) automatically — no human unless the pref changes.
            # Only a confident non-standard verdict buries the term (KTD: confidence floor).
            tt = _role_for(verdict, config.hidden_confidence_floor)
            outcomes.append(
                ApprovedOutcome(
                    OUTCOME_SYNONYM_ADD,
                    [comp],
                    comp,
                    anchor,
                    verdict,
                    nearest,
                    "merge",
                    "confident attach",
                    term_type_by_uid={comp.uid: tt},
                )
            )

    # New self-clusters (ADD candidates).
    minted_iris: set[str] = (
        set()
    )  # batch-local claims, so two sibling clusters can't mint one IRI
    for cluster in plan.new_clusters:
        rep = canonical_member(cluster)
        # Mint through the single source of truth for cluster IRIs: one IRI per cluster,
        # curated-term-wins representative, collision semantics — the same shape generate persists.
        minted = build_concept(cluster, config.base_iri).iri
        th = thresholds.get(rep.kind)
        tau_high = th.tau_high if th is not None else 1.01

        if _iri_exists(graph, minted) or minted in minted_iris:
            # A low-similarity homonym of an existing concept, or two distinct new clusters whose
            # names slugify to the same IRI — never overwrite/collapse a stable IRI;
            # let a human decide. build_concepts() raises on this at bootstrap; here we route to
            # review instead of silently merging the second cluster over the first.
            outcomes.append(
                ApprovedOutcome(
                    OUTCOME_GRAY,
                    cluster,
                    rep,
                    minted,
                    None,
                    [],
                    "new",
                    "slug collides with existing IRI",
                )
            )
            continue
        minted_iris.add(minted)

        if not _cluster_cohesive(cluster, centered, tau_high, duplicate_pairs):
            outcomes.append(
                ApprovedOutcome(
                    OUTCOME_GRAY,
                    cluster,
                    rep,
                    minted,
                    judge(rep),
                    [],
                    "new",
                    "borderline cluster cohesion",
                )
            )
            continue

        # The representative's forms are AlterTerms (it carries the PrefTerm). Each non-rep member
        # is verifier-split into alter/hidden — a non-standard member is a HiddenTerm on the new
        # concept, not a standard synonym. Singleton clusters incur no extra verifier calls.
        term_type_by_uid = {rep.uid: "alter"}
        for m in cluster:
            if m.uid == rep.uid:
                continue
            term_type_by_uid[m.uid] = _role_for(
                judge(m), config.hidden_confidence_floor
            )

        outcomes.append(
            ApprovedOutcome(
                OUTCOME_ADD,
                cluster,
                rep,
                minted,
                judge(rep),
                [],
                "new",
                "new concept",
                term_type_by_uid=term_type_by_uid,
            )
        )

    return outcomes
