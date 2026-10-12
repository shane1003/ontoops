"""Generate-mode clustering.

Partition components by ``kind``; within a partition, merge terms connected by a curated
``duplicates`` edge OR by cosine >= tau_high. Property merges additionally require
compatible ``domain``/``range``. ``synonyms`` are NOT merge edges — they are surface
variants routed to the mapping table. Cross-kind merges never happen, so ``hasTask``
as a data property and as an object property stay distinct.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from app.tboxcraft.calibration import Thresholds, collect_duplicate_pairs
from app.tboxcraft.models import Component
from app.tboxcraft.similarity import cosine


class _DisjointSet:
    def __init__(self, items: list[str]):
        self._parent = {i: i for i in items}

    def find(self, x: str) -> str:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)  # deterministic root

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for item in self._parent:
            out.setdefault(self.find(item), []).append(item)
        return out


@dataclass(frozen=True)
class MergeEdge:
    a: str
    b: str
    source: str  # "duplicate" | "embedding"
    cosine: float | None = None


@dataclass
class ClusterResult:
    clusters: list[list[Component]] = field(default_factory=list)
    merge_edges: list[MergeEdge] = field(default_factory=list)
    # Edges above tau_high that were NOT taken, because taking them would have put two members
    # into one concept without the two being similar to each other. Counted rather than dropped
    # quietly: a run where this is large is a run whose threshold is wrong, and that is worth
    # knowing before the concept count is read.
    n_chained_refused: int = 0
    # [{a, b, kind, reason}] — curated duplicate claims the property gate would not let through.
    # The gate is right (merging properties whose domain/range differ is unsafe), but the curator
    # said these are one concept and the pipeline said otherwise, so somebody has to be told.
    refused_duplicates: list[dict] = field(default_factory=list)


def properties_compatible(a: Component, b: Component) -> bool:
    """Two properties may merge only when domain and range match."""
    return a.domain == b.domain and set(a.range_values()) == set(b.range_values())


def _groups_cohesive(
    left: list[str], right: list[str], vectors, tau_high: float, duplicate_pairs: set
) -> bool:
    """May these two groups become one concept?

    Only when every pair drawn across them is either a curated duplicate or directly similar.
    Pairs *within* each group are not re-checked: each group is already cohesive, or is held
    together by curated duplicates, which are authoritative on their own (a curator asserting
    A=B=C need not be pairwise similar in embedding space).

    This is the same rule ``approval._cluster_cohesive`` applies on the update path.
    """
    for a in left:
        for b in right:
            if tuple(sorted((a, b))) in duplicate_pairs:
                continue
            va, vb = vectors.get(a), vectors.get(b)
            if va is None or vb is None or cosine(va, vb) < tau_high:
                return False
    return True


def cluster_components(
    components: list[Component],
    vectors: dict[str, np.ndarray],
    thresholds: dict[str, Thresholds],
) -> ClusterResult:
    result = ClusterResult()
    by_uid = {c.uid: c for c in components}
    duplicate_pairs = set(collect_duplicate_pairs(components))

    for kind in sorted({c.kind for c in components}):
        kind_comps = sorted(
            (c for c in components if c.kind == kind), key=lambda c: c.name
        )
        uids = [c.uid for c in kind_comps]
        ds = _DisjointSet(uids)
        tau_high = thresholds[kind].tau_high if kind in thresholds else 1.01

        # Curated duplicate edges (property gate still applies).
        for a, b in sorted(duplicate_pairs):
            if a in by_uid and b in by_uid and by_uid[a].kind == kind:
                if by_uid[a].is_property and not properties_compatible(
                    by_uid[a], by_uid[b]
                ):
                    # Not a merge, and not nothing either. A curator asserting "these are the same
                    # concept" for two properties whose domain/range disagree is a contradiction
                    # this layer cannot settle: merging them would erase a real modelling
                    # difference, and dropping the claim mints two concepts that then fight over
                    # the same surface string. The update path routes exactly this to review
                    # (approval, "property domain/range mismatch"); recorded here so the
                    # bootstrap can do the same instead of skipping in silence.
                    result.refused_duplicates.append(
                        {
                            "a_uid": a,
                            "b_uid": b,
                            "a": by_uid[a].name,
                            "b": by_uid[b].name,
                            "kind": kind,
                            "reason": "property domain/range mismatch",
                        }
                    )
                    continue
                ds.union(a, b)
                result.merge_edges.append(MergeEdge(a, b, "duplicate"))

        # Embedding edges above tau_high, strongest first, and only where the two sides are
        # similar to *each other* — not merely each similar to something in between.
        #
        # Union-find alone is single linkage, so A~B and B~C put A and C under one IRI however
        # far apart they are. That does not degrade gently. A tau_high that was too low
        # made a few percent of class pairs into edges and every class landed in a single
        # concept; even a moderately higher one left most of them there. Requiring the whole
        # merged group to hold together bounds the damage from a wrong threshold to the pairs the
        # threshold actually accepted, instead of letting it propagate along paths.
        #
        # Strongest-first with ties broken on uid so the greedy choice is fixed: without an order
        # this becomes run-dependent, and two runs of one batch stop being comparable.
        candidates = []
        for i in range(len(uids)):
            for j in range(i + 1, len(uids)):
                a, b = uids[i], uids[j]
                if a not in vectors or b not in vectors:
                    continue
                sim = cosine(vectors[a], vectors[b])
                if sim < tau_high:
                    continue
                if by_uid[a].is_property and not properties_compatible(
                    by_uid[a], by_uid[b]
                ):
                    continue
                candidates.append((sim, a, b))
        candidates.sort(key=lambda t: (-t[0], t[1], t[2]))

        members = ds.groups()
        for sim, a, b in candidates:
            ra, rb = ds.find(a), ds.find(b)
            if ra == rb:
                continue
            if not _groups_cohesive(
                members[ra], members[rb], vectors, tau_high, duplicate_pairs
            ):
                result.n_chained_refused += 1
                continue
            ds.union(a, b)
            merged = members.pop(ra) + members.pop(rb)
            members[ds.find(a)] = merged
            result.merge_edges.append(MergeEdge(a, b, "embedding", round(sim, 4)))

        for _root, group_uids in sorted(ds.groups().items()):
            cluster = sorted((by_uid[u] for u in group_uids), key=lambda c: c.name)
            result.clusters.append(cluster)

    result.clusters.sort(key=lambda cl: cl[0].name)
    return result


# --- Attach-only clustering ---


@dataclass(frozen=True)
class AttachDecision:
    """A new component attached to an existing (live) anchor concept.

    ``band`` is ``"merge"`` for a confident attach (>= tau_high, SynonymADD-bound) or
    ``"gray"`` for a low-confidence attach (gray band) that must go to human review.
    """

    uid: str
    anchor_iri: str
    band: str  # "merge" | "gray"
    score: float
    # "vector" — the anchor was the nearest neighbour and its cosine cleared tau_high.
    # "name"   — the anchor is written exactly this candidate's name. ``score`` still carries the
    #            real cosine (it gates the IRI-renewal branch downstream), so it will often be low;
    #            that is the finding, not a defect.
    matched_on: str = "vector"


@dataclass
class ClusterPlan:
    """Result of attach-only clustering: attaches to frozen anchors plus new self-clusters."""

    attached: list[AttachDecision] = field(default_factory=list)
    new_clusters: list[list[Component]] = field(default_factory=list)
    # Why the gray/new boundary fell where it did, so the rule stays checkable after this run.
    # {"0.5-0.6": n, ...} over every candidate's top-1 score, plus the counts each branch took.
    nearest_score_histogram: dict = field(default_factory=dict)
    n_attached_by_margin: int = 0  # top-1 stood out from the rest of its neighbourhood
    n_new_background: int = 0  # over tau_low but indistinct — these used to all be gray
    merge_edges: list[MergeEdge] = field(default_factory=list)


def _histogram(scores: list[float]) -> dict:
    """Top-1 scores in 0.1 buckets. The shape of this is the evidence for where the boundary
    belongs; a single mean would hide that the mass sits in one band."""
    buckets = Counter(
        f"{lo:.1f}-{lo + 0.1:.1f}"
        for lo in (min(max(int(s * 10), 0), 9) / 10 for s in scores)
    )
    return dict(sorted(buckets.items()))


def attach_and_cluster(
    components: list[Component],
    centered_vectors: dict[str, np.ndarray],
    retriever,
    thresholds: dict[str, Thresholds],
    *,
    attach_margin: float = 0.05,
) -> ClusterPlan:
    """Attach-only clustering: existing DB concepts are frozen anchors.

    For each new component, look up its kind-filtered nearest *live* anchor and band the top
    score with the per-kind :class:`Thresholds`: ``merge`` -> confident attach,
    ``gray`` -> low-confidence attach (review), ``new`` -> leftover. Leftovers self-cluster
    among themselves via :func:`cluster_components` (existing union-find, kind partition, and
    property domain/range gate) so each new cluster becomes one ADD candidate. Existing
    anchors never enter the union-find, so they are never re-partitioned, merged, or split.

    ``retriever`` must expose ``nearest(vector, kind) -> list`` of matches carrying ``.iri``
    and ``.score`` (e.g. :class:`~tboxcraft.pipeline.update.DbRetriever`). ``centered_vectors``
    are already centered into the stored space; the retriever does not re-center here.
    """
    plan = ClusterPlan()
    leftovers: list[Component] = []
    scores: list[float] = []
    exact_of = getattr(retriever, "exact", None)
    for c in sorted(components, key=lambda c: (c.kind, c.name)):
        vec = centered_vectors.get(c.uid)
        th = thresholds.get(c.kind)
        nearest = retriever.nearest(vec, c.kind) if vec is not None else []
        top = nearest[0].score if nearest else 0.0
        # Every candidate that had a neighbourhood, including the name-matched ones — the
        # histogram is the evidence for where the boundary belongs, and leaving out the branch
        # that already resolves would flatter it.
        if nearest:
            scores.append(top)

        # An existing term written exactly this name is the same term, whatever the cosine says.
        # The candidate's vector is its gloss, and a gloss is assembled from the CQs the component
        # arrived with, so one term described by two documents embeds far apart — two terms both
        # named ``Activity`` came out under any usable tau_high. Every candidate then
        # went to review instead of merging, and a second document grew the queue rather than the
        # TBox. This is the same call the cold path already makes when it folds names differing
        # only in case (an identity fact, not a similarity question); the warm path had no
        # equivalent. Ambiguous names resolve to None upstream and fall through to similarity.
        anchor = exact_of(c.name, c.kind) if exact_of is not None else None
        if anchor is not None:
            score = next((m.score for m in nearest if m.iri == anchor), top)
            plan.attached.append(
                AttachDecision(c.uid, anchor, "merge", score, matched_on="name")
            )
            continue

        band = th.band(top) if th is not None else "new"
        if band == "merge" and nearest:
            plan.attached.append(AttachDecision(c.uid, nearest[0].iri, band, top))
            continue
        if band == "gray" and nearest:
            # Over tau_low, under tau_high. Whether that means "near something" depends on how the
            # rest of the neighbourhood looks: in a store of a few thousand terms almost every
            # candidate has SOMETHING at 0.6-0.8, and routing all of those to review is what took
            # the gray share past half and stranded the individuals whose classes never got minted.
            # No absolute exception here. One used to attach anything within attach_margin of
            # tau_high whatever the neighbourhood looked like, as a guard against a family of
            # near-duplicates flattening the margin. Over successive documents that branch
            # grew steeply while the margin branch stayed flat and then fell, so it
            # was not catching families — it was catching the background, because
            # ``tau_high - attach_margin`` is an absolute threshold compared against a maximum
            # over a growing store, which is the very drift the margin rule exists to remove.
            # The family it guarded is handled a band higher: three terms at 0.87 to a candidate
            # are over tau_high and merge, and exact-name proliferation is caught by name matching.
            runner_up = nearest[1].score if len(nearest) > 1 else 0.0
            if top - runner_up >= attach_margin:
                plan.n_attached_by_margin += 1
            else:
                # Closest of many equally-mediocre neighbours — background, not a match.
                plan.n_new_background += 1
                leftovers.append(c)
                continue
            plan.attached.append(AttachDecision(c.uid, nearest[0].iri, band, top))
            continue
        leftovers.append(c)  # "new" (or no anchors / no threshold) -> self-clustering

    plan.nearest_score_histogram = _histogram(scores)
    if leftovers:
        sub = cluster_components(leftovers, centered_vectors, thresholds)
        plan.new_clusters = sub.clusters
        plan.merge_edges = sub.merge_edges
    return plan
