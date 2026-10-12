"""Per-kind threshold calibration.

Positives come from curated ``duplicates`` (the only field that links distinct components).
Negatives are randomly sampled cross-concept pairs (PU-learning). ``tau_high`` is a low
percentile of positive cosines (precision-favoring for merges); ``tau_low`` an upper
percentile of negative cosines (below which pairs are confidently distinct). Calibration is
per ``kind`` because clustering is kind-partitioned; the class partition usually has too few
``duplicates`` positives and falls back to conservative seed defaults.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app.tboxcraft.models import Component
from app.tboxcraft.similarity import cosine

logger = logging.getLogger(__name__)

MIN_POSITIVES = 5
TAU_HIGH_PERCENTILE = 5.0  # accept only pairs at least as similar as 95% of positives
TAU_LOW_PERCENTILE = 95.0  # reject pairs less similar than 95% of negatives

# How far below the bulk of the positives a pair has to sit to be treated as miscurated rather
# than merely weak. In MAD units, so it is scale-free: what counts as "far" is set by how spread
# the sample itself is, not by a cosine constant that would need retuning per provider.
OUTLIER_MAD_FACTOR = 3.0
_MAD_TO_SIGMA = 1.4826  # MAD -> standard-deviation-equivalent for a normal sample


@dataclass(frozen=True)
class Thresholds:
    tau_high: float
    tau_low: float
    provider: str
    # "seed"       — too few positives to calibrate; the seed default is in force.
    # "calibrated" — computed from curated duplicates and used as computed.
    # "floored"    — computed, but looser than the seed, so the seed was kept. A distinct state
    #                rather than a silent clamp: it says calibration ran AND that its answer was
    #                rejected, which is the only signal that the positives are miscurated.
    source: str
    # How many curated duplicate pairs the threshold was computed from. Reported because the
    # threshold alone does not say how much to trust it: at MIN_POSITIVES the percentile is
    # essentially the minimum of the sample, so "calibrated from 5" and "calibrated from 50"
    # are different claims wearing the same number.
    n_positives: int = 0
    # [{a, b, cosine, median, cutoff}] — curated duplicate pairs left out of the sample as
    # miscurations. Reported rather than only dropped: the excluded pair is the actionable half of
    # the finding — it names a `duplicates` entry the producer got wrong, which no threshold number
    # can convey. A tuple because the dataclass is frozen; nothing hashes it.
    excluded_positives: tuple[dict, ...] = ()

    def band(self, sim: float) -> str:
        """Route a similarity into ``merge`` / ``new`` / ``gray``."""
        if sim >= self.tau_high:
            return "merge"
        if sim <= self.tau_low:
            return "new"
        return "gray"


def collect_duplicate_pairs(components: list[Component]) -> list[tuple[str, str]]:
    """Same-kind duplicate pairs as ``(uid, uid)`` from curated duplicates fields.

    ``duplicates`` reference component names; they are resolved within the referencing
    component's kind (names are unique per kind), and returned as kind-qualified uids so
    dual-kind names like ``hasTask`` never cross-link.
    """
    by_kind_name = {(c.kind, c.name): c for c in components}
    pairs: set[tuple[str, str]] = set()
    for c in components:
        for dup in c.dom_duplicates:
            other = by_kind_name.get((c.kind, dup))
            if other is not None and other.name != c.name:
                pairs.add(tuple(sorted((c.uid, other.uid))))  # dedupe reciprocals
    return sorted(pairs)


def sample_negative_pairs(
    names: list[str], n: int, rng: np.random.Generator
) -> list[tuple[str, str]]:
    """Randomly sample up to ``n`` distinct-name pairs (assumed distinct concepts)."""
    if len(names) < 2:
        return []
    pairs: set[tuple[str, str]] = set()
    max_pairs = len(names) * (len(names) - 1) // 2
    target = min(n, max_pairs)
    attempts = 0
    while len(pairs) < target and attempts < target * 20:
        i, j = rng.choice(len(names), size=2, replace=False)
        pairs.add(tuple(sorted((names[i], names[j]))))
        attempts += 1
    return sorted(pairs)


def reject_positive_outliers(
    scored: list[tuple[tuple[str, str], float]],
) -> tuple[list[float], list[dict]]:
    """Split scored positives into ``(kept cosines, [{a, b, cosine, median, cutoff}])``.

    ``duplicates`` is a curated claim that two components are the same concept, and some of those
    claims are of a different nature — a spelling variant, or a broader/narrower pair. Their cosine
    sits far below the genuine ones, and because ``TAU_HIGH_PERCENTILE`` is essentially the minimum
    at a small sample, ONE of them sets the threshold for the whole kind. That is what dragged
    tau_high down and chained every class into a single concept.

    Median and MAD rather than mean and standard deviation: the thing being removed is exactly the
    thing that would move a mean, so a mean-based rule is dragged by the outlier it is meant to
    catch. MAD tolerates up to half the sample being wrong.

    One-sided. A positive far ABOVE the rest is two components that really are the same thing, not
    evidence of miscuration — trimming it would raise the threshold on no evidence.

    Trimming is not conditioned on enough positives surviving it. The case that motivated this is
    five positives with one miscuration, so any such condition excuses exactly the sample it was
    written for. What survives may well be too small to calibrate from — and then the caller falls
    back to the seed and says ``source="seed"``, which is the honest reading: not enough evidence,
    rather than "calibration ran and its answer was rejected".

    ``MAD == 0`` declines to trim: half or more of the sample shares one cosine, so every deviation
    is infinitely many MADs and the rule would cut at the median, discarding half on no evidence.
    """
    if len(scored) < MIN_POSITIVES:
        return [c for _, c in scored], []
    values = np.asarray([c for _, c in scored], dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0.0:
        return list(values), []
    cutoff = median - OUTLIER_MAD_FACTOR * _MAD_TO_SIGMA * mad
    keep = [(pair, c) for pair, c in scored if c >= cutoff]
    excluded = [
        {
            "a": pair[0],
            "b": pair[1],
            "cosine": round(c, 4),
            "median": round(median, 4),
            "cutoff": round(cutoff, 4),
        }
        for pair, c in scored
        if c < cutoff
    ]
    return [c for _, c in keep], sorted(
        excluded, key=lambda r: (r["cosine"], r["a"], r["b"])
    )


def _threshold_from_pairs(
    vectors: dict[str, np.ndarray],
    positives: list[tuple[str, str]],
    provider: str,
    seed_high: float,
    seed_low: float,
    rng: np.random.Generator,
    neg_sample: int,
) -> Thresholds:
    scored = [
        ((a, b), cosine(vectors[a], vectors[b]))
        for a, b in positives
        if a in vectors and b in vectors
    ]
    if len(scored) < MIN_POSITIVES:
        return Thresholds(seed_high, seed_low, provider, "seed", len(scored))

    pos_cos, excluded = reject_positive_outliers(scored)
    if len(pos_cos) < MIN_POSITIVES:
        # Trimming can take the sample below what is worth calibrating from — the motivating case
        # is five positives with one miscuration. That is a seed run with a finding attached, not
        # a floored one: the seed is in force because the evidence ran out, not because the
        # computed answer was rejected.
        return Thresholds(
            seed_high, seed_low, provider, "seed", len(pos_cos), tuple(excluded)
        )
    if excluded:
        logger.warning(
            "calibration: excluded %d curated duplicate pair(s) from the positive sample — their "
            "cosine sits more than %.1f MAD below the median (e.g. %s); a duplicate claim that far "
            "from the rest is a miscuration to fix upstream, not a weak same-concept pair",
            len(excluded),
            OUTLIER_MAD_FACTOR,
            ", ".join(f"{r['a']}={r['b']} at {r['cosine']}" for r in excluded[:3]),
        )

    names = sorted(vectors)
    negatives = sample_negative_pairs(names, neg_sample, rng)
    neg_cos = [
        cosine(vectors[a], vectors[b])
        for a, b in negatives
        if a in vectors and b in vectors
    ]

    tau_high = float(np.percentile(pos_cos, TAU_HIGH_PERCENTILE))
    source = "calibrated"
    if tau_high < seed_high:
        # Calibration exists to sharpen the seed for a provider, not to loosen it. A computed
        # threshold below the conservative default means the positives are no longer describing
        # "the same concept" — one curated duplicate that is really a spelling variant is enough,
        # because at a small sample TAU_HIGH_PERCENTILE is essentially their minimum.
        #
        # The floor is not caution, it is proportionality. Merging is single-linkage (see
        # cluster.cluster_components), so a threshold somewhat too low does not over-merge
        # somewhat: it chains the partition into one concept. A handful of
        # class positives, one of them a casing pair at a low cosine, dragged tau_high down and
        # every class collapsed into one.
        logger.warning(
            "calibration: computed tau_high %.4f is below the %.4f seed over %d positives — "
            "keeping the seed; the positives are not behaving like same-concept pairs "
            "(a spelling variant curated as a duplicate does this)",
            tau_high,
            seed_high,
            len(pos_cos),
        )
        tau_high, source = seed_high, "floored"
    tau_low = float(np.percentile(neg_cos, TAU_LOW_PERCENTILE)) if neg_cos else seed_low
    # Keep the band non-inverted.
    tau_low = min(tau_low, tau_high)
    return Thresholds(
        tau_high, tau_low, provider, source, len(pos_cos), tuple(excluded)
    )


def calibrate(
    components: list[Component],
    vectors: dict[str, np.ndarray],
    provider: str,
    *,
    seed_high: float,
    seed_low: float,
    rng_seed: int = 0,
    neg_sample: int = 500,
) -> dict[str, Thresholds]:
    """Calibrate thresholds per ``kind``; unresolved kinds get seed defaults."""
    rng = np.random.default_rng(rng_seed)
    all_positives = collect_duplicate_pairs(components)
    result: dict[str, Thresholds] = {}
    for kind in sorted({c.kind for c in components}):
        kind_uids = {c.uid for c in components if c.kind == kind}
        kind_vectors = {u: v for u, v in vectors.items() if u in kind_uids}
        kind_positives = [
            (a, b) for a, b in all_positives if a in kind_uids and b in kind_uids
        ]
        result[kind] = _threshold_from_pairs(
            kind_vectors,
            kind_positives,
            provider,
            seed_high,
            seed_low,
            rng,
            neg_sample,
        )
    return result
