"""Grouping concepts into verifier batches.

The verifier is asked about one candidate per call, so a bootstrap cost one LLM round trip per
concept. Judging several at once divides the call count by the batch size. Which concepts share a
call is not arbitrary: a batch is drawn from the embedding neighbourhood, so the terms judged
together are the ones that actually inform each other, and it never spans kinds — whether a class
is the official term and whether a data property is are not the same question, and mixing them
spends context on candidates that cannot disambiguate one another.

This is a batching decision, not a clustering one: ``cluster.py`` decides which components *are*
the same concept, and its answer is already fixed by the time anything here runs. Nothing in this
module can merge, split, or rename a concept.
"""

from __future__ import annotations

import logging

from app.tboxcraft.similarity import cosine

logger = logging.getLogger(__name__)


def canonical_member(concept):
    """The member a concept's verdict is about — the one that gave it its name."""
    return next(
        (m for m in concept.members if m.name == concept.canonical_name),
        concept.members[0],
    )


def group_for_verification(
    concepts: list, vectors: dict | None, max_size: int
) -> list[list]:
    """Partition ``concepts`` into batches of at most ``max_size``, deterministically.

    ``vectors`` maps component uid -> centered embedding. A concept whose canonical member has no
    vector is still batched, just by kind and IRI order rather than by neighbourhood — a missing
    vector must not leave a concept unverified.

    ``max_size <= 1`` puts every concept in its own batch, which is the un-batched path exactly.
    Order is fixed by IRI and ties break on IRI, so the same input yields the same batches on
    every run; a batching that shifted run to run would make two runs incomparable for no reason.
    """
    if max_size <= 1:
        return [[c] for c in concepts]
    batches: list[list] = []
    for kind in sorted({c.kind for c in concepts}):
        of_kind = sorted((c for c in concepts if c.kind == kind), key=lambda c: c.iri)
        batches.extend(_neighbourhoods(of_kind, vectors, max_size))
    return batches


def _vector_for(concept, vectors: dict | None):
    if not vectors:
        return None
    return vectors.get(canonical_member(concept).uid)


def _neighbourhoods(concepts: list, vectors: dict | None, max_size: int) -> list[list]:
    """Greedy nearest-neighbour batches: take the first unbatched concept, then the closest
    unbatched concepts to it until the batch is full."""
    remaining = list(concepts)
    batches: list[list] = []
    while remaining:
        seed = remaining.pop(0)
        seed_vec = _vector_for(seed, vectors)
        if seed_vec is None:
            picked = remaining[: max_size - 1]
        else:
            scored = []
            for other in remaining:
                other_vec = _vector_for(other, vectors)
                if other_vec is not None:
                    scored.append((-cosine(seed_vec, other_vec), other.iri, other))
            # Ties break on IRI so equally-similar neighbours are picked in a fixed order.
            scored.sort(key=lambda s: (s[0], s[1]))
            picked = [s[2] for s in scored[: max_size - 1]]
            if not picked:  # nothing else in this kind carries a vector
                picked = remaining[: max_size - 1]
        for concept in picked:
            remaining.remove(concept)
        batches.append([seed, *picked])
    return batches


def run_batched_judgment(
    components: list,
    provider,
    *,
    cluster_size: int,
    concurrency: int,
    glosses: dict | None = None,
    nearest_by_uid: dict | None = None,
    domain: str | None = None,
    stage: str = "verify",
) -> tuple[dict, int, int]:
    """Judge ``components`` with fan-out and batching. Returns ``({uid: Verdict}, batches, calls)``.

    Shared rather than reimplemented per pipeline. The bootstrap had this and the update path did
    not: ``approval`` called the verifier inline inside its outcome loop, so the warm path ran at
    concurrency 1 for its whole length while the cold path ran at 32, so a small corpus ingested
    incrementally took as long as a far larger one in one shot. Two copies of a fan-out policy is
    how that gap reappears, so there is one.

    A verifier that stops answering must not be hammered once per candidate. The stop takes a
    *streak* of batches that came back with nothing usable rather than a single failure: with N
    calls in flight the first failure arrives next to N-1 others that may well have answered.
    Batches already running are drained rather than abandoned — they are paid for either way, and
    reading them keeps both the verdicts and the call count honest.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from app.tboxcraft.verify_term import _DEFAULT_DOMAIN, verify_terms

    domain = domain or _DEFAULT_DOMAIN
    size = max(1, cluster_size)
    batches = [components[i : i + size] for i in range(0, len(components), size)]
    if not batches:
        return {}, 0, 0

    workers = max(1, min(concurrency, len(batches)))
    verdicts: dict = {}
    streak = calls = 0
    stopped = False
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="verify") as pool:
        pending = {
            pool.submit(
                verify_terms,
                batch,
                provider,
                glosses=glosses,
                nearest_by_uid=nearest_by_uid,
                domain=domain,
            ): batch
            for batch in batches
        }
        for future in as_completed(pending):
            if future.cancelled():
                continue
            judged, n_calls = (
                future.result()
            )  # verify_terms degrades rather than raising
            calls += n_calls
            answered = 0
            for member, verdict in zip(pending[future], judged):
                if verdict.status == "unavailable":
                    continue
                verdicts[member.uid] = verdict
                answered += 1
            streak = 0 if answered else streak + 1
            if streak >= workers and not stopped:
                stopped = True
                queued = sum(1 for f in pending if f.cancel())
                logger.warning(
                    "%s: %d consecutive verifier batches came back empty — cancelling the %d "
                    "still queued",
                    stage,
                    streak,
                    queued,
                )
    return verdicts, len(batches), calls
