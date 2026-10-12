"""Tier orchestration: exact matches, then the two suggestion channels."""

import logging
from dataclasses import dataclass
from typing import Protocol

from rapidfuzz import fuzz

from app.clients.pg import to_literal
from app.core.db import QueryTimeout, SchemaMissing
from app.services.iri import UnroutableIriError, graph_id_from_iri
from app.services.key_index import KeyIndex
from app.services.normalize import normalize_term, separator_variants
from app.services.repository import Repository

logger = logging.getLogger(__name__)

LABEL_PRECEDENCE = {"PREF": 0, "ALTER": 1, "HIDDEN": 2}

# A suggestion channel degrades (drops out of tiers_executed) rather than
# failing the request on these, and only these: a statement timeout, a missing
# relation (the ontorouter schema was never applied — only SEMANTIC_SQL reads
# it, so this cannot mask a broken exact tier), any
# socket-level failure (ConnectionError and TimeoutError are both OSError
# subclasses), or a non-2xx / malformed response from TEI, which
# ``EmbedClient`` raises as ValueError. Anything else — a bug in our own code —
# still propagates, so it surfaces as a 500 instead of a silently absent
# channel.
DEGRADABLE_ERRORS = (QueryTimeout, SchemaMissing, OSError, ValueError)


@dataclass(frozen=True)
class ResultEntry:
    """A confirmed pointer into a graph.

    Carries no score: exactness is the guarantee. ``pref_label`` names the
    term's preferred surface form, which ``label_form`` tells you the match
    itself often was not.
    """

    graph_id: str
    iri: str
    term_type: str
    label_form: str
    pref_label: str | None


@dataclass(frozen=True)
class SuggestionEntry:
    """A non-confirmed pointer. Always carries a score on its channel's scale."""

    graph_id: str
    iri: str
    term_type: str
    label_form: str
    pref_label: str | None
    score: float


def _sort_key(entry: ResultEntry | SuggestionEntry) -> tuple[float, int, str, str]:
    """Order candidates for the same key, best first.

    One ordering serves all three channels, so a scored channel and an unscored
    one cannot drift apart on how they break a tie. Score descending comes
    first where a score exists (a ``ResultEntry`` has none and contributes a
    constant), then label precedence, then graph_id, then IRI.

    Args:
        entry: A result or suggestion candidate.

    Returns:
        A tuple that sorts the better candidate first.
    """
    score = entry.score if isinstance(entry, SuggestionEntry) else 0.0
    return (
        -score,
        LABEL_PRECEDENCE.get(entry.label_form, len(LABEL_PRECEDENCE)),
        entry.graph_id,
        entry.iri,
    )


def _keep_best[Entry: (ResultEntry, SuggestionEntry)](
    entries: list[Entry],
) -> dict[tuple[str, str], Entry]:
    """Collapse candidates to one per (graph_id, IRI) under ``_sort_key``.

    Args:
        entries: Candidates, in arbitrary order.

    Returns:
        The surviving candidate per key. Independent of arrival order.
    """
    best: dict[tuple[str, str], Entry] = {}
    for entry in entries:
        key = (entry.graph_id, entry.iri)
        current = best.get(key)
        if current is None or _sort_key(entry) < _sort_key(current):
            best[key] = entry
    return best


def exact_tier(repo: Repository, key: str, max_words: int) -> list[ResultEntry]:
    """Resolve confirmed matches for a normalized key.

    Probes the separator variants against the upstream ``lower(surface_text)``
    index; keys longer than ``max_words`` fall back to normalizing in SQL.

    Args:
        repo: The database repository.
        key: The normalized lookup key.
        max_words: Word cap for variant expansion.

    Returns:
        Deduplicated entries ordered by label precedence, then graph, then IRI.
    """
    variants = separator_variants(key, max_words)
    rows = (
        repo.exact(variants) if variants is not None else repo.exact_by_normalized(key)
    )
    entries: list[ResultEntry] = []
    for row in rows:
        try:
            graph_id = graph_id_from_iri(row.iri)
        except UnroutableIriError:
            # v1 assumption: every term IRI is document-derived. A row that
            # breaks it is skipped here and caught loudly by the provenance
            # regression rather than silently mis-routed.
            continue
        entries.append(
            ResultEntry(graph_id, row.iri, row.term_type, row.form_type, row.pref_label)
        )
    return sorted(_keep_best(entries).values(), key=_sort_key)


def lexical_tier(
    repo: Repository,
    index: KeyIndex,
    key: str,
    k: int,
    min_score: float,
    max_words: int,
    candidate_keys: int,
    exclude: frozenset[tuple[str, str]] = frozenset(),
) -> list[SuggestionEntry]:
    """Produce edit-distance suggestions for a normalized key.

    Candidates come from scanning the in-memory key index with rapidfuzz, not
    from a prefix query: a prefix prefetch can only find strings the query is a
    prefix of, so it never produced a candidate for a misspelling. The keys it
    picks are then expanded into their separator spellings and fetched through
    the same indexed statement the exact tier uses.

    Args:
        repo: The database repository.
        index: The in-memory key index.
        key: The normalized lookup key.
        k: Maximum entries to return.
        min_score: Admission threshold on the [0, 1] similarity scale.
        max_words: Word cap for variant expansion.
        candidate_keys: How many keys to take from the index before fetching.
        exclude: ``(graph_id, IRI)`` keys the caller already holds as confirmed
            results. Dropped before ranking so ``k`` buys alternatives rather
            than echoing back what the consumer was already given.

    Returns:
        Deduplicated suggestions, score descending, capped at k.
    """
    matches = index.nearest(key, candidate_keys, min_score)
    if not matches:
        return []
    variants = [
        variant
        for match in matches
        for variant in (separator_variants(match, max_words) or [match])
    ]
    rows = repo.exact(variants)

    scored: list[SuggestionEntry] = []
    for row in rows:
        try:
            graph_id = graph_id_from_iri(row.iri)
        except UnroutableIriError:
            continue
        if (graph_id, row.iri) in exclude:
            continue
        score = fuzz.ratio(key, row.surface_text.lower()) / 100.0
        if score < min_score:
            continue
        scored.append(
            SuggestionEntry(
                graph_id, row.iri, row.term_type, row.form_type, row.pref_label, score
            )
        )
    return sorted(_keep_best(scored).values(), key=_sort_key)[:k]


class Embedder(Protocol):
    """The embedding surface the semantic tier depends on."""

    def embed_one(self, text: str) -> list[float]: ...


def semantic_tier(
    repo: Repository,
    vector_literal: str,
    model_id: str,
    k: int,
    min_score: float,
    exclude: frozenset[tuple[str, str]] = frozenset(),
) -> list[SuggestionEntry]:
    """Produce embedding-similarity suggestions.

    Args:
        repo: The database repository.
        vector_literal: The query vector as a pgvector literal.
        model_id: The configured embedding model id.
        k: Maximum entries to return.
        min_score: Admission threshold on the [0, 1] cosine scale.
        exclude: ``(graph_id, IRI)`` keys the caller already holds as confirmed
            results. Dropped before ranking so ``k`` buys alternatives.

    Returns:
        Deduplicated suggestions, score descending, capped at k.
    """
    scored: list[SuggestionEntry] = []
    for row in repo.semantic(vector_literal, model_id, k * 4):
        try:
            graph_id = graph_id_from_iri(row.iri)
        except UnroutableIriError:
            continue
        if (graph_id, row.iri) in exclude:
            continue
        # Cosine similarity of L2-normalized vectors is in [-1, 1]; the contract
        # publishes a [0, 1] scale, so a negative similarity clamps to zero.
        score = max(0.0, min(1.0, row.score))
        if score < min_score:
            continue
        scored.append(
            SuggestionEntry(
                graph_id, row.iri, row.term_type, row.form_type, row.pref_label, score
            )
        )
    return sorted(_keep_best(scored).values(), key=_sort_key)[:k]


@dataclass(frozen=True)
class LookupOutcome:
    """The full answer for one term, before DTO serialization."""

    normalized_key: str
    results: list[ResultEntry]
    lexical: list[SuggestionEntry]
    semantic: list[SuggestionEntry]
    tiers_executed: list[str]


def lookup(
    repo: Repository,
    index: KeyIndex,
    term: str,
    k: int,
    depth: str,
    max_words: int,
    lexical_min_score: float,
    candidate_keys: int,
    embedder: Embedder | None = None,
    model_id: str | None = None,
    semantic_min_score: float = 0.0,
) -> LookupOutcome:
    """Run the tier pipeline for one term.

    Args:
        repo: The database repository.
        index: The in-memory key index backing the lexical channel.
        term: The raw consumer term.
        k: Per-channel suggestion cap.
        depth: Tier depth: 'exact', 'lexical' or 'semantic'.
        max_words: Word cap for variant expansion.
        lexical_min_score: Admission threshold for lexical suggestions.
        candidate_keys: How many keys the lexical channel takes from the index.
        embedder: The embedding client, or ``None`` when the semantic tier is
            unconfigured.
        model_id: The configured embedding model id, or ``None`` when
            unconfigured.
        semantic_min_score: Admission threshold for semantic suggestions.

    Returns:
        The results, both suggestion channels, and the tiers that actually ran.
        Neither suggestion channel repeats a confirmed result: a consumer that
        already holds those pointers is offered alternatives instead.
    """
    key = normalize_term(term)
    results = exact_tier(repo, key, max_words)
    executed = ["exact"]

    # Both suggestion channels run over the same key the exact tier matched, so
    # without this every exact match would also score at the top of the lexical
    # channel and spend the k budget echoing back what `results` already holds.
    confirmed = frozenset((entry.graph_id, entry.iri) for entry in results)

    lexical: list[SuggestionEntry] = []
    if depth in ("lexical", "semantic"):
        try:
            lexical = lexical_tier(
                repo,
                index,
                key,
                k,
                lexical_min_score,
                max_words,
                candidate_keys,
                exclude=confirmed,
            )
        except DEGRADABLE_ERRORS as exc:
            # The channel is dropped and omitted from tiers_executed; results
            # already succeeded and must still be served.
            logger.warning("lexical tier degraded, channel dropped: %r", exc)
        else:
            executed.append("lexical")

    semantic: list[SuggestionEntry] = []
    if depth == "semantic" and embedder is not None and model_id is not None:
        try:
            vector = embedder.embed_one(term)
            semantic = semantic_tier(
                repo,
                to_literal(vector),
                model_id,
                k,
                semantic_min_score,
                exclude=confirmed,
            )
        except DEGRADABLE_ERRORS as exc:
            # Timeout, connection failure or a non-2xx TEI response: drop the
            # channel, keep HTTP 200.
            logger.warning("semantic tier degraded, channel dropped: %r", exc)
        else:
            executed.append("semantic")

    return LookupOutcome(key, results, lexical, semantic, executed)
