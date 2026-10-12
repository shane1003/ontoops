"""Lookup endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.clients.embed import embed_client
from app.core.config import Settings, get_settings
from app.core.db import QueryTimeout
from app.schemas.lookup import (
    TERM_MAX_LENGTH,
    BatchLookupRequest,
    BatchLookupResponse,
    LookupMeta,
    LookupResponse,
    PointerEntry,
    ScoredEntry,
    Suggestions,
    TierDepth,
)
from app.services.key_index import KeyIndex
from app.services.repository import PostgresRepository, Repository
from app.services.routing import Embedder, LookupOutcome, lookup

router = APIRouter()

TermParam = Annotated[str, Query(min_length=1, max_length=TERM_MAX_LENGTH)]
KParam = Annotated[int, Query(ge=1, le=50)]


def get_repository(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> Repository:
    """Provide the request-scoped repository.

    Args:
        request: The incoming request, carrying app state.
        settings: Injected settings.

    Returns:
        The Postgres-backed repository.
    """
    return PostgresRepository(request.app.state.db, timeout_s=settings.pg_timeout_s)


def get_key_index(request: Request) -> KeyIndex:
    """Provide the process-wide fuzzy key index.

    Args:
        request: The incoming request, carrying app state.

    Returns:
        The index built at start-up and refreshed on an interval.
    """
    index: KeyIndex = request.app.state.key_index
    return index


def get_embedder(
    settings: Annotated[Settings, Depends(get_settings)],
    tiers: TierDepth = TierDepth.SEMANTIC,
) -> Embedder | None:
    """Provide the TEI client, or None when the semantic tier will not run.

    Args:
        settings: Injected settings.
        tiers: The requested tier depth. Only the semantic depth uses an
            embedder, so shallower requests never build one.

    Returns:
        An embedding client, or None.
    """
    if tiers is not TierDepth.SEMANTIC or not settings.semantic_enabled:
        return None
    return embed_client(str(settings.embed_url), settings.embed_timeout_s)


def _to_response(term: str, outcome: LookupOutcome) -> LookupResponse:
    return LookupResponse(
        term=term,
        normalized_key=outcome.normalized_key,
        results=[PointerEntry(**vars(e)) for e in outcome.results],
        suggestions=Suggestions(
            lexical=[ScoredEntry(**vars(e)) for e in outcome.lexical],
            semantic=[ScoredEntry(**vars(e)) for e in outcome.semantic],
        ),
        meta=LookupMeta(tiers_executed=outcome.tiers_executed),
    )


def _run(
    repo: Repository,
    index: KeyIndex,
    term: str,
    k: int,
    tiers: TierDepth,
    settings: Settings,
    embedder: Embedder | None,
) -> LookupResponse:
    # Query(min_length=1) accepts a whitespace-only term, so the blank check is
    # enforced here for GET; the batch body enforces it in its validator.
    if not term.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="term must not be empty or whitespace-only",
        )
    try:
        outcome = lookup(
            repo,
            index,
            term,
            k=k,
            depth=tiers.value,
            max_words=settings.exact_variant_max_words,
            lexical_min_score=settings.lexical_min_score,
            candidate_keys=settings.lexical_candidate_keys,
            embedder=embedder,
            model_id=settings.embedding_model_id,
            semantic_min_score=settings.semantic_min_score,
        )
    except QueryTimeout as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the exact tier exceeded the configured statement timeout",
        ) from exc
    return _to_response(term, outcome)


@router.get("/lookup", response_model=LookupResponse)
def lookup_one(
    term: TermParam,
    repo: Annotated[Repository, Depends(get_repository)],
    index: Annotated[KeyIndex, Depends(get_key_index)],
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[Embedder | None, Depends(get_embedder)],
    k: KParam = 5,
    tiers: TierDepth = TierDepth.SEMANTIC,
) -> LookupResponse:
    """Route a single term to the graphs and IRIs that hold it.

    Args:
        term: The term to look up (1-256 characters, not blank).
        repo: Injected repository.
        index: Injected fuzzy key index.
        settings: Injected settings.
        embedder: Injected embedding client, or ``None`` when unconfigured.
        k: Per-channel suggestion cap (1-50).
        tiers: How deep to run the tier pipeline.

    Returns:
        Confirmed results plus the two suggestion channels.
    """
    return _run(repo, index, term, k, tiers, settings, embedder)


@router.post("/lookup", response_model=BatchLookupResponse)
def lookup_batch(
    payload: BatchLookupRequest,
    repo: Annotated[Repository, Depends(get_repository)],
    index: Annotated[KeyIndex, Depends(get_key_index)],
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[Embedder | None, Depends(get_embedder)],
    k: KParam = 5,
    tiers: TierDepth = TierDepth.SEMANTIC,
) -> BatchLookupResponse:
    """Route a batch of terms, applying the single-term logic per item.

    Args:
        payload: The batch body (1-100 non-blank terms).
        repo: Injected repository.
        index: Injected fuzzy key index.
        settings: Injected settings.
        embedder: Injected embedding client, or ``None`` when unconfigured.
        k: Per-channel suggestion cap (1-50).
        tiers: How deep to run the tier pipeline.

    Returns:
        One answer per submitted term, in submission order.
    """
    return BatchLookupResponse(
        items=[
            _run(repo, index, term, k, tiers, settings, embedder)
            for term in payload.terms
        ]
    )
