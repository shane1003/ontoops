"""Load endpoints: individuals as a JSON body or an uploaded component.json.

Status codes distinguish *our* failures from the caller's: a precondition the caller must
fix upstream (source_ids absent from the `source` registry, a malformed or empty base IRI)
is **422**, not 503 — 503 means "retry later", and none of those become true by retrying.
Every 422 body carries a stable ``detail.code`` so callers need not parse prose, and need
not confuse ours with FastAPI's own request-validation 422.
"""

from __future__ import annotations

import json
from typing import Any

import psycopg
from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.iri import resolve_base_iri

from ..core.config import BaseIriRequired, InvalidBaseIri
from ..schemas.load import LoadResponse, to_response
from ..services.loader import (
    LoadReport,
    SourceRegistryMissing,
    UnregisteredSources,
    extract_individuals,
    run,
)
from ..services.resolver import EmptyTermNamespace
from .health import _connection

router = APIRouter()

# A payload can name thousands of missing sources; the list is diagnostic, so cap what we
# echo and keep the full count alongside it.
MAX_REPORTED_SOURCES = 50

def _normalize(payload: Any) -> list[dict]:
    try:
        return extract_individuals(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _resolve_namespace(
    base_iri: str | None,
    iri_domain: str | None,
    document: str | None,
    iri_root: str | None,
) -> tuple[str, str, dict[str, str] | None] | None:
    """Resolves this request's namespace (app.core.iri).

    A failure stays the existing invalid_base_iri 422.
    """
    try:
        return resolve_base_iri(base_iri, root=iri_root, domain=iri_domain, document=document)
    except InvalidBaseIri as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_base_iri", "message": str(exc)},
        ) from exc


def _run_load(
    individuals: list[dict],
    dry_run: bool,
    resolved: tuple[str, str, dict[str, str] | None] | None,
) -> LoadResponse:
    base_iri = resolved[0] if resolved else None
    with _connection() as conn:
        report = _run_on(conn, individuals, dry_run, base_iri)
    # Reaching here with nothing sent means run() succeeded with the env default.
    return to_response(
        report,
        base_iri_source=resolved[1] if resolved else "env",
        base_iri_pieces=resolved[2] if resolved else None,
    )


def _run_on(
    conn: psycopg.Connection, individuals: list[dict], dry_run: bool, base_iri: str | None
) -> LoadReport:
    """Run the loader on ``conn``, mapping loader/DB failures to their HTTP status."""
    try:
        return run(conn, individuals, dry_run=dry_run, base_iri=base_iri)
    except InvalidBaseIri as exc:
        # The caller's namespace, malformed. Never repaired silently — the minted IRIs are
        # stored, so a quietly "fixed" base becomes permanent identity nobody can reproduce.
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_base_iri",
                "message": str(exc),
            },
        ) from exc
    except BaseIriRequired as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "base_iri_required",
                "message": str(exc),
            },
        ) from exc
    except EmptyTermNamespace as exc:
        # Wrong namespace or a TBox not loaded for this document: without this the load
        # resolves nothing and reports a clean, empty success.
        raise HTTPException(
            status_code=422,
            detail={
                "code": "empty_term_namespace",
                "message": str(exc),
            },
        ) from exc
    except UnregisteredSources as exc:
        # Caller-fixable precondition: the upstream must register these sources first.
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unregistered_sources",
                "message": "source registry is missing entries these individuals point at — "
                "the upstream must register them before loading; nothing was written",
                "missing_sources": exc.missing[:MAX_REPORTED_SOURCES],
                "n_missing_sources": len(exc.missing),
                # Individuals and values are reported apart: the upstream fix is the same
                # (emit source_ref), but which path leaked is otherwise invisible.
                "n_unsourced_individuals": exc.unsourced_subjects,
                "n_unsourced_values": exc.unsourced_values,
                "n_individuals": exc.individuals,
            },
        ) from exc
    except SourceRegistryMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except psycopg.errors.UndefinedTable as exc:
        raise HTTPException(
            status_code=503,
            detail=f"ABox schema missing — run the migration step first "
            f"(python -m app.aboxmapper.cli.migrate): {exc}",
        ) from exc
    except (psycopg.errors.ForeignKeyViolation, psycopg.errors.NotNullViolation) as exc:
        # Backstop: the pre-check above catches these, but a source deleted mid-load still
        # lands here — an input/precondition failure either way, never "retry later".
        raise HTTPException(
            status_code=422,
            detail={
                "code": "source_constraint_violation",
                "message": f"source precondition violated during the write: {exc}",
            },
        ) from exc
    except psycopg.Error as exc:  # genuine database problems only
        raise HTTPException(status_code=503, detail=f"database error: {exc}") from exc


@router.post("/load", response_model=LoadResponse)
def load(
    payload: Any = Body(...),
    dry_run: bool = False,
    base_iri: str | None = Query(None),
    document: str | None = Query(None),
    iri_domain: str | None = Query(None),
    iri_root: str | None = Query(None),
) -> LoadResponse:
    """Load individuals sent as the JSON body (array, ``{"individuals": [...]}``, or component.json).

    ``?base_iri=`` is the document's namespace for this load — it scopes which TBox terms may
    resolve and where new individual IRIs are minted, and beats ``ONTOLOGY_BASE_IRI``. The
    response echoes the namespace actually used, its source, and the assembly pieces.

    Piece path: send ``?iri_domain=…&document=…`` (+ optional ``iri_root``, default
    ONTOLOGY_IRI_ROOT) and the server assembles the namespace with the single rule of
    app.core.iri. If ``base_iri`` comes too, it is checked against the assembled one and a
    mismatch is a 422 (dual-run check during the transition).
    """
    resolved = _resolve_namespace(base_iri, iri_domain, document, iri_root)
    return _run_load(_normalize(payload), dry_run, resolved)


@router.post("/load/upload", response_model=LoadResponse)
async def load_upload(
    file: UploadFile = File(...),
    dry_run: bool = False,
    base_iri: str | None = Query(None),
    document: str | None = Query(None),
    iri_domain: str | None = Query(None),
    iri_root: str | None = Query(None),
) -> LoadResponse:
    """Load individuals from an uploaded component.json file (params as ``/load``).

    Only the file receive is async: the load is blocking DB work, so it goes to the
    threadpool; run on the event loop, it would stall the whole app (including /health)
    for the duration of the load (app.main policy).
    """
    raw = await file.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400, detail=f"uploaded file is not valid JSON: {exc}"
        ) from exc
    resolved = _resolve_namespace(base_iri, iri_domain, document, iri_root)
    return await run_in_threadpool(_run_load, _normalize(payload), dry_run, resolved)
