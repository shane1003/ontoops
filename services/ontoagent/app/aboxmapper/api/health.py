"""Liveness, readiness and target reporting."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

import psycopg
from fastapi import APIRouter, HTTPException

from app.core import db

from ..core.config import (
    BaseIriRequired,
    InvalidBaseIri,
    get_falkordb_config,
    get_ontology_base_iri,
)
from ..schemas.target import FalkorTarget, PostgresTarget, TargetResponse

router = APIRouter()


@contextmanager
def _connection() -> Iterator[psycopg.Connection]:
    """Acquire the ABox connection, or fail with the status that matches the cause.

    A missing/invalid ``TBOXCRAFT_DATABASE_URL`` is ours to fix, so it is a 500; a database
    that will not answer (or an exhausted pool) is a 503 the caller may retry. Only
    *acquisition* is mapped here — errors raised while using the connection keep their own
    handling in the caller.

    Yields:
        An open connection (pooled when the app opened ``app.core.db``'s pool).

    Raises:
        HTTPException: 500 when unconfigured, 503 when the database is unreachable.
    """
    stack = ExitStack()
    try:
        conn = stack.enter_context(db.connection())
    except RuntimeError as exc:  # missing/invalid configuration
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except psycopg.OperationalError as exc:  # DB unreachable / pool timeout
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}") from exc
    with stack:
        yield conn


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
def ready() -> dict:
    try:
        with db.connection():
            pass
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}") from exc
    return {"status": "ready"}


@router.get("/target", response_model=TargetResponse)
def target() -> TargetResponse:
    """Report which stores this instance writes to, so a caller can catch a wrong target.

    Postgres is read from the **live connection**, not the environment: the migration CLI
    can be pointed elsewhere with ``--dsn``, so an env echo could disagree with reality and
    a target report that can lie is worse than none. Passwords are never included.
    """
    with _connection() as conn:
        info = conn.info
        pg = PostgresTarget(host=info.host, port=info.port, dbname=info.dbname, user=info.user)

    try:
        default_base_iri, default_base_iri_source = get_ontology_base_iri(), "env"
    except InvalidBaseIri:
        # Configured but unusable. Reported as source "env" with a null value so this is
        # distinguishable from "nothing configured" — the load itself fails loudly with
        # invalid_base_iri, and a target report should not make a broken default look absent.
        default_base_iri, default_base_iri_source = None, "env"
    except BaseIriRequired:
        # No default at all: a load then requires an explicit ?base_iri=. Not an error here.
        default_base_iri, default_base_iri_source = None, None

    try:
        cfg = get_falkordb_config()
    except RuntimeError:
        # No default graph: /graph/tbox then requires an explicit ?graph=. Not an error here.
        return TargetResponse(
            postgres=pg,
            falkordb=None,
            default_base_iri=default_base_iri,
            default_base_iri_source=default_base_iri_source,
        )
    return TargetResponse(
        postgres=pg,
        falkordb=FalkorTarget(host=cfg["host"], port=cfg["port"], graph=cfg["graph"]),
        default_base_iri=default_base_iri,
        default_base_iri_source=default_base_iri_source,
    )
