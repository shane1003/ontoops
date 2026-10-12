"""FastAPI service exposing the ABox loader.

Endpoints
---------
GET  /health        liveness (no DB touch)
GET  /ready         readiness — verifies the tboxcraft DB is reachable
GET  /target        which stores this instance writes to (no secrets)
POST /load          run the loader on individuals sent as the JSON body
POST /load/upload   run the loader on an uploaded component.json file
POST /graph/tbox    build the TBox graph in FalkorDB from a Turtle document

One document is one KG, told apart by its base IRI: ``?base_iri=`` scopes term resolution
and individual minting for a load, ``?graph=`` picks the FalkorDB named graph. Both beat the
environment default, and both are echoed in the response.

This service issues no DDL. The ABox schema is created by the separate one-shot migration
step (``python -m app.aboxmapper.cli.migrate``), so the API can run with data-write privileges only.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core import db

from .api import graph, health, load


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the shared Postgres pool for the app's lifetime.

    Without a DSN it starts with no pool (requests report the cause as 500/503). Runs
    that skip the lifespan (TestClient outside a context, CLI) also work, through the
    single-connection fallback of app.core.db.
    """
    try:
        db.open_pool()
    except RuntimeError:
        pass
    try:
        yield
    finally:
        db.close_pool()


app = FastAPI(
    title="aboxmapper",
    description="Load individuals into the tboxcraft ABox store.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(health.router)
app.include_router(load.router)
app.include_router(graph.router)
