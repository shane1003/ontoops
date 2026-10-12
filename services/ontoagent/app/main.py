"""App assembly (entry point): FastAPI instance + module router includes + shared endpoints.

Each module owns its endpoints through the APIRouter in `app/<module>/router.py`; here
those routers are included and assembled into one server (adding a module = one include
line).

All handlers are sync: the runners are blocking (LLM), so wrapping them in async def
would only block the event loop. FastAPI runs sync handlers in the threadpool.

Import-safe contract: importing this module alone needs no LLM key or network.
Secrets (API keys etc.) are never put in any response or log.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.aboxmapper.main import app as aboxmapper_app
from app.aboxmapper.main import lifespan as aboxmapper_lifespan
from app.core.logging_setup import configure_logging as core_configure_logging
from app.conceptualizer.router import _NO_LLM_MSG, _error
from app.conceptualizer.router import router as conceptualizer_router
from app.tboxcraft.router import init_state as init_tboxcraft_state
from app.tboxcraft.router import lifespan as tboxcraft_lifespan
from app.tboxcraft.router import router as tboxcraft_router

log = logging.getLogger("conceptualizer.api")


def configure_logging() -> None:
    """Sets the `conceptualizer` logger to CONCEPTUALIZER_LOG_LEVEL (default INFO).

    This module is the API entry point, so it configures **at import time**:
    `uvicorn app.main:app` does not go through main(). The mechanism (package logger level
    + a handler only when nobody has attached one) is owned by core.logging_setup.
    """
    core_configure_logging(
        env_var="CONCEPTUALIZER_LOG_LEVEL",
        logger_name="conceptualizer",
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


configure_logging()

_SERVICE_NAME = "conceptualizer"
_SERVICE_VERSION = "0.1.0"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Combines the tboxcraft lifespan with the aboxmapper shared pool.

    Starlette does not run the lifespan of a mounted sub-app, so the root app opens and
    closes the aboxmapper pool in its place. Both sides skip pool creation when
    TBOXCRAFT_DATABASE_URL is unset (only their routes return 500/503), so a
    conceptualizer-only startup without a DB is safe.
    """
    async with tboxcraft_lifespan(app), aboxmapper_lifespan(app):
        yield


app = FastAPI(
    title="conceptualizer API",
    version=_SERVICE_VERSION,
    description=(
        "Synchronous API: CQ document → TBox component + ABox individual classification"
        " (single stage)."
    ),
    # tboxcraft's resources (DB pool, providers, tracer) open and close on this app's state.
    lifespan=_lifespan,
)

app.include_router(conceptualizer_router, tags=["conceptualizer"])
# tboxcraft became a router in 3b; the prefix include merges it into the one /docs:
# /tbox/health, /tbox/generate, /tbox/update, /tbox/resolve, /tbox/review/*, /tbox/target.
# Fill the state defaults first so the /tbox routes answer with a 503 degrade even for
# requests before the lifespan.
init_tboxcraft_state(app)
app.include_router(tboxcraft_router, prefix="/tbox", tags=["tboxcraft"])
# The aboxmapper module is a mounted sub-app (/abox): it is router-based, but the original
# repository's API tests check the sub-app at the root path, so the mount keeps those tests
# as they are. No lifespan.
# /abox/health, /abox/ready, /abox/target, /abox/load(/upload), /abox/graph/tbox.
app.mount("/abox", aboxmapper_app)


# =========================================================================
# Infrastructure endpoints
# =========================================================================


@app.get("/health")
def health() -> dict:
    """Health check."""
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    """Service name/version + list of endpoints."""
    return {
        "service": _SERVICE_NAME,
        "version": _SERVICE_VERSION,
        "endpoints": [
            {"method": "GET", "path": "/health"},
            {"method": "GET", "path": "/"},
            {"method": "GET", "path": "/target"},
            {"method": "POST", "path": "/extract"},
            {"method": "*", "path": "/tbox/… (tboxcraft — see /docs)"},
            {"method": "*", "path": "/abox/… (aboxmapper — see /abox/docs)"},
        ],
    }


# =========================================================================
# Error mapping (exception → HTTP status)
# =========================================================================


@app.exception_handler(RuntimeError)
def _runtime_error_handler(_request: Request, exc: RuntimeError):
    # Map the RuntimeError the LLM client raises when the key is missing to 503.
    log.warning("RuntimeError → 503: %s", exc)
    return _error(503, _NO_LLM_MSG)


@app.exception_handler(Exception)
def _unexpected_handler(_request: Request, exc: Exception):
    # Unexpected error: 500 with a generic message, exposing no stack trace or secrets.
    log.exception("unexpected error during processing")
    return _error(500, "An internal server error occurred.")


# The server is started one way only, by calling uvicorn directly (same in compose,
# README, Dockerfile):
#   uvicorn app.main:app --host 0.0.0.0 --port 8602
# Note that omitting --host opens it on 127.0.0.1 (local only).
