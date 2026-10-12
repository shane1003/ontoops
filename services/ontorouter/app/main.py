"""FastAPI application factory."""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import admin, lookup
from app.clients.embed import embed_client
from app.core.config import get_settings
from app.core.db import Database
from app.core.schema import apply_ddl
from app.services.key_index import KeyIndex
from app.services.repository import PostgresRepository
from app.services.semantic_index import run_backfill

# A module logger only: configuring handlers or levels is the app host's job
# (uvicorn, or whatever supervises the container), never a library's.
logger = logging.getLogger(__name__)


async def _periodic_backfill(app: FastAPI) -> None:
    """Run the backfill on the configured interval until cancelled.

    Args:
        app: The FastAPI application, carrying the open connection pool.
    """
    settings = get_settings()
    while True:
        await asyncio.sleep(settings.embed_backfill_interval_s)
        repo = PostgresRepository(app.state.db, timeout_s=settings.pg_timeout_s)
        embedder = embed_client(str(settings.embed_url), settings.embed_timeout_s)
        try:
            embedded = await asyncio.to_thread(
                run_backfill, repo, embedder, str(settings.embedding_model_id)
            )
        except Exception:  # noqa: BLE001 - a failed pass must not kill the loop
            # Without this, a TEI outage, a dimension mismatch and a missing
            # ontorouter schema are one indistinguishable silent no-op, forever.
            logger.exception("embedding backfill pass failed; retrying next interval")
        else:
            logger.info("embedding backfill pass complete: %d rows embedded", embedded)


async def _periodic_key_index_refresh(app: FastAPI) -> None:
    """Rebuild the fuzzy key index on the configured interval until cancelled.

    Args:
        app: The FastAPI application, carrying the pool and the index.
    """
    settings = get_settings()
    while True:
        await asyncio.sleep(settings.key_index_refresh_interval_s)
        repo = PostgresRepository(app.state.db, timeout_s=settings.pg_timeout_s)
        try:
            loaded = await asyncio.to_thread(app.state.key_index.refresh, repo)
        except Exception:  # noqa: BLE001 - a failed refresh must not kill the
            # loop; the previous keys stay in place and simply age further.
            logger.exception("key index refresh failed; keeping the previous keys")
        else:
            logger.info("key index refreshed: %d keys", loaded)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the connection pool and start the periodic backfill for the app's life.

    Args:
        app: The FastAPI application.

    Yields:
        None, while the pool is open and the periodic task is running.
    """
    settings = get_settings()
    app.state.db = Database(settings.dsn)
    app.state.key_index = KeyIndex()
    try:
        applied = apply_ddl(app.state.db)
        logger.info("schema up to date: applied %s", ", ".join(applied) or "nothing")
    except Exception:  # noqa: BLE001 - a schema the service cannot create is
        # not fatal: exact and lexical never read the ontorouter schema, and a
        # semantic lookup degrades through the channel-drop path rather than
        # failing the request. Logged loudly because it always means either a
        # least-privilege role or a genuinely broken DDL file.
        logger.exception(
            "could not apply the checked-in DDL; semantic tier may degrade"
        )
    repo = PostgresRepository(app.state.db, timeout_s=settings.pg_timeout_s)
    try:
        loaded = await asyncio.to_thread(app.state.key_index.refresh, repo)
        logger.info("key index loaded: %d keys", loaded)
    except Exception:  # noqa: BLE001 - a cold index degrades the lexical
        # channel; results still come from live SQL, so this must not stop the
        # service from starting.
        logger.exception("could not load the key index; lexical tier will degrade")

    tasks = [asyncio.create_task(_periodic_key_index_refresh(app))]
    if settings.semantic_enabled:
        tasks.append(asyncio.create_task(_periodic_backfill(app)))
    try:
        yield
    finally:
        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        app.state.db.close()


def create_app() -> FastAPI:
    """Build the ontorouter FastAPI application.

    Returns:
        The configured application.
    """
    app = FastAPI(title="ontorouter", version="0.1.0", lifespan=lifespan)
    app.include_router(lookup.router)
    app.include_router(admin.router)
    return app


app = create_app()
