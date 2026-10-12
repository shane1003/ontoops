"""psycopg3 connection pool with pgvector registration.

``register_vector`` runs per connection (via the pool ``configure`` callback) so numpy
float32 arrays round-trip to the ``vector`` type without manual serialization.

The two entry points have different roles:

- :func:`shared_pool`: for the server (lifespan). Joins the process-wide pool of
  ``app.core.db`` (shares connections with aboxmapper; never two pools on one DB).
- :func:`build_pool`: creates a dedicated pool. Used where a pool with its own lifetime
  is needed for an explicit DSN, such as the CLI (standalone process) and integration
  tests.
"""

from __future__ import annotations

from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.core import db as core_db
from app.core.db import ENV_VAR as DATABASE_URL_ENV_VAR
from app.tboxcraft.config import Config


def _require_url(config: Config) -> str:
    if not config.database_url:
        raise RuntimeError(
            f"{DATABASE_URL_ENV_VAR} is not set. Set it in .env/compose "
            "(no static credential is shipped as a default)."
        )
    return config.database_url


def shared_pool(config: Config) -> ConnectionPool:
    """Join the process-wide pool from ``config.database_url``. Fails closed if unset.

    pgvector registration is done by the shared pool's connection-configure
    (app.core.db).
    """
    return core_db.open_pool(_require_url(config))


def build_pool(config: Config, *, open: bool = True, **pool_kwargs) -> ConnectionPool:
    """Build a dedicated connection pool from ``config.database_url``. Fails closed if unset."""
    return ConnectionPool(
        conninfo=_require_url(config),
        configure=register_vector,
        open=open,
        **pool_kwargs,
    )
