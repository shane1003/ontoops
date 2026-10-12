"""Single point of Postgres access: DSN resolution + psycopg connections/pool.

Every module of this repository (tboxcraft, aboxmapper) points at the same Postgres
through one DSN environment variable, ``TBOXCRAFT_DATABASE_URL``. No host/port/user piece
env vars are used: one DSN string is the only connection representation, and compose
assembles and injects it.

- :func:`get_database_url`: reads the DSN from env (+ ``.env``). Fails clearly if unset.
- :func:`connect`: one-shot psycopg connection (per CLI/one-off job).
- :func:`open_pool` / :func:`close_pool` / :func:`connection`: process-wide connection
  pool. When the app lifespan calls :func:`open_pool`, :func:`connection` borrows from the
  pool; without a pool (tests, CLI) the behavior is preserved as a one-shot
  :func:`connect`.

tboxcraft needs pgvector type registration, so it configures its own pool
(``app/tboxcraft/db/pool.py``); only DSN resolution is unified here.

Every connection gets a default ``connect_timeout`` (a value set in the DSN itself wins).
It keeps worker threads from being held indefinitely by an unresponsive DB.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg_pool import ConnectionPool

ENV_VAR = "TBOXCRAFT_DATABASE_URL"

# Upper bound (seconds) on establishing a connection. Query run time (statement_timeout)
# is left alone: a bulk load can legitimately take long, so whoever knows the workload sets
# that bound as a DSN option.
DEFAULT_CONNECT_TIMEOUT = 10


def get_database_url(env_var: str = ENV_VAR) -> str:
    """Return the shared Postgres DSN, or raise a clear error if unset."""
    load_dotenv()
    url = os.environ.get(env_var)
    if not url:
        raise RuntimeError(
            f"{env_var} is not set. Point it at the tboxcraft Postgres, e.g. "
            "postgresql://tboxcraft@localhost:5432/tboxcraft"
        )
    return url


def _with_default_timeout(url: str) -> str:
    """Add ``connect_timeout`` unless the DSN already chose one (the DSN wins)."""
    if "connect_timeout" in conninfo_to_dict(url):
        return url
    return make_conninfo(url, connect_timeout=DEFAULT_CONNECT_TIMEOUT)


def connect(url: str | None = None) -> psycopg.Connection:
    """Open a psycopg connection, defaulting to the configured DSN.

    ``url`` is any psycopg conninfo string — a ``postgresql://…`` URL or the keyword form
    ``host=… dbname=…`` — so callers can take the target from outside (e.g. the migration
    CLI's ``--dsn``) instead of the environment.
    """
    return psycopg.connect(_with_default_timeout(url or get_database_url()))


# --- Shared pool --------------------------------------------------------------
# One per process. The lifespan opens and closes it; when it is not open, connection()
# falls back to a one-shot connection, so test/CLI paths running without a pool behave
# unchanged.

_pool: ConnectionPool | None = None


def _register_types(conn: psycopg.Connection) -> None:
    """Registers pgvector types per connection; skipped on a DB without the extension.

    The shared pool serves both a consumer that uses vectors (tboxcraft) and one that does
    not (aboxmapper), so a registration failure is not made a pool failure. On a DB
    without the extension, the side that actually uses vectors sees the failure at its
    first vector query.
    """
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        pass


def open_pool(url: str | None = None, **pool_kwargs) -> ConnectionPool:
    """Open the process-wide pool (idempotent). Raises if no DSN is configured.

    If already open it is returned as is: the **first call** decides the DSN (in practice
    the same, since every module uses the same env DSN). ``open=True`` fills in the
    background, so this does not die when the DB is not up yet; :func:`connection` answers
    with PoolTimeout (->503) at the point of actual use.
    """
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=_with_default_timeout(url or get_database_url()),
            configure=_register_types,
            open=True,
            **pool_kwargs,
        )
    return _pool


def close_pool() -> None:
    """Close and forget the process-wide pool (no-op if none is open)."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def connection(url: str | None = None) -> Iterator[psycopg.Connection]:
    """Yield a connection: from the pool when one is open, else a one-shot connect.

    The pool path follows the psycopg convention: commit when the body exits normally,
    rollback on an exception, then return the connection. The one-shot path keeps the
    previous close() behavior (exit without commit = rollback). An explicit ``url``
    always bypasses the pool (the pool is bound to the one environment DSN).
    """
    if url is None and _pool is not None:
        with _pool.connection() as conn:
            yield conn
    else:
        conn = connect(url)
        try:
            yield conn
        finally:
            conn.close()
