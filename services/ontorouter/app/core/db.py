"""Postgres connection handling with a per-query statement timeout."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg_pool import ConnectionPool


class QueryTimeout(Exception):
    """Raised when a query exceeds the configured statement timeout."""


class SchemaMissing(Exception):
    """Raised when a queried relation does not exist.

    Only ``SEMANTIC_SQL`` reads the ontorouter schema, so in practice this
    means the DDL has not been applied — the service could not create it and
    no operator did either. Translating it here keeps the driver behind this
    module and lets the semantic channel drop instead of failing the request.
    """


class Database:
    """Owns a psycopg connection pool.

    Args:
        dsn: A psycopg connection string.
        min_size: Minimum pooled connections.
        max_size: Maximum pooled connections.
    """

    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 8) -> None:
        self._pool = ConnectionPool(
            dsn, min_size=min_size, max_size=max_size, open=True
        )

    @contextmanager
    def connection(self, timeout_s: float | None = None) -> Iterator[Any]:
        """Yield a pooled connection inside a transaction.

        Args:
            timeout_s: Statement timeout for every query on this connection. The
                timeout is applied with SET LOCAL so it dies with the transaction.

        Yields:
            An open psycopg connection.

        Raises:
            QueryTimeout: If a query is cancelled by the statement timeout.
        """
        with self._pool.connection() as conn:
            try:
                with conn.transaction():
                    if timeout_s is not None:
                        # SET does not support the extended-query-protocol
                        # placeholder ($1): the value must be a literal in the
                        # SQL text. Safe here because it is an int computed
                        # from Settings, never user input.
                        timeout_ms = int(timeout_s * 1000)
                        conn.execute(f"SET LOCAL statement_timeout = {timeout_ms}")
                    yield conn
            except psycopg.errors.QueryCanceled as exc:
                raise QueryTimeout(str(exc)) from exc
            except psycopg.errors.UndefinedTable as exc:
                raise SchemaMissing(str(exc)) from exc

    @contextmanager
    def session(self) -> Iterator[Any]:
        """Check out one pooled connection for the caller to manage explicitly.

        Unlike ``connection()``, no transaction or statement timeout is applied
        automatically: the caller commits or rolls back as needed. This exists
        for callers that must pin a single physical connection across several
        statements — e.g. a Postgres session-scoped advisory lock, which can
        only be released by the exact connection that took it, not by any
        connection the pool happens to hand back on a later checkout.

        Yields:
            A pooled connection, checked out for the duration of the block.
        """
        with self._pool.connection() as conn:
            yield conn

    def close(self) -> None:
        """Close the pool."""
        self._pool.close()
