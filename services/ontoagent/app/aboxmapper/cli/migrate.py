"""One-shot ABox schema migration entrypoint.

DDL is deliberately separated from the application runtime: the API and CLI never issue
DDL, so they can run with data-write privileges only. Run this once before the app — as the
compose `abox-migrate` service, or by hand:

    python -m app.aboxmapper.cli.migrate    # or: aboxmapper-migrate
    aboxmapper-migrate --wait 60            # wait for a DB that is still starting

The target database is given from outside as a DSN, so one build can migrate any
deployment without editing ``.env``:

    aboxmapper-migrate --dsn postgresql://user:pw@db.example.net:5432/tboxcraft
    aboxmapper-migrate --dsn "host=db.example.net port=5432 dbname=tboxcraft user=abox"

Both DSN forms are psycopg conninfo strings. Without ``--dsn`` the environment
(``TBOXCRAFT_DATABASE_URL``) is used, as before.

Exit codes: 0 ok · 2 misconfigured · 3 database unreachable · 4 migration failed
"""

from __future__ import annotations

import argparse
import logging
import time

import psycopg

from app.core import db

from ..core.schema import apply_pending_migrations, ensure_schema

log = logging.getLogger(__name__)


def _connect_with_retry(timeout: float, dsn: str | None = None) -> psycopg.Connection:
    """Connect to ``dsn`` (or the configured URL), tolerating a DB that is still coming up."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return db.connect(dsn)
        except psycopg.OperationalError:
            if time.monotonic() >= deadline:
                raise
            log.info("database not ready yet, retrying…")
            time.sleep(1.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create the ABox schema if it is missing.")
    parser.add_argument(
        "--dsn",
        metavar="DSN",
        default=None,
        help="target database as a psycopg conninfo string — either a URL "
        "(postgresql://user:pw@host:5432/db) or keyword form "
        "('host=… port=… dbname=… user=…'). Default: $TBOXCRAFT_DATABASE_URL.",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=30.0,
        help="seconds to keep retrying while the database starts (default: 30)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        conn = _connect_with_retry(args.wait, args.dsn)
    except RuntimeError as exc:  # missing/invalid configuration
        log.error("%s", exc)
        return 2
    except psycopg.OperationalError as exc:
        log.error("database unavailable after %.0fs: %s", args.wait, exc)
        return 3

    # Say which database is about to receive DDL — with an external DSN the target is no
    # longer implied by the environment, and applying it to the wrong DB is the failure mode.
    info = conn.info
    log.info("migrating %s@%s:%s/%s", info.user, info.host, info.port, info.dbname)

    try:
        created = ensure_schema(conn)
        applied = apply_pending_migrations(conn)
    except psycopg.Error as exc:
        log.error("migration failed: %s", exc)
        return 4
    finally:
        conn.close()

    log.info(
        "ABox schema %s; migrations applied: %s",
        "created" if created else "already present",
        ", ".join(applied) or "none",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
