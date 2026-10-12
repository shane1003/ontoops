"""ABox schema bootstrap + migration runner.

This module owns the ABox tables (`abox_individual`, `abox_data_assertion`). On startup we
create them when they are missing and leave them untouched when they already exist, so
`docker compose up` works against an empty database as well as a provisioned one.

Two steps, both run by :mod:`app.aboxmapper.cli.migrate`:

1. :func:`ensure_schema` — CREATE the tables when absent (fresh DB). Gated on a cheap
   existence check so the common path is one query, not a re-run of the whole DDL.
2. :func:`apply_pending_migrations` — apply every bundled ``sql/migrations/*.sql`` (name
   order) that a ledger table has not yet recorded. This is what evolves an **existing** DB
   (e.g. adding ``source_id``); ``ensure_schema`` alone never touches a populated database.

A fresh CREATE already embodies every migration, so :func:`ensure_schema` stamps the ledger
with all migration filenames — the runner then only migrates pre-existing databases. An
advisory lock serialises concurrent workers.
"""

from __future__ import annotations

import logging
from importlib import resources

_SENTINEL = "public.abox_individual"
_LEDGER = "aboxmapper_schema_migrations"
_LOCK_KEY = 8_604_001  # arbitrary, stable: serialises schema bootstrap across workers

log = logging.getLogger(__name__)


def schema_sql() -> str:
    """The bundled ABox DDL (packaged, so it resolves regardless of CWD)."""
    return resources.files("app.aboxmapper").joinpath("sql", "abox_schema.sql").read_text(encoding="utf-8")


def migration_files() -> list[tuple[str, str]]:
    """Bundled ``sql/migrations/*.sql`` as ``(filename, sql)`` pairs, sorted by name."""
    root = resources.files("app.aboxmapper").joinpath("sql", "migrations")
    files = [
        (entry.name, entry.read_text(encoding="utf-8"))
        for entry in root.iterdir()
        if entry.name.endswith(".sql")
    ]
    files.sort(key=lambda pair: pair[0])
    return files


def _ensure_ledger(cur) -> None:
    cur.execute(
        f"CREATE TABLE IF NOT EXISTS {_LEDGER} ("
        "filename text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
    )


def ensure_schema(conn) -> bool:
    """Create the ABox tables if absent. Returns True when they were created.

    On a fresh create the ledger is stamped with every migration filename, since the created
    schema already includes their effect — so :func:`apply_pending_migrations` only migrates
    databases that predate a migration.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_KEY,))
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (_SENTINEL,))
        if cur.fetchone()[0]:
            conn.commit()
            return False
        cur.execute(schema_sql())
        _ensure_ledger(cur)
        for name, _ in migration_files():
            cur.execute(
                f"INSERT INTO {_LEDGER} (filename) VALUES (%s) ON CONFLICT (filename) DO NOTHING",
                (name,),
            )
    conn.commit()
    log.info("ABox schema created (%s was missing)", _SENTINEL)
    return True


def apply_pending_migrations(conn) -> list[str]:
    """Apply bundled migrations not yet in the ledger (name order); return applied filenames.

    Runs whether ``ensure_schema`` created or skipped the schema: an existing DB whose schema
    predates a migration gets it applied here. Each file and its ledger row commit together,
    so a re-run skips already-applied files. A session advisory lock serialises workers.
    """
    applied: list[str] = []
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_lock(%s)", (_LOCK_KEY,))
    conn.commit()
    try:
        with conn.cursor() as cur:
            _ensure_ledger(cur)
            conn.commit()
            cur.execute(f"SELECT filename FROM {_LEDGER}")
            done = {row[0] for row in cur.fetchall()}
        for name, sql in migration_files():
            if name in done:
                continue
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    cur.execute(
                        f"INSERT INTO {_LEDGER} (filename) VALUES (%s) "
                        "ON CONFLICT (filename) DO NOTHING",
                        (name,),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            applied.append(name)
            log.info("applied migration %s", name)
    finally:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
        conn.commit()
    return applied
