"""CLI entrypoint for the ABox loader.

Exit codes: 0 ok · 2 input/configuration error · 3 precondition failed (nothing written):
a bad or missing base IRI, a namespace with no terms, or source_ids the registry lacks.
"""

from __future__ import annotations

import argparse
import json
import logging

from app.core import db

from ..core.config import BaseIriRequired, InvalidBaseIri
from ..services.loader import (
    DEFAULT_INPUT,
    SourceRegistryMissing,
    UnregisteredSources,
    read_input_file,
    run,
)
from ..services.resolver import EmptyTermNamespace

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load individuals from component.json into abox_individual."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"Path to component.json (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and report without writing to the database.",
    )
    parser.add_argument(
        "--base-iri",
        default=None,
        help="document namespace for this run — scopes which TBox terms resolve and where "
        "individual IRIs are minted (default: $ONTOLOGY_BASE_IRI)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        individuals = read_input_file(args.input)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        log.error("input error: %s", exc)
        return 2

    try:
        conn = db.connect()
    except RuntimeError as exc:  # missing/invalid configuration
        log.error("%s", exc)
        return 2

    try:
        report = run(conn, individuals, dry_run=args.dry_run, base_iri=args.base_iri)
    except (InvalidBaseIri, BaseIriRequired, EmptyTermNamespace) as exc:
        log.error("%s", exc)
        return 3
    except UnregisteredSources as exc:  # upstream ran out of order — nothing was written
        log.error("%s; register them upstream first", exc)
        for source_id in exc.missing:
            log.error("unregistered source_id: %s", source_id)
        return 3
    except SourceRegistryMissing as exc:
        log.error("%s", exc)
        return 3
    finally:
        conn.close()

    log.info("ABox load %s: %s", "DRY-RUN" if args.dry_run else "DONE", report.summary())
    for s in report.skips:
        detail = f" candidates={list(s.candidates)}" if s.candidates else ""
        log.warning(
            "skip %s %s token=%r reason=%s class=%s source=%s%s",
            s.scope,
            s.kind,
            s.token,
            s.reason,
            s.class_name,
            s.source_id,
            detail,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
