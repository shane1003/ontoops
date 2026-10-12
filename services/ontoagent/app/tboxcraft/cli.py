"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys

from app.tboxcraft.config import Config
from app.core.logging_setup import configure_logging
from app.tboxcraft.pipeline import generate, update


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tboxcraft", description="Generate and validate a TBox."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Bootstrap a TBox from components.json.")
    gen.add_argument("--components", required=True, help="Path to component.json")
    gen.add_argument(
        "--output", default=None, help="Output directory (default: config)"
    )
    gen.add_argument(
        "--provider",
        choices=["e5", "remote_e5"],
        default=None,
        help="Embedding provider override",
    )
    gen.add_argument(
        "--no-reasoner", action="store_true", help="Skip the HermiT gate stage"
    )
    gen.add_argument(
        "--oops", action="store_true", help="Run the (advisory) OOPS! scan"
    )

    upd = sub.add_parser(
        "update",
        help="Route a later batch against the accumulated ontology (needs the DB).",
    )
    upd.add_argument(
        "--components", required=True, help="Path to the batch component.json"
    )
    upd.add_argument(
        "--no-reasoner", action="store_true", help="Skip the HermiT gate stage"
    )
    upd.add_argument(
        "--oops", action="store_true", help="Run the (advisory) OOPS! scan"
    )
    return parser


def _run_generate(args) -> int:
    config = Config.from_env()
    if args.provider:
        config.embedding_provider = args.provider
    if args.output:
        config.output_dir = args.output

    result = generate(
        args.components,
        config,
        run_reasoner=not args.no_reasoner,
        run_oops=args.oops,
    )
    if not result.ok:
        print("TBox rejected by validation gate:", file=sys.stderr)
        for reason in result.gate_reasons:
            print(f"  - {reason}", file=sys.stderr)
        print(f"Report: {result.report_path}", file=sys.stderr)
        return 1

    print(f"TBox written: {result.ttl_path}")
    print(f"Mapping:      {result.mapping_path}")
    print(f"Report:       {result.report_path}")
    print(f"Merge report: {result.merge_report_path}")
    print(f"{result.n_components} components -> {result.n_concepts} concepts")
    return 0


def _run_update(args) -> int:
    config = Config.from_env()
    from app.tboxcraft.db import Repository, build_pool

    pool = build_pool(config)
    repo = Repository(pool)
    try:
        result = update(
            args.components,
            config,
            repo=repo,
            run_reasoner=not args.no_reasoner,
            run_oops=args.oops,
        )
    finally:
        pool.close()

    if not result.ok:
        print("Update rejected by validation gate:", file=sys.stderr)
        for reason in result.gate_reasons:
            print(f"  - {reason}", file=sys.stderr)
        return 1

    print(
        f"Update: {result.n_candidates} candidates -> "
        f"{result.n_new} new, {result.n_merged} merged, "
        f"{result.n_relabel} relabel-queued, {result.n_gray} gray-queued"
    )
    print(f"Ontology version: {result.version_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Same configuration as the API path, so a log line behaves the same however tboxcraft is run.
    configure_logging(env_var="TBOXCRAFT_LOG_LEVEL")
    args = _build_parser().parse_args(argv)
    rc = 2
    if args.command == "generate":
        rc = _run_generate(args)
    elif args.command == "update":
        rc = _run_update(args)
    from app.core.tracing import (
        flush_tracer,  # flush buffered traces before exit (no-op if none)
    )

    flush_tracer()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
