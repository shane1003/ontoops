"""CLI entry point (argparse; the same stack as the tboxcraft/aboxmapper CLIs).

It only declares the options and delegates the actual flow to the extract runner.

Subcommands:
 - extract : CQList → annotated TBox components + ABox individuals (no TTL)

Example:
    uv run conceptualizer extract data/cq/military_cq.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conceptualizer",
        description=(
            "conceptualizer: CQ document → TBox components + ABox individual "
            "classification"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="logging level (DEBUG/INFO/WARNING/ERROR)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ex = sub.add_parser(
        "extract",
        help=(
            "extract a CQList into annotated TBox components + ABox individuals "
            "(no TTL)"
        ),
    )
    ex.add_argument("cq_file", type=Path, help="path of the input CQList (JSON)")
    ex.add_argument(
        "--config", type=Path, default=Path("config/settings.yaml"), help="config file"
    )
    ex.add_argument(
        "--components-out",
        type=Path,
        default=None,
        help=(
            "output path of the annotated components artifact "
            "(default: from settings)"
        ),
    )
    ex.add_argument(
        "--individuals-out",
        type=Path,
        default=None,
        help=(
            "output path of the individuals artifact "
            "(default: individuals.json next to the components)"
        ),
    )
    ex.add_argument(
        "--language",
        default=None,
        help=(
            "fallback language of the domain-layer text (correct_term, notes) "
            "(BCP-47, e.g. ko). Used only for values the classifier gave no "
            "language for; without it they go out as und."
        ),
    )
    return parser


def app(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    # Initialize logging before running the subcommand (default INFO, so progress
    # logs are visible).
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    from app.core.config import load_settings
    from app.conceptualizer.service.runner import run_extraction

    settings = load_settings(args.config)
    output = run_extraction(
        args.cq_file,
        settings,
        components_out=args.components_out,
        individuals_out=args.individuals_out,
        language=args.language,
    )
    print(
        f"extract done: components={len(output.components)} "
        f"individuals={len(output.individuals)}"
    )


if __name__ == "__main__":
    app()
