"""CLI entrypoint for building the TBox graph in FalkorDB from a Turtle file."""

from __future__ import annotations

import argparse
import logging

from ..services import graph

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the TBox graph in FalkorDB from a Turtle (.ttl) file."
    )
    parser.add_argument("ttl", help="Path to the TBox .ttl file")
    parser.add_argument(
        "--graph",
        default=None,
        help="target named graph for this run (default: $FALKORDB_GRAPH)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        fg = graph.connect_falkordb(args.graph)
    except RuntimeError as exc:  # no --graph and FALKORDB_GRAPH unset
        log.error("%s", exc)
        return 2

    report = graph.run(args.ttl, fg)
    log.info("TBox graph built in %s: %s", fg.name, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
