"""Command-line entry point of the evaluation package.

Subcommands are registered here but implemented in their own modules so that
the scoring side (score/aggregate/envcheck) never imports the execution
side (the service clients, stages and runner).
"""

import os
import subprocess
from pathlib import Path
from typing import cast

import typer

app = typer.Typer(help="OntoOps evaluation.", no_args_is_help=True)


@app.command()
def score(
    run_dir: Path = typer.Argument(..., help="experiments/runs/<run_id>"),
    results: Path = typer.Option(Path("experiments/results/results.jsonl")),
    exclude: list[str] = typer.Option(
        [], "--exclude", help="Manual slug whose cells are left unscored; repeatable."
    ),
) -> None:
    """Re-derives metric rows from stored run artifacts and appends them."""
    from evaluation.results import append_rows
    from evaluation.scoring.score_run import score_run

    rows = score_run(run_dir, frozenset(exclude))
    typer.echo(f"{append_rows(results, rows)} rows -> {results}")


@app.command()
def aggregate(
    results: Path = typer.Option(Path("experiments/results/results.jsonl")),
    out: Path = typer.Option(Path("experiments/results/aggregates.jsonl")),
    n_boot: int = typer.Option(1000, help="Bootstrap resamples."),
    seed: int = typer.Option(0, help="RNG seed, so aggregates are reproducible."),
) -> None:
    """Builds aggregates.jsonl: means, failure counts and bootstrap intervals."""
    from evaluation.results import aggregate as aggregate_rows
    from evaluation.results import read_rows, superseded_cells, write_aggregates

    rows = read_rows(results)
    dropped = superseded_cells(rows)
    if dropped:
        # Silently keeping one of two scorings is how a rerun ends up reported
        # with the previous run's numbers, so say it out loud.
        typer.echo(
            f"warning: {len(dropped)} cell(s) were scored more than once; "
            f"the newest scoring is used. Affected:"
        )
        for (experiment, run_type, dataset, metric, index), stamps in sorted(
            dropped.items(), key=lambda item: str(item[0])
        )[:10]:
            typer.echo(
                f"    {experiment}/{run_type}/{dataset}/{index} {metric}: "
                f"ignoring {len(stamps)} older row(s)"
            )
    aggs = aggregate_rows(rows, n_boot=n_boot, seed=seed)
    typer.echo(f"{write_aggregates(out, aggs)} cells -> {out}")


PINS = Path("experiments/pins.yaml")
ENV_FILE = Path(".env")
ENV_TEMPLATE = Path(".env.example")
COMPOSE = ["docker", "compose", "--env-file", str(ENV_FILE)]


@app.command()
def envcheck(
    profile: str = typer.Option("run", help="run | score"),
    weights: str = typer.Option(
        "google/gemma-4-31B-it", help="Weights both LLM servers must serve."
    ),
    pins: Path = typer.Option(PINS),
) -> None:
    """Checks the runtime environment; exits 1 on any missing item."""
    from evaluation.envcheck import (
        Profile,
        format_report,
        load_pins,
        read_env_file,
        run_checks,
    )

    if profile not in ("run", "score"):
        raise typer.BadParameter("profile must be 'run' or 'score'")
    # The process environment wins over .env, as it does for docker compose.
    env = {**read_env_file(ENV_FILE), **os.environ}
    results = run_checks(
        cast(Profile, profile),
        pins=load_pins(pins),
        env=env,
        stack_defaults=read_env_file(ENV_TEMPLATE),
        weights=weights,
    )
    typer.echo(format_report(results))
    if not all(result.ok for result in results):
        raise typer.Exit(code=1)


@app.command()
def run(
    config: Path = typer.Argument(..., help="experiments/configs/<name>.yaml"),
    allow_untagged: bool = typer.Option(
        False,
        "--allow-untagged",
        help="Run from an untagged or dirty tree (pilots only).",
    ),
) -> None:
    """Runs every cell of a config against the running stack."""
    from evaluation.runner import RunRefusedError, run_experiment

    try:
        run_dir = run_experiment(config, allow_untagged=allow_untagged)
    except RunRefusedError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"run -> {run_dir}")


@app.command()
def prep(
    slugs: list[str] = typer.Argument(None, help="Manuals to prepare; all by default."),
    sources: Path = typer.Option(Path("corpus/SOURCES.yaml")),
) -> None:
    """Verifies, cuts and extracts the doctrine manuals (see SOURCES.yaml)."""
    from evaluation.prep.manual import HashMismatchError, load_sources, prepare

    corpus = load_sources(sources)
    chosen = [corpus.get(slug) for slug in slugs] if slugs else corpus.documents
    failed = False
    for source in chosen:
        try:
            report = prepare(source)
        except (FileNotFoundError, HashMismatchError) as exc:
            typer.echo(f"{source.slug}: {exc}", err=True)
            failed = True
            continue
        pages = report["pages"]
        typer.echo(
            f"{source.slug}: body {pages['body']}, glossary {pages['glossary']}, "
            f"index {pages['index']}, entries {report.get('glossary_entries', '-')}/"
            f"{report.get('index_rows', '-')}"
            + (" (hand-checked assets kept)" if source.hand_checked else "")
        )
    if failed:
        raise typer.Exit(code=1)


@app.command()
def up() -> None:
    """Builds the service images from services/ and starts the stack."""
    # --wait returns once every service is healthy and every one-shot schema
    # step has exited, so a run can start right after.
    completed = subprocess.run([*COMPOSE, "up", "-d", "--build", "--wait"])
    raise typer.Exit(code=completed.returncode)


@app.command()
def down(
    volumes: bool = typer.Option(
        False, "--volumes", help="Also delete the stack's databases and caches."
    ),
) -> None:
    """Stops the stack; its data survives unless --volumes is given."""
    args = [*COMPOSE, "down", *(["--volumes"] if volumes else [])]
    raise typer.Exit(code=subprocess.run(args).returncode)
