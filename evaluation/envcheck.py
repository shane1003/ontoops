"""Environment checks for the ``run`` and ``score`` profiles.

``run`` is what executing an experiment needs: Docker with the Compose plugin,
the three service sources vendored under ``services/``, the stack images
frozen by digest, a ``.env`` with the machine's addresses and the production
stack settings of ``.env.example`` unchanged, and the two LLM servers serving
the pinned weights. ``score`` is what recomputing a published
number needs, and it deliberately needs nothing beyond the repository and the
stored artifacts, so that anyone can reproduce a table without the stack.

Every probe (PATH lookup, HTTP GET, source lookup) is injectable, so the decision
logic is unit-tested without touching the machine.
"""

import json
import shutil
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

Profile = Literal["run", "score"]

# The two LLM servers, by the .env keys that address them. Production runs CQ
# generation and ontoagent against differently configured servers.
LLM_ENDPOINTS: tuple[tuple[str, str, str], ...] = (
    ("llm_cq_generation", "CQGEN_LLM_BASE_URL", "CQGEN_LLM_MODEL"),
    ("llm_ontoagent", "ONTOAGENT_LLM_BASE_URL", "ONTOAGENT_LLM_MODEL"),
)
REQUIRED_ENV: tuple[str, ...] = (
    "POSTGRES_PASSWORD",
    "ONTOAGENT_LLM_ROOT_URL",
    *(key for _, base, model in LLM_ENDPOINTS for key in (base, model)),
)
# The production settings the stack runs with. Their values live in
# .env.example (tracked); a .env that differs would change the method without
# a trace, so the run profile compares the two and every manifest records them.
STACK_SETTINGS: tuple[str, ...] = (
    "EMBEDDING_MODEL_ID",
    "EMBEDDING_REVISION",
    "EMBEDDING_DTYPE",
    "CQ_CHUNK_CHAR_BUDGET",
    "CQ_GENERATOR_MAX_TOKENS",
    "CONCEPTUALIZER_DOMAIN",
    "CONCEPTUALIZER_CHUNK_CONCURRENCY",
    "TBOXCRAFT_VERIFY_CONCURRENCY",
    "TBOXCRAFT_REASONER_ENABLED",
    "TBOXCRAFT_OOPS_ENABLED",
    "KEY_INDEX_REFRESH_INTERVAL_S",
)


@dataclass(frozen=True)
class CheckResult:
    """One check: what it is, whether it passed, where it looked, how to fix."""

    name: str
    ok: bool
    where: str
    hint: str


class ServicePin(BaseModel):
    """A service the stack runs, vendored under ``services/<name>/``.

    Attributes:
        repo: Name of the upstream repository the source was copied from.
        commit: The upstream commit the vendored source is an exact copy of.
        image: A registry image by digest, or None while images are built
            locally from the vendored source.
    """

    repo: str
    commit: str
    image: str | None = None


class ImagePin(BaseModel):
    """A third-party image of the stack, frozen by digest."""

    image: str


class Pins(BaseModel):
    """Contents of ``experiments/pins.yaml``."""

    services: dict[str, ServicePin]
    stack: dict[str, ImagePin]


def load_pins(path: Path) -> Pins:
    """Reads and validates the pin file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Pins.model_validate(raw)


def read_env_file(path: Path) -> dict[str, str]:
    """Reads ``KEY=value`` lines the way docker compose does; no expansion.

    Args:
        path: Usually ``.env`` at the repository root.

    Returns:
        The assignments; an empty dict when the file does not exist.
    """
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def host_view(url: str) -> str:
    """Rewrites a container-side address to the one the host itself uses."""
    return url.replace("host.docker.internal", "localhost")


def has_build_context(path: Path) -> bool:
    """Tells whether ``path`` is a service source Docker can build."""
    return (path / "Dockerfile").is_file()


def _http_get(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
            body: str = response.read().decode("utf-8", errors="replace")
    except Exception:
        # Any network or HTTP failure means "not reachable" for our purposes.
        return None
    return body


def _served(body: str | None, model: str, weights: str) -> tuple[bool, str]:
    """Checks a ``/v1/models`` body for the alias and the weights behind it."""
    if body is None:
        return False, "unreachable"
    try:
        models = json.loads(body).get("data", [])
    except (ValueError, AttributeError):
        return False, "not an OpenAI-compatible model list"
    for entry in models:
        if entry.get("id") == model:
            root = entry.get("root") or entry.get("id")
            if root != weights:
                return False, f"serves {root}, pinned {weights}"
            return True, f"max_model_len={entry.get('max_model_len')}"
    return False, f"model {model!r} not listed"


def run_checks(
    profile: Profile,
    *,
    pins: Pins,
    env: Mapping[str, str],
    stack_defaults: Mapping[str, str],
    weights: str,
    services_root: Path = Path("services"),
    http_get: Callable[[str], str | None] = _http_get,
    which: Callable[[str], str | None] = shutil.which,
    has_source: Callable[[Path], bool] = has_build_context,
) -> list[CheckResult]:
    """Evaluates every check of a profile.

    Args:
        profile: ``run`` to execute experiments, ``score`` to recompute only.
        pins: Output of ``load_pins``.
        env: The ``.env`` assignments, overridden by the process environment.
        stack_defaults: The ``.env.example`` assignments, which hold the
            production value of every ``STACK_SETTINGS`` key.
        weights: Model weights both LLM servers must serve (their ``root``).
        services_root: Directory holding the vendored service sources.
        http_get: Probe returning the response body or ``None``.
        which: Probe for executables on PATH.
        has_source: Probe telling whether a directory is a build context.

    Returns:
        Results in a stable order; each failure carries a fix hint.
    """
    results: list[CheckResult] = []
    if profile == "score":
        return results

    docker = which("docker")
    results.append(
        CheckResult(
            "docker",
            docker is not None,
            docker or "PATH",
            "Install Docker Engine with the Compose plugin.",
        )
    )

    for name, pin in pins.services.items():
        source = services_root / name
        results.append(
            CheckResult(
                f"source:{name}",
                has_source(source),
                str(source),
                f"Restore {source}/ from this repository; it is the vendored copy "
                f"of {pin.repo} at {pin.commit}.",
            )
        )

    loose = sorted(
        name for name, pin in pins.stack.items() if "@sha256:" not in pin.image
    )
    results.append(
        CheckResult(
            "stack_digests",
            not loose,
            "experiments/pins.yaml",
            f"Pin these images by digest: {', '.join(loose)}.",
        )
    )

    missing = [key for key in REQUIRED_ENV if not env.get(key)]
    results.append(
        CheckResult(
            "env",
            not missing,
            ".env",
            f"Copy .env.example to .env and set: {', '.join(missing)}.",
        )
    )

    drift = [
        key
        for key in STACK_SETTINGS
        if not stack_defaults.get(key) or env.get(key) != stack_defaults.get(key)
    ]
    results.append(
        CheckResult(
            "stack_settings",
            not drift,
            ".env",
            f"Set {', '.join(drift)} to the value in .env.example; "
            f"a deliberate deviation is an edit to .env.example.",
        )
    )

    for name, base_key, model_key in LLM_ENDPOINTS:
        base = env.get(base_key)
        model = env.get(model_key, "")
        if not base:
            continue  # already reported by the env check
        url = host_view(base).rstrip("/") + "/models"
        ok, detail = _served(http_get(url), model, weights)
        results.append(
            CheckResult(
                name,
                ok,
                url,
                f"{detail}. Serve {weights} as {model!r} with vLLM at {base_key}.",
            )
        )
    return results


def format_report(results: list[CheckResult]) -> str:
    """Renders one line per check, with a hint under each failure."""
    if not results:
        return "[OK] score profile needs only this repository and stored artifacts"
    lines: list[str] = []
    for result in results:
        mark = "OK" if result.ok else "FAIL"
        lines.append(f"[{mark}] {result.name}  ({result.where})")
        if not result.ok:
            lines.append(f"       hint: {result.hint}")
    return "\n".join(lines)
