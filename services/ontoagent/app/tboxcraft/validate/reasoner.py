"""HermiT reasoning via owlready2.

Requires a JVM. owlready2 cannot parse Turtle, so callers pass RDF/XML. An inconsistent
ontology raises ``OwlReadyInconsistentOntologyError``; unsatisfiable (incoherent) classes do
NOT raise, so we enumerate them separately via ``world.inconsistent_classes()``.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReasonerResult:
    consistent: bool | None
    unsatisfiable_classes: list[str] = field(default_factory=list)
    error: str | None = None


def run_hermit(
    rdfxml_path: str | Path, *, java_exe: str | None = None
) -> ReasonerResult:
    """Run HermiT on an RDF/XML ontology file. Uses an isolated owlready2 ``World``."""
    import owlready2
    from owlready2 import OwlReadyInconsistentOntologyError, World

    if java_exe:
        owlready2.JAVA_EXE = java_exe

    abspath = Path(rdfxml_path).resolve()
    world = World()
    try:
        onto = world.get_ontology(abspath.as_uri()).load()
        with onto:
            owlready2.sync_reasoner_hermit(world)
    except OwlReadyInconsistentOntologyError:
        return ReasonerResult(consistent=False)
    except Exception as exc:  # noqa: BLE001 - reasoner/JVM failure surfaces as an error
        # Consistency is *unknown* (not proven) when the reasoner cannot run — never claim
        # consistent=True here, or gate() would fail open and commit an unchecked ontology.
        return ReasonerResult(consistent=None, error=f"{type(exc).__name__}: {exc}")

    unsat = [c.iri for c in world.inconsistent_classes()]
    return ReasonerResult(consistent=True, unsatisfiable_classes=sorted(unsat))


def run_hermit_on_rdfxml(rdfxml: str, *, java_exe: str | None = None) -> ReasonerResult:
    """Convenience wrapper: reason over an RDF/XML string via a temp file."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".owl", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(rdfxml)
        tmp = fh.name
    try:
        return run_hermit(tmp, java_exe=java_exe)
    finally:
        Path(tmp).unlink(missing_ok=True)
