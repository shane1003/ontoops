"""Syntax parse + lightweight OWL 2 DL profile check.

The DL check targets the punning class that HermiT-via-owlready2 handles poorly: an entity
declared as more than one of {class, object property, data property}. Catching it here (before
the reasoner) fails deterministically instead of relying on undefined OWL-Full behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rdflib import Graph
from rdflib.namespace import OWL, RDF


@dataclass
class SyntaxResult:
    ok: bool
    error: str | None = None
    graph: Graph | None = None


def parse_ttl(path: str | Path) -> SyntaxResult:
    try:
        graph = Graph().parse(str(path), format="turtle")
    except Exception as exc:  # noqa: BLE001 - surface any parse failure as a result
        return SyntaxResult(ok=False, error=f"{type(exc).__name__}: {exc}")
    return SyntaxResult(ok=True, graph=graph)


@dataclass
class DLProfileResult:
    ok: bool
    violations: list[str] = field(default_factory=list)


_TYPED = (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty)


def dl_profile_check(graph: Graph) -> DLProfileResult:
    """Flag entities typed as more than one of class / object-property / data-property."""
    kinds: dict[str, set] = {}
    for entity_type in _TYPED:
        for subject in graph.subjects(RDF.type, entity_type):
            kinds.setdefault(str(subject), set()).add(entity_type)
    violations = sorted(iri for iri, types in kinds.items() if len(types) > 1)
    return DLProfileResult(ok=not violations, violations=violations)
