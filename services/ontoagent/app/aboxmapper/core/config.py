"""Configuration for aboxmapper: base IRI and FalkorDB targets.

The Postgres DSN is resolved by the repo-wide helper ``app.core.db`` (env var
``TBOXCRAFT_DATABASE_URL``), not here.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# The validation rule lives in one repository-wide place (app.core.iri); the existing
# import path is kept as a re-export.
from app.core.iri import InvalidBaseIri as InvalidBaseIri
from app.core.iri import validate_base_iri as validate_base_iri


class BaseIriRequired(RuntimeError):
    """No base IRI was given for this call and none is configured."""


def get_ontology_base_iri(base_iri: str | None = None) -> str:
    """Base IRI that individual IRIs are minted under.

    ``base_iri`` names the namespace for this call and **wins over** ``ONTOLOGY_BASE_IRI``,
    which stays the deployment default — one document is one namespace, so the caller that
    knows which document this is must be able to say so (same policy as the FalkorDB graph).
    Whichever wins is validated the same way; nothing is normalised.
    """
    if base_iri is None:
        load_dotenv()
        base_iri = os.environ.get("ONTOLOGY_BASE_IRI")
    if not base_iri:
        raise BaseIriRequired(
            "no base IRI: pass one for this call or set ONTOLOGY_BASE_IRI, e.g. "
            "http://example.org/military# or http://example.org/military"
        )
    return validate_base_iri(base_iri)


def get_falkordb_config(graph: str | None = None) -> dict:
    """Return FalkorDB connection settings from the environment.

    ``graph`` names the target graph for this call and **wins over** ``FALKORDB_GRAPH``,
    which stays the deployment default — so one instance can serve several documents
    instead of needing one container per named graph. The env var is required only when no
    caller supplies a graph.

    FALKORDB_HOST defaults to 'falkordb', FALKORDB_PORT to 6379. FALKORDB_PASSWORD is
    optional: unset (``None``) means no authentication — right for a local/isolated stack;
    set it for a shared instance (e.g. multi-consumer) that requires auth.
    """
    load_dotenv()
    graph = graph or os.environ.get("FALKORDB_GRAPH")
    if not graph:
        raise RuntimeError(
            "FALKORDB_GRAPH is not set and no graph was given for this call. It names the "
            "graph to write this run/document to, e.g. ipb | adp."
        )
    return {
        "host": os.environ.get("FALKORDB_HOST", "falkordb"),
        "port": int(os.environ.get("FALKORDB_PORT", "6379")),
        "graph": graph,
        "password": os.environ.get("FALKORDB_PASSWORD") or None,
    }
