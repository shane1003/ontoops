"""Response DTO for /target — which stores this instance writes to."""

from __future__ import annotations

from pydantic import BaseModel


class PostgresTarget(BaseModel):
    host: str | None
    port: int | None
    dbname: str | None
    user: str | None


class FalkorTarget(BaseModel):
    host: str
    port: int
    graph: str


class TargetResponse(BaseModel):
    postgres: PostgresTarget
    falkordb: FalkorTarget | None  # null when no default graph is configured
    # The *default* namespace only. A load's ?base_iri= beats it, and this endpoint cannot
    # know what a later request will pass — so it is named for what it is. The namespace a
    # load actually used comes back in that load's own response (LoadResponse.base_iri).
    default_base_iri: str | None
    # Where that default came from. This service has no built-in fallback — a load with no
    # namespace fails rather than inventing one — so the value is "env" or null, never
    # "built-in". Sibling services do carry a built-in default, and a consumer reading both
    # needs to tell "nothing configured" from "configured elsewhere in code" without
    # special-casing which service it is talking to.
    default_base_iri_source: str | None
