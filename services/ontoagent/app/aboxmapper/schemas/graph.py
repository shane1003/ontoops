"""Response DTO for the TBox graph endpoints."""

from __future__ import annotations

from pydantic import BaseModel


class GraphReport(BaseModel):
    graph: str  # the named graph actually written (echoed: a typo in
    # ?graph= silently creates a new one, so it must be visible)
    # Source of the graph name (request | assembled | env); same shape as base_iri_source.
    # "assembled" when the server assembled it from pieces (?domain=&document=).
    graph_source: str | None = None
    classes: int
    subclass_edges: int
    object_edges: int  # (:Class)-[:OBJECT_PROP]->(:Class), one per domain×range
    skipped_props: int  # OBJECT_PROP with no class domain/range → no edge (visible)
    statements: int
