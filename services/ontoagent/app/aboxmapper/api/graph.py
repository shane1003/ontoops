"""TBox graph endpoints: build the FalkorDB graph from a Turtle document."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.iri import InvalidBaseIri, assemble_graph_name

from ..schemas.graph import GraphReport
from ..services import graph

router = APIRouter()


def _graph_for(
    graph_name: str | None, domain: str | None, document: str | None
) -> tuple[str | None, str]:
    """Resolves this request's graph name and its source (request | assembled | env).

    Same contract as the IRI piece path (same iri_domain/document pieces): when
    ``document`` comes, the name is assembled by the server rule (``{domain}/{document}``
    of app.core.iri); if ``?graph=`` comes too, they are compared and a mismatch is a 422.
    If the server assembles the IRI but the graph name stays in a client template, the two
    rules drift apart.
    """
    if document is None:
        return graph_name, ("request" if graph_name else "env")
    try:
        assembled = assemble_graph_name(domain or "", document)
    except InvalidBaseIri as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if graph_name is not None and graph_name != assembled:
        raise HTTPException(
            status_code=422,
            detail="graph disagrees with the server-assembled name — the caller's "
            f"assembly rule has drifted: sent {graph_name!r}, assembled {assembled!r}",
        )
    return assembled, ("request" if graph_name else "assembled")


def _build_tbox(
    ttl: bytes, graph_name: str | None = None, graph_source: str = "request"
) -> GraphReport:
    try:
        ttl_text = ttl.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"body is not valid UTF-8: {exc}") from exc
    try:
        tbox = graph.parse_tbox_text(ttl_text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid TTL: {exc}") from exc
    try:
        fg = graph.connect_falkordb(graph_name)
    except RuntimeError as exc:  # no ?graph= and FALKORDB_GRAPH unset
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    try:
        report = graph.load(fg, tbox)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"falkordb error: {exc}") from exc
    return GraphReport(graph=fg.name, graph_source=graph_source, **report)


@router.post("/graph/tbox", response_model=GraphReport)
async def build_tbox(
    request: Request,
    graph_name: str | None = Query(None, alias="graph"),  # alias: don't shadow the module
    iri_domain: str | None = Query(None),
    document: str | None = Query(None),
) -> GraphReport:
    """Build the TBox graph in FalkorDB from a Turtle document sent as the request body.

    ``?graph=`` picks the target named graph for this request; without it the deployment
    default ``FALKORDB_GRAPH`` is used. The response echoes the graph actually written and
    its source. Send the ``?iri_domain=…&document=…`` pieces and the server assembles the
    graph name (``{domain}/{document}``); if ``graph`` comes too, they are compared and a
    mismatch is a 422.

    Only the body receive is async: parsing and the graph build are blocking, so they go to
    the threadpool; run on the event loop, they would stall the whole app (including
    /health) for the duration of the load (app.main policy).
    """
    name, source = _graph_for(graph_name, iri_domain, document)
    body = await request.body()
    return await run_in_threadpool(_build_tbox, body, name, source)


@router.post("/graph/tbox/upload", response_model=GraphReport)
async def build_tbox_upload(
    file: UploadFile = File(...),
    graph_name: str | None = Query(None, alias="graph"),
    iri_domain: str | None = Query(None),
    document: str | None = Query(None),
) -> GraphReport:
    """Build the TBox graph in FalkorDB from an uploaded .ttl file (params as ``/graph/tbox``)."""
    name, source = _graph_for(graph_name, iri_domain, document)
    ttl = await file.read()
    return await run_in_threadpool(_build_tbox, ttl, name, source)
