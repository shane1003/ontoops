"""tboxcraft router: exposes the generate pipeline and term resolution as an APIRouter.

app.main includes it with prefix="/tbox" and combines the lifespan (the move into the unified app:
the create_app closure of the old service/app.py turned into module functions). The state
(config, repo, provider, llm, model_id, model_mean, tracer) lives on app.state under the
same names as before, and handlers read it via request.app.state; this keeps the tests'
pre-seed pattern (app.state.repo = ...).

`/generate` is a sync `def` so Starlette runs the reasoner-heavy pipeline in a threadpool
without blocking the event loop. `/resolve` does exact surface->IRI plus optional vector
k-NN; a query *term* is embedded via the remote-e5 provider held in app state.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.core import db as core_db
from app.core.logging_setup import configure_logging
from app.core.iri import resolve_base_iri
from app.tboxcraft.config import Config
from app.tboxcraft.pipeline.generate import generate
from app.tboxcraft.pipeline.update_apply import update
from app.tboxcraft.review import apply_review_decision, list_pending, verify_pending
from app.tboxcraft.similarity import center_query
from app.core.tracing import Tracer, get_tracer

logger = logging.getLogger(__name__)


# The IRI namespace this batch's terms belong to. One document is one KG, so the namespace is a
# property of the *document*, not of the service — a single env var cannot serve two documents
# ingesting into one store. The env value stays as the default so an existing caller that sends
# nothing keeps its behaviour; a request that sends one wins.
# Sending it: the trailing '#' MUST be percent-encoded as %23 in a URL. A raw '#' is a fragment
# delimiter, so curl and friends cut the value there and the field arrives empty — the batch then
# silently falls back to the env default, i.e. lands in the wrong namespace with no error anywhere.
#   curl  ...?base_iri=http%3A%2F%2Fexample.org%2FIPB%23
#   requests / httpx: params={"base_iri": "http://example.org/IPB#"} encodes it for you.
# In a JSON body (the usual path for /update) no encoding is needed.
_BASE_IRI_FIELD = Field(
    default=None,
    description="IRI namespace for this batch (e.g. http://example.org/IPB#). "
    "Defaults to TBOXCRAFT_BASE_IRI. A trailing '#' selects hash-style minting. "
    "In a URL the '#' must be percent-encoded as %23, or it is cut as a fragment "
    "and this arrives empty.",
)


# The domain against which the verifier judges "is this a correct term in use in this
# domain". If the document brings its own domain, that is the most specific declaration
# and the deployment env is the fallback: the same path base_iri took (query and body both
# accepted, query first). Paired request: the moment conceptualizer reads
# ontology.domain, this value must pass through too, or the classifier and the verifier
# judge by different yardsticks.
_DOMAIN_FIELD = Field(
    default=None,
    description="Verifier domain for this batch. "
    "Defaults to TBOXCRAFT_DOMAIN_DESCRIPTION, then the built-in neutral label.",
)


# Piece path (send pieces instead of a full base_iri and the server assembles it with the
# app.core.iri rule): document is the trigger, the path piece is iri_domain, the root is
# iri_root → ONTOLOGY_IRI_ROOT env. If base_iri comes too, it is compared with the
# assembled one and a mismatch is a 422.
# iri_domain is a **different value** from _DOMAIN_FIELD above: domain is a description
# for the verifier and iri_domain is a path
# segment slug ("military"). Making one parameter serve both picks one of two defects:
# the description leaks into the path, or the slug displaces the verification
# criterion. So the two are kept apart.
_DOCUMENT_FIELD = Field(
    default=None,
    description="Document title piece for server-side namespace assembly "
    "({iri_root}/{iri_domain}/{encoded document}#). Needs iri_domain; percent-encoded "
    "here, so spaces/non-ASCII are fine.",
)
_IRI_DOMAIN_FIELD = Field(
    default=None,
    description="Namespace path segment (a bare slug, e.g. 'military'). Distinct from "
    "`domain`, which is the verifier's descriptive phrase and never enters the IRI.",
)
_IRI_ROOT_FIELD = Field(
    default=None,
    description="Scheme+host root piece (e.g. http://example.org, trailing slash "
    "tolerated). Defaults to ONTOLOGY_IRI_ROOT.",
)


class GenerateRequest(BaseModel):
    components_path: str
    base_iri: str | None = _BASE_IRI_FIELD
    domain: str | None = _DOMAIN_FIELD
    document: str | None = _DOCUMENT_FIELD
    iri_domain: str | None = _IRI_DOMAIN_FIELD
    iri_root: str | None = _IRI_ROOT_FIELD
    # None = follow the instance setting (TBOXCRAFT_REASONER_ENABLED / _OOPS_ENABLED). Explicit
    # true/false still wins, so an existing caller is unaffected — but the default is now an
    # operational decision made once, not one every caller re-makes per request.
    run_reasoner: bool | None = None
    run_oops: bool | None = None
    persist_db: bool = True
    verify: bool = True  # advisory Generate-time domain-term annotation; no gate


class UpdateRequest(BaseModel):
    # POST the components document inline ({components:[...]}; other top-level keys such as the
    # `individuals` block are ignored). `components` is required.
    # CQs are not sent here and tboxcraft does not load them: the upstream ingestion model owns
    # cq and must have written them before this runs. Components reference them by id
    # via source_cq_ids; an ingest that runs first stores terms without provenance (n_cq_missing).
    components: list[dict]
    base_iri: str | None = _BASE_IRI_FIELD
    domain: str | None = _DOMAIN_FIELD
    document: str | None = _DOCUMENT_FIELD
    iri_domain: str | None = _IRI_DOMAIN_FIELD
    iri_root: str | None = _IRI_ROOT_FIELD
    # Adapter-declared rdb sources as the adapter's sources.annotate() shape:
    # {database:{dsn_label, schema_name, ...}, sources:[{source_id, kind:'rdb', table_name, ...}]}.
    # The document path derives its sources from cq (sync_document_sources); rdb has no
    # derivation here, so the adapter sends them and tboxcraft registers the leaf tables
    # (rdb_database/table/ref) + source rows before any term link. Empty is a no-op.
    sources: dict = {}
    run_reasoner: bool | None = None  # see GenerateRequest
    run_oops: bool | None = None
    verify: bool = (
        True  # run the domain-term verifier over cluster reps at ingest
    )


class ReviewDecision(BaseModel):
    action: str  # merge | hide | new | relabel | delete | reject
    target_iri: str | None = None
    label: str | None = None
    run_reasoner: bool = True


class ResolveRequest(BaseModel):
    surface: str | None = None
    term: str | None = None
    vector: list[float] | None = None
    k: int = Field(default=5, ge=1, le=100)  # bounded so k<=0 can't produce LIMIT -1
    # Restrict both the exact and the k-NN lookup to one term type. Names are unique per
    # kind, not globally, so a caller that omits this can be handed either owner of a
    # dual-kind name. An individual anchors only to a class and should say so.
    kind: str | None = None  # "class" | "object_property" | "data_property"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Self-contained entry point: so it works when attached directly to the main app
    # without create_app, the state defaults (old create_app initialization) and logging
    # are ensured here. Both are idempotent.
    init_state(app)
    configure_logging(env_var="TBOXCRAFT_LOG_LEVEL")
    config: Config = app.state.config
    # Build the tracer here (not in create_app) so importing the module never spins up a
    # Langfuse client — only a served app does.
    app.state.tracer = get_tracer(config)
    # Build the DB pool + repository and the query-embedding provider unless a test
    # pre-seeded them. Degrade, never abort: this lifespan belongs to the WHOLE unified
    # app, so an exception here (dead DB at boot → PoolTimeout on the first checkout)
    # would take down /extract and /abox with it — modules that never asked for this
    # database at startup (standalone tboxcraft could fail closed; the merge changed the
    # blast radius). With repo=None every /tbox DB route already answers 503, which keeps
    # the misconfiguration loud where it belongs without silencing the other modules.
    if getattr(app.state, "repo", None) is None and config.database_url:
        from app.tboxcraft.db import Repository, shared_pool

        try:
            # Join the process-wide pool (app.core.db), not a dedicated one: share
            # connections with aboxmapper on the same DB and never hold two pools.
            app.state.pool = shared_pool(config)
            app.state.repo = Repository(app.state.pool)
            # Resolve the active model_id once (from config, not the query provider), so
            # vector k-NN works independent of whether the embedding provider loaded
            # This is also the first real connection checkout — a dead DB
            # surfaces here, not at pool construction.
            app.state.model_id = app.state.repo.get_model_id(
                config.e5_model, config.embedding_revision
            )
            app.state.model_mean = app.state.repo.get_model_mean(
                config.e5_model, config.embedding_revision
            )
        except Exception as exc:  # noqa: BLE001 - degrade but make the reason visible
            logger.critical(
                "tboxcraft DB unavailable at startup (%s) — /tbox DB routes will answer "
                "503 until restart; /extract and /abox are unaffected: %s",
                type(exc).__name__,
                exc,
            )
            pool = getattr(app.state, "pool", None)
            if pool is not None:
                pool.close()
            # Also forget the shared pool registration: a closed pool left behind kills
            # /abox too until a restart; forgotten, the single-connection fallback brings
            # it back when the DB recovers.
            core_db.close_pool()
            app.state.pool = None
            app.state.repo = None
            app.state.model_id = None
            app.state.model_mean = None
    elif getattr(app.state, "model_id", None) is None and app.state.repo is not None:
        # A test pre-seeded repo without model_id (the pre-merge shape) still resolves it.
        app.state.model_id = app.state.repo.get_model_id(
            config.e5_model, config.embedding_revision
        )
        app.state.model_mean = app.state.repo.get_model_mean(
            config.e5_model, config.embedding_revision
        )
    if getattr(app.state, "provider", None) is None:
        from app.tboxcraft.embed import make_provider

        try:
            app.state.provider = make_provider(config)
        except Exception as exc:  # noqa: BLE001 - degrade but make the reason visible
            logger.warning("embedding provider unavailable: %s", exc)
            app.state.provider = None  # /resolve term-embedding unavailable without it
    if getattr(app.state, "llm", None) is None:
        from app.tboxcraft.llm import make_llm

        try:
            app.state.llm = make_llm(config)
        except Exception as exc:  # noqa: BLE001 - verifier is optional; review still works
            logger.warning("LLM verifier unavailable: %s", exc)
            app.state.llm = None
    yield
    app.state.tracer.flush()  # drain buffered traces before exit
    pool = getattr(app.state, "pool", None)
    if pool is not None:
        pool.close()
    # A closed shared pool left registered is inherited by the next lifespan (tests) in
    # the same process; forget it here so the next open starts fresh.
    core_db.close_pool()


router = APIRouter()


def _active_model_id(request: Request) -> int | None:
    """Cached active embedding-model id + whitening mean (re-query once if unresolved)."""
    if request.app.state.model_id is None and request.app.state.repo is not None:
        cfg = request.app.state.config
        request.app.state.model_id = request.app.state.repo.get_model_id(
            cfg.e5_model, cfg.embedding_revision
        )
        request.app.state.model_mean = request.app.state.repo.get_model_mean(
            cfg.e5_model, cfg.embedding_revision
        )
    return request.app.state.model_id

def _flag(requested: bool | None, instance_default: bool) -> bool:
    """A request's explicit choice, else the instance setting. None means "not stated"."""
    return instance_default if requested is None else requested

def _domain_for(
    request: Request, req, query_domain: str | None
) -> tuple[str, str]:
    """Resolves this batch's verification domain and its source (request | env | built-in).

    Order: query → body → fallback. The request wins for the same reason as base_iri: a
    document that brings its own domain is the most specific declaration. An empty string
    folds to "not said" (same contract as env_or). Why the source is returned too: a value
    that never arrives silently falls back, so without domain_source in the response,
    "the request said so" and "not carried, deployment default" cannot be told apart from
    outside (exactly why default_base_iri_source exists).
    """
    supplied = query_domain or getattr(req, "domain", None)
    if supplied:
        return supplied, "request"
    cfg: Config = request.app.state.config
    source = "env" if os.environ.get("TBOXCRAFT_DOMAIN_DESCRIPTION") else "built-in"
    return cfg.domain_description, source


def _cfg_for(
    request: Request,
    req,
    query_base_iri: str | None = None,
    *,
    query_document: str | None = None,
    query_iri_root: str | None = None,
    query_iri_domain: str | None = None,
) -> tuple[Config, str, dict[str, str] | None]:
    """The run config with this request's namespace applied, or 422 saying what is wrong.

    Accepted from the query string *or* the JSON body, query first. Body-only was wrong: the
    policy this follows is aboxmapper's ``?graph=``, so callers send it in the URL — and a
    pydantic model field is read from the body ONLY, so a query value was dropped without a
    word and the batch silently fell back to the env namespace. Every term of two documents
    landed in one namespace and nothing in the response said so.

    Rejected rather than repaired: upstream has sent a base IRI in backticks and one carrying
    %20, and quietly trimming either mints every term of the batch under a namespace nobody
    asked for. An IRI is an identity, so the wrong one is not a cosmetic defect — it is a
    different ontology wearing the right name, and it is expensive to take back.

    Piece path: when ``document`` (+ ``domain``, ``iri_root``) comes, the assembly rule
    (app.core.iri) builds the base IRI; this path ends the client template string owning
    the rule. When a full IRI and pieces come together, they are compared and a mismatch
    is a 422 (dual-run check during the transition).

    Returns:
        ``(config, base_iri_source, base_iri_pieces)``: source is request | assembled |
        env | built-in; pieces echoes the pieces used for assembly (None if none came).
    """
    cfg = request.app.state.config
    supplied = query_base_iri if query_base_iri is not None else req.base_iri
    document = query_document if query_document is not None else getattr(req, "document", None)
    iri_root = query_iri_root if query_iri_root is not None else getattr(req, "iri_root", None)
    # The path piece uses only iri_domain, a different slot from the verifier's `domain`
    # (a description). The two meanings are split at the parameter level so neither the
    # env domain_description leaks into the path nor the slug displaces the criterion.
    iri_domain = (
        query_iri_domain if query_iri_domain is not None else getattr(req, "iri_domain", None)
    )
    try:
        resolved = resolve_base_iri(supplied, root=iri_root, domain=iri_domain, document=document)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if resolved is None:
        source = "env" if os.environ.get("TBOXCRAFT_BASE_IRI") else "built-in"
        return cfg, source, None
    base_iri, source, pieces = resolved
    return dataclasses.replace(cfg, base_iri=base_iri), source, pieces

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/target")
def target(request: Request):
    """Which database this instance actually writes to, and the namespace it defaults to.

    Read from a live connection (``conn.info``), never echoed from the environment. The two can
    disagree — an externally supplied DSN overrides the env, and then an env echo is a dashboard
    that lies about the one thing it exists to answer. A caller checking it is about to load
    data; being confidently wrong is worse than being unavailable.

    Which is why an unreachable database is a 503 rather than a null ``postgres`` block. This
    exists so a pipeline can compare tboxcraft's ``dbname`` with aboxmapper's before ingesting
    — they are joined by a foreign key and must be the same database — and ``None == None``
    would make that comparison pass while neither side had actually been read.

    No password: ``conn.info`` carries one and it is deliberately not in the fields below.
    """
    repo = request.app.state.repo
    if repo is None:
        raise HTTPException(status_code=503, detail="database unavailable")
    try:
        with repo.pool.connection() as conn:
            info = conn.info
            postgres = {
                "host": info.host,
                "port": info.port,
                "dbname": info.dbname,
                "user": info.user,
            }
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"database unreachable: {exc}"
        ) from exc
    cfg = request.app.state.config
    return {
        "postgres": postgres,
        # The namespace a batch gets when it sends none — NOT the namespace of any particular
        # batch, which travels in the /update response instead. The two are separate names on
        # purpose: comparing "what this instance defaults to" against "what that batch used" is
        # the loop that catches a base_iri which never arrived.
        "default_base_iri": cfg.base_iri,
        # Whether that default came from TBOXCRAFT_BASE_IRI or from the built-in fallback.
        # Reported rather than returning null when unset, because a batch sending no base_iri
        # does not get "nothing" — it gets the built-in, and calling that null would hide a
        # real namespace behind a word meaning "none".
        "default_base_iri_source": (
            "env" if os.environ.get("TBOXCRAFT_BASE_IRI") else "built-in"
        ),
        # The criterion the verifier judges "is this a correct term of {domain}" by. A
        # wrong domain raises nothing and only lowers judging quality, a silent defect, so
        # it is carried for a harness to check against the runtime fact, not an env echo
        # (same reason as default_base_iri above).
        "domain_description": cfg.domain_description,
    }

@router.post("/generate")
def run_generate(
    request: Request,
    req: GenerateRequest,
    base_iri: str | None = Query(default=None),
    domain: str | None = Query(default=None),
    document: str | None = Query(default=None),
    iri_domain: str | None = Query(default=None),
    iri_root: str | None = Query(default=None),
):
    cfg, base_iri_source, base_iri_pieces = _cfg_for(
        request,
        req,
        base_iri,
        query_document=document,
        query_iri_root=iri_root,
        query_iri_domain=iri_domain,
    )
    domain_used, domain_source = _domain_for(request, req, domain)
    cfg = dataclasses.replace(cfg, domain_description=domain_used)
    # Fail loudly rather than silently writing files when DB persistence was requested.
    if req.persist_db and request.app.state.repo is None:
        raise HTTPException(
            status_code=503,
            detail="persistence requested but database is unavailable",
        )
    # /generate persists a FIRST batch into an empty store: replace_topology rewrites the whole
    # structure from this batch's graph, so running it against a store that already holds an
    # ontology would drop every earlier edge and keep only this batch's. The warm path (adding
    # to an existing ontology) is /update — which bootstraps through generate() internally only
    # when the store is empty, so this guard sits on the endpoint, not on that call.
    if (
        req.persist_db
        and request.app.state.repo is not None
        and request.app.state.repo.get_latest_ontology(cfg.base_iri) is not None
    ):
        raise HTTPException(
            status_code=409,
            detail="an ontology already exists; /generate targets an empty store — "
            "use /update to extend an existing ontology",
        )
    # Isolate each request's artifacts so concurrent /generate calls don't clobber output.
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    req_cfg = dataclasses.replace(
        cfg, output_dir=tempfile.mkdtemp(prefix="gen_", dir=cfg.output_dir)
    )
    result = generate(
        req.components_path,
        req_cfg,
        run_reasoner=_flag(req.run_reasoner, cfg.reasoner_enabled),
        run_oops=_flag(req.run_oops, cfg.oops_enabled),
        persist_db=req.persist_db,
        repo=request.app.state.repo,
        verify=req.verify,
        llm=request.app.state.llm,
    )
    if not result.ok:
        raise HTTPException(
            status_code=422, detail={"gate_reasons": result.gate_reasons}
        )
    return {
        "ok": result.ok,
        "ttl_path": str(result.ttl_path) if result.ttl_path else None,
        "persisted": result.persisted,
        "n_components": result.n_components,
        "n_concepts": result.n_concepts,
        "mapping_delta": result.mapping_delta,  # per-component raw name -> minted IRI
        # Verifier health: a down/misconfigured LLM degrades per concept and still returns
        # ok=true, so the caller needs these to tell "verified" from "silently unverified".
        "n_verified": result.n_verified,
        "n_verifier_unavailable": result.n_verifier_unavailable,
        "verifier_degraded": result.verifier_degraded,
        # What verification cost: batches formed and LLM round trips actually spent. Reported,
        # never asserted on — they say how the verdicts were obtained, not what they say.
        "n_clusters": result.n_clusters,
        "n_verify_calls": result.n_verify_calls,
        # The merge threshold actually in force, per kind, with the sample it came from.
        # A partition whose source left "seed" is one whose clustering is no longer governed
        # by a vetted number — the caller sees that before reading the concept count.
        "calibration": result.calibration,
        "n_chained_refused": result.n_chained_refused,
        # Strings that name more than one concept. A caller that sees an individual
        # dropped as ambiguous downstream finds the reason here and nowhere else.
        "surface_collisions": result.surface_collisions,
        # Detection vs action: which collisions actually reached the queue.
        "queued_surface_collisions": result.queued_surface_collisions,
        "pref_label_collisions": result.pref_label_collisions,
        "queued_pref_label_collisions": result.queued_pref_label_collisions,
        # One name minted as two kinds — both readings asserted at once. Reported and
        # queued separately, so "found" and "acted on" stay distinguishable.
        # The namespace this batch actually used. Echoed because the failure mode is
        # silent: a base_iri that does not arrive (wrong place in the request, a raw
        # '#' cut as a fragment) falls back to the env default, and every term lands
        # in the wrong namespace with nothing in the response saying so. One field
        # makes that checkable instead of inferable from minted IRIs.
        "base_iri": cfg.base_iri,
        # Echo of the source (request | assembled | env | built-in) and assembly pieces. A
        # caller that sent pieces can keep checking "were my domain/document used as
        # sent"; echoing only the full IRI would leave the piece path nothing to compare
        # against (same shape as domain_source).
        "base_iri_source": base_iri_source,
        "base_iri_pieces": base_iri_pieces,
        # The domain this batch was actually judged under and its source
        # (request | env | built-in). /tbox/target's domain_description is the
        # **default**; what this batch was judged under is reported only here. A value
        # that never arrives silently falls back, so it is echoed for exactly the same
        # reason base_iri is.
        "domain": cfg.domain_description,
        "domain_source": domain_source,
        # How the gate decided, not just that it passed. ok=true with
        # reasoner_ran=false is a TBox nobody proved consistent.
        "validation": result.validation,
        "kind_conflicts": result.kind_conflicts,
        "queued_kind_conflicts": result.queued_kind_conflicts,
        # Hash namespace only: neither side of a contested name is minted (see /update).
        "n_kind_conflicts_withheld": result.n_kind_conflicts_withheld,
        "n_terms_withheld": result.n_terms_withheld,
        "withheld_kind_conflicts": result.withheld_kind_conflicts,
        # Names whose slugs collide, so one IRI would name several terms. Both sides are
        # withheld: an identity, unlike a label, cannot be un-merged by a later review.
        "n_slug_collisions_withheld": result.n_slug_collisions_withheld,
        "n_slug_terms_withheld": result.n_slug_terms_withheld,
        "withheld_slug_collisions": result.withheld_slug_collisions,
        # Curated terms that could not be the representative label (it is one per term and English)
        # and stayed as alternative surfaces instead. Not a failure, but "the curator's term is not
        # this term's displayed name" has to be visible, and `label_is_english` false marks the
        # worse shape where the fallback was not English either.
        "non_english_labels": result.non_english_labels,
        # Surfaces whose producer declared `en` over a string with no Latin letter. The
        # declaration is refused and written `und`; the count is the message back to the
        # producer, and without it "they fixed it" and "we are still covering for them" are
        # the same picture.
        "n_lang_declarations_overridden": result.n_lang_declarations_overridden,
        "overridden_lang_declarations": result.overridden_lang_declarations,
        "folded_names": result.folded_names,
        "refused_duplicates": result.refused_duplicates,
        "n_hidden_surfaces": result.n_hidden_surfaces,
        # Upstream TBox/ABox verdict routing (a component marked abox breaks the contract).
        "n_tbox": result.n_tbox,
        "n_conflict": result.n_conflict,
        "n_abox": result.n_abox,
        "n_verdict_queued": result.n_verdict_queued,
        "n_review_rows": result.n_review_rows,
        # What this run dropped or altered. Reported so a caller can tell an incomplete load
        # from a clean one; a missing CQ link in particular only fails one stage later.
        "n_cq_linked": result.n_cq_linked,
        "n_cq_missing": result.n_cq_missing,
        "cq_provenance_degraded": result.cq_provenance_degraded,
        "n_hierarchy_skipped": result.n_hierarchy_skipped,
        "hierarchy_cycles_broken": result.hierarchy_cycles_broken,
        "n_domain_range_skipped": result.n_domain_range_skipped,
        "datatype_substitutions": result.datatype_substitutions,
        "unresolved_broaders": result.unresolved_broaders,
        # Both: the count is what a dashboard tracks, the rows are what tells a
        # partial resolution (range narrowed silently) from a total one.
        "n_unresolved_ranges": len(result.unresolved_ranges),
        "unresolved_ranges": result.unresolved_ranges,
        # The domain counterpart of ranges: makes visible the loss where multi-value
        # domains were dropped whole. Empty means no rdfs:domain; non-empty means a
        # narrower claim than the input (partial resolution).
        "n_unresolved_domains": len(result.unresolved_domains),
        "unresolved_domains": result.unresolved_domains,
        # Provenance counters. A broken source raises nothing — it shows only as a
        # count, so a caller can pin the two invariants at 0. n_unattributed_edges is a
        # merge-only artifact (non-zero is normal); its size is the quality signal.
        "n_terms_without_source": result.n_terms_without_source,
        "n_surface_forms_skipped": result.n_surface_forms_skipped,
        "n_edge_sources_skipped": result.n_edge_sources_skipped,
        "n_unattributed_edges": result.n_unattributed_edges,
    }

@router.post("/update")
def run_update(
    request: Request,
    req: UpdateRequest,
    base_iri: str | None = Query(default=None),
    domain: str | None = Query(default=None),
    document: str | None = Query(default=None),
    iri_domain: str | None = Query(default=None),
    iri_root: str | None = Query(default=None),
):
    # Admin/ingest side: route a batch against the accumulated ontology (bootstraps if empty).
    # Parameter validation (including the namespace check) comes **before** the DB
    # availability check, the same order as /abox/load. A mismatched assembly must be
    # caught as a 422 even when the DB is down, so the transition's dual-run check can be
    # relied on to "always run" (hidden behind a 503, it would look passed without
    # having run).
    cfg, base_iri_source, base_iri_pieces = _cfg_for(
        request,
        req,
        base_iri,
        query_document=document,
        query_iri_root=iri_root,
        query_iri_domain=iri_domain,
    )
    domain_used, domain_source = _domain_for(request, req, domain)
    if request.app.state.repo is None:
        raise HTTPException(status_code=503, detail="database unavailable")
    cfg = dataclasses.replace(cfg, domain_description=domain_used)
    result = update(
        None,
        cfg,
        repo=request.app.state.repo,
        provider=request.app.state.provider,
        llm=request.app.state.llm,
        verify=req.verify,
        run_reasoner=_flag(req.run_reasoner, cfg.reasoner_enabled),
        run_oops=_flag(req.run_oops, cfg.oops_enabled),
        components_doc={"components": req.components},
        sources=req.sources,
    )
    if not result.ok:
        raise HTTPException(
            status_code=422,
            detail={"gate_reasons": result.gate_reasons},
        )
    return {
        "ok": True,
        "ttl": result.ttl,  # the improved TBox for this batch (serialized Turtle)
        "version_id": result.version_id,
        # Routing outcome of every candidate, and the raw-name -> IRI rows a caller
        # reconciles with (mapping_delta resolvable now, pending sent to review).
        "n_candidates": result.n_candidates,
        "n_new": result.n_new,
        "n_merged": result.n_merged,
        "n_relabel": result.n_relabel,
        "n_gray": result.n_gray,
        "mapping_delta": result.mapping_delta,
        "pending": result.pending,
        # Components that never became candidates: withheld (kind or slug collisions) or
        # folded into another name. With n_candidates they account for every component.
        "n_terms_withheld": result.n_terms_withheld,
        "n_slug_terms_withheld": result.n_slug_terms_withheld,
        "folded_names": result.folded_names,
        "n_unresolved_ranges": len(result.unresolved_ranges),
        "n_unresolved_domains": len(result.unresolved_domains),
        # CQ provenance: a cited CQ missing from the store only fails one stage later.
        "n_cq_linked": result.n_cq_linked,
        "n_cq_missing": result.n_cq_missing,
        # The namespace and domain this batch actually used, with their sources. A value
        # that does not arrive falls back silently, so the caller checks these echoes.
        "base_iri": cfg.base_iri,
        "base_iri_source": base_iri_source,
        "base_iri_pieces": base_iri_pieces,
        "domain": cfg.domain_description,
        "domain_source": domain_source,
        # How the gate decided (ok with reasoner_ran=false is a TBox nobody proved
        # consistent) and whether the verifier answered or degraded silently.
        "validation": result.validation,
        "n_verified": result.n_verified,
        "n_verifier_unavailable": result.n_verifier_unavailable,
        "verifier_degraded": result.verifier_degraded,
    }

@router.get("/review/pending")
def review_pending(request: Request, limit: int = 100, base_iri: str | None = Query(default=None)):
    # Human-workflow side: list queued gray/relabel/delete items (with any LLM verdict).
    # base_iri narrows to one document's queue: a reviewer works on one graph, and "is this the
    # same concept as that one" has a different answer in a different one. Omitted lists all,
    # which is what an operator surveying the whole queue wants.
    if request.app.state.repo is None:
        raise HTTPException(status_code=503, detail="database unavailable")
    return {"pending": list_pending(request.app.state.repo, limit, base_iri=base_iri)}

@router.post("/review/verify")
def review_verify(request: Request, limit: int = 100):
    # Attach the advisory domain-term verifier verdict to pending Major-Update items.
    # The human decision stays authoritative; this only adds context.
    if request.app.state.repo is None:
        raise HTTPException(status_code=503, detail="database unavailable")
    if request.app.state.llm is None:
        raise HTTPException(status_code=503, detail="LLM verifier unavailable")
    n = verify_pending(
        request.app.state.repo,
        request.app.state.llm,
        limit=limit,
        domain=request.app.state.config.domain_description,
    )
    return {"verified": n}

@router.post("/review/{review_id}/resolve")
def review_resolve(request: Request, review_id: str, decision: ReviewDecision):
    if request.app.state.repo is None:
        raise HTTPException(status_code=503, detail="database unavailable")
    result = apply_review_decision(
        request.app.state.repo,
        review_id,
        decision.model_dump(),
        config=request.app.state.config,
        run_reasoner=decision.run_reasoner,
    )
    if not result.ok:
        # 404 when the item is gone; 409 for a stale target or a gate rejection.
        code = 404 if "not found" in result.reason else 409
        raise HTTPException(status_code=code, detail=result.reason)
    return {
        "ok": True,
        "action": result.action,
        "version_id": result.version_id,
        "detail": result.detail,
    }

@router.post("/resolve")
def resolve(request: Request, req: ResolveRequest):
    repo = request.app.state.repo
    if repo is None:
        raise HTTPException(status_code=503, detail="database unavailable")

    # Exact surface -> IRI. A bare `term` is also tried as a surface, so the caller can send
    # one field and get exact-then-kNN reconciliation.
    exact = (
        repo.resolve_surface(req.surface or req.term, kind=req.kind)
        if (req.surface or req.term)
        else None
    )

    nearest: list[dict] = []
    query_vec = None
    if req.vector is not None:
        query_vec = np.asarray(req.vector, dtype=np.float32)  # needs no provider
        # k is bounded for the same reason (a bad client value must not reach Postgres raw).
        # Unvalidated, a wrong-dimension vector raises out of center_query's `vector - mean`
        # as a numpy broadcast error, and a zero vector yields a NaN score that Starlette
        # cannot serialize — both 500s for what is a client mistake.
        # 1024 matches the fixed vector column, as in generate.py's persist_vectors guard.
        if (
            query_vec.ndim != 1
            or query_vec.size != 1024
            or not np.all(np.isfinite(query_vec))
            or not np.any(query_vec)
        ):
            raise HTTPException(
                status_code=422,
                detail="vector must be 1024 finite floats and not all-zero",
            )
    elif req.term is not None:
        provider = request.app.state.provider
        if provider is None:
            raise HTTPException(
                status_code=503, detail="embedding provider unavailable"
            )
        query_vec = provider.embed([req.term])[0]

    if query_vec is not None:
        model_id = _active_model_id(request)
        if model_id is not None:
            # Center the query into the stored centered space before k-NN.
            query_vec = center_query(query_vec, request.app.state.model_mean)
            # /resolve stays symmetric with exact surface resolution (which is not
            # deprecation-filtered): a deprecate-in-place concept still carries surfaces and
            # must remain fuzzy-resolvable. The deprecated exclusion is an attach-only guard,
            # not a query-time one.
            nearest = [
                {"iri": iri, "score": score}
                for iri, score in repo.nearest_concepts(
                    query_vec,
                    model_id,
                    req.k,
                    kind=req.kind,
                    include_deprecated=True,
                )
            ]

    # Server owns the acceptance decision so callers never re-judge similarity.
    threshold = request.app.state.config.resolve_threshold
    top_score = nearest[0]["score"] if nearest else None
    if exact is not None:
        iri, match, confident = exact, "exact", True
    elif top_score is not None and top_score >= threshold:
        iri, match, confident = nearest[0]["iri"], "knn", True
    else:
        iri, match, confident = None, "none", False

    # Trace match/score so the acceptance threshold can be tuned against real traffic.
    with request.app.state.tracer.trace(
        "resolve", input={"surface": req.surface, "term": req.term}
    ) as tr:
        tr.update(
            metadata={
                "match": match,
                "confident": confident,
                "score": top_score,
                "threshold": threshold,
            }
        )

    return {
        "iri": iri,  # null when no confident match
        "match": match,  # "exact" | "knn" | "none"
        "confident": confident,
        "score": top_score,  # top k-NN cosine (null if none computed)
        "threshold": threshold,  # server-owned acceptance cutoff
        "nearest": nearest,  # full candidate list for inspection
    }


def init_state(app: FastAPI, config: Config | None = None) -> None:
    """Fills defaults for the app.state keys tboxcraft uses.

    Existing values are respected (test pre-seed). A safety net so the /tbox routes answer
    with a 503 degrade even under a TestClient running without the lifespan (same meaning
    as the old create_app state initialization).
    """
    st = app.state
    if getattr(st, "config", None) is None:
        st.config = config or Config.from_env()
    for name in ("repo", "provider", "llm", "model_id", "model_mean"):
        if not hasattr(st, name):
            setattr(st, name, None)
    if getattr(st, "tracer", None) is None:
        # No-op until the lifespan builds the real tracer (keeps module import free of
        # side effects).
        st.tracer = Tracer(None, "pre-lifespan")


def create_app(config: Config | None = None) -> FastAPI:
    """Standalone/test assembly: an independent FastAPI with the router at the root path.

    In the unified app, app.main includes this router with prefix="/tbox" and combines the
    lifespan.
    """
    configure_logging(env_var="TBOXCRAFT_LOG_LEVEL")
    app = FastAPI(title="tboxcraft", lifespan=lifespan)
    init_state(app, config)
    app.include_router(router)
    return app
