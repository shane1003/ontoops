"""Review-queue consumer: apply a human decision to a queued item.

The Update orchestrator enqueues gray/relabel items (and never auto-produces a delete);
this module closes the loop. A human picks merge / hide / new / relabel / delete / reject
('merge' keeps a word as an AlterTerm, 'hide' keeps it as a HiddenTerm for ABox, 'reject'
discards it), and the decision is applied through the same mutate → validate → commit machinery
as ingest:
every ontology-mutating action re-loads the latest snapshot, applies the change, re-runs the
syntax + HermiT gate, and commits the new snapshot + resolution rows + the resolved-queue flip
in one transaction. A target that has changed since enqueue is reported as a stale conflict and
left pending. Deletion is reachable only here, never from ingest.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

from psycopg import errors as pg_errors
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

from app.tboxcraft.config import Config
from app.tboxcraft.db.repository import StaleOntologyError
from app.tboxcraft.emit.tbox import serialize_turtle
from app.tboxcraft.models import Component, Hierarchy
from app.tboxcraft.pipeline.update_apply import _add_new_concepts, _surface_entries
from app.tboxcraft.represent import (
    mint,
    pref_label_of,
    pref_lang_for,
    representative_label,
)
from app.tboxcraft.validate import gate, validate_tbox

logger = logging.getLogger(__name__)

# OWL entity type per component kind, for a renamed IRI's deprecated tombstone.
_OWL_TYPE = {
    "class": OWL.Class,
    "object_property": OWL.ObjectProperty,
    "data_property": OWL.DatatypeProperty,
}


def _migrate_iri(graph: Graph, old: URIRef, new: URIRef) -> None:
    """Rewrite every triple where ``old`` is subject or object to ``new`` (reference
    migration): the concept's own axioms move, and every reference (domain/range, subClassOf)
    re-points, so nothing dangles after the rename."""
    for s, p, o in list(graph.triples((old, None, None))):
        graph.remove((s, p, o))
        graph.add((new, p, o))
    for s, p, o in list(graph.triples((None, None, old))):
        graph.remove((s, p, o))
        graph.add((s, p, new))


@dataclass
class ApplyResult:
    ok: bool
    action: str | None = None
    reason: str = ""
    version_id: str | None = None
    detail: dict | None = None  # e.g. delete outcome: {deprecated, references}


def list_pending(repo, limit: int = 100, *, base_iri: str | None = None) -> list[dict]:
    """Pending items, optionally only one namespace's. See Repository.list_pending_reviews."""
    return repo.list_pending_reviews(limit, base_iri=base_iri)


def verify_pending(
    repo, provider, *, limit: int = 100, domain: str | None = None
) -> int:
    """Attach the advisory domain-term verifier verdict to pending review items so a human
    adjudicates with it. Covers every risky entry type the 4-layer pipeline enqueues —
    ``relabel`` (UPDATE), ``delete`` (DELETE), and ``gray`` (low-confidence attach) — filling a
    verdict for items enqueued without one (e.g. the ingest verifier was disabled/unavailable).
    Runs at review time, not on the ingest path. The verifier never raises — an
    unavailable LLM records ``status="unavailable"`` and the human still decides. Returns the
    count verified."""
    from app.tboxcraft.verify_term import _DEFAULT_DOMAIN, verify_term

    domain = domain or _DEFAULT_DOMAIN
    verified = 0
    for item in repo.list_pending_reviews(limit):
        if item["entry_type"] not in ("relabel", "delete", "gray"):
            continue
        if (item.get("llm_verdict") or {}).get("status") == "ok":
            continue  # already carries a verdict (e.g. attached at ingest) — don't re-spend a call
        payload = item.get("payload") or {}
        if "name" not in payload or "kind" not in payload:
            continue  # cannot rebuild the candidate without at least name+kind
        component = _component_from_payload(payload)
        nearest = [
            {
                "iri": m.get("iri"),
                "score": m.get("score"),
                "label": repo.get_pref_label(m["iri"]) if m.get("iri") else None,
            }
            for m in (item.get("nearest") or [])
        ]
        verdict = verify_term(
            component, provider, gloss=item.get("gloss"), nearest=nearest, domain=domain
        )
        repo.set_review_verdict(item["id"], verdict.to_dict())
        verified += 1
    return verified


def _component_from_payload(payload: dict) -> Component:
    return Component(
        name=payload["name"],
        kind=payload["kind"],
        domain=payload.get("domain"),
        range_raw=payload.get("range"),
        correct_term=payload.get("correct_term"),
        correct_term_en=payload.get("correct_term_en"),
        dom_notes=payload.get("dom_notes"),
        lang=payload.get("lang"),
        notes_lang=payload.get("notes_lang"),
        dom_duplicates=tuple(payload.get("duplicates") or []),
        dom_synonyms=tuple(payload.get("synonyms") or []),
        dom_hierarchy=Hierarchy(broader=tuple(payload.get("broader") or [])),
        source_cq_ids=tuple(payload.get("source_cq_ids") or []),
    )


def _validate(graph: Graph, config: Config, run_reasoner: bool, run_oops: bool):
    report = validate_tbox(
        graph=graph,
        run_reasoner=run_reasoner,
        run_oops=run_oops,
        java_exe=config.java_exe,
        oops_endpoint=config.oops_endpoint,
    )
    return gate(report)


def _config_for_item(config: Config, item: dict, review_id: str) -> Config:
    """``config`` re-pointed at the namespace the queued item belongs to.

    Falls back to the instance default only when the row carries no namespace, which is the
    pre-`base_iri` shape section 3d of the schema backfills what it can of: an item with no
    ``term_id`` (GRAY / KIND_CONFLICT — exactly the ones most in need of review) cannot have its
    namespace derived from an IRI, so it stays NULL when the store holds more than one. Those
    resolve where they always did, and the log says so rather than leaving it to be inferred.
    """
    queued = item.get("base_iri")
    if not queued:
        logger.warning(
            "review: item %s carries no base_iri (queued before the column, or backfilled as "
            "ambiguous) — applying under this instance's default %s",
            review_id,
            config.base_iri,
        )
        return config
    if queued != config.base_iri:
        logger.info(
            "review: item %s belongs to %s, not this instance's default %s — applying in the "
            "namespace it was queued under",
            review_id,
            queued,
            config.base_iri,
        )
    return dataclasses.replace(config, base_iri=queued)


def apply_review_decision(
    repo,
    review_id: str,
    decision: dict,
    *,
    config: Config,
    run_reasoner: bool = True,
    run_oops: bool = False,
) -> ApplyResult:
    """Apply one human decision. ``decision`` = {action, target_iri?, label?}.

    The namespace comes from the **queued item**, not from this instance's default. Applying a
    decision mints IRIs, loads that namespace's snapshot chain and writes a new one, so acting in
    the wrong graph is not a wrong read — it puts a term under a base IRI nobody asked for, and an
    IRI is the one thing here a later review cannot take back. `/review/pending` already scopes by
    namespace while this path did not, so a reviewer listing one document's queue and resolving
    from it could land the result in another.
    """
    item = repo.get_review(review_id)
    if item is None:
        return ApplyResult(False, reason="review item not found")
    if item["status"] != "pending":
        return ApplyResult(False, reason=f"already {item['status']}")
    config = _config_for_item(config, item, review_id)

    action = decision.get("action")
    payload = item["payload"] or {}
    target_iri = decision.get("target_iri") or payload.get("target_iri")
    logger.info(
        "review: applying id=%s action=%s target=%s", review_id, action, target_iri
    )

    if action == "reject":
        repo.mark_review_resolved(review_id, "rejected", decision)
        return ApplyResult(True, action="reject")

    if action in ("merge", "hide"):
        # The domain-expert keep/drop decision on a clustered word, both surface-only on the
        # target concept (no IRI change, no re-gate): 'merge' keeps it as an AlterTerm (clean
        # synonym / Use For); 'hide' drops it from the standard synonym set but retains it as a
        # HiddenTerm — a messy document form kept only so ABox mapping can still resolve it.
        # Neither ever discards the word; 'reject' above is the only true discard.
        term_type = "alter" if action == "merge" else "hidden"
        if not target_iri:
            return ApplyResult(
                False, action=action, reason=f"{action} needs a target_iri"
            )
        cid = repo.get_concept_id(target_iri)
        if cid is None:  # stale: target changed/removed since enqueue
            return ApplyResult(
                False, action=action, reason="stale target: concept no longer exists"
            )
        component = _component_from_payload(payload)
        with repo.transaction() as conn:
            # A human decision is authoritative: move an existing surface between roles rather
            # than no-op'ing (update_role), so the endpoint's success reflects the DB state.
            repo.insert_surface_forms(
                cid,
                _surface_entries(component),
                term_type=term_type,
                update_role=True,
                conn=conn,
            )
            repo.mark_review_resolved(review_id, "applied", decision, conn=conn)
        logger.info(
            "review: applied '%s' -> %s (%sTerm)",
            action,
            target_iri,
            "Alter" if term_type == "alter" else "Hidden",
        )
        return ApplyResult(True, action=action)

    # --- ontology-mutating actions load the latest snapshot and re-gate ---
    latest = repo.get_latest_ontology(config.base_iri)
    if latest is None:
        return ApplyResult(False, action=action, reason="no accumulated ontology")
    base_id, ttl = latest
    graph = Graph()
    graph.parse(data=ttl, format="turtle")

    try:
        if action == "new":
            component = _component_from_payload(payload)
            iri = mint(config.base_iri, component.kind, component.name)
            # Guard both the snapshot graph and the concept table: an existing row would be
            # clobbered by upsert_concept's ON CONFLICT, silently overwriting a stable IRI.
            if (URIRef(iri), RDF.type, None) in graph or repo.get_concept_id(
                iri
            ) is not None:
                return ApplyResult(False, action="new", reason="stale: IRI now exists")
            # A term cannot be created without its model + vector (the column is NOT NULL), so
            # check before running the reasoner rather than spending a HermiT pass to then abort.
            model_id = repo.get_model_id(config.e5_model, config.embedding_revision)
            vec = repo.get_review_vector(review_id)
            if model_id is None or vec is None:
                return ApplyResult(
                    False,
                    action="new",
                    reason="cannot create term: embedding model/vector unavailable",
                )
            _add_new_concepts(graph, [component], config.base_iri)
            passed, reasons = _validate(graph, config, run_reasoner, run_oops)
            if not passed:
                return ApplyResult(False, action="new", reason="; ".join(reasons))
            with repo.transaction() as conn:
                vid = repo.save_ontology_version(
                    serialize_turtle(graph),
                    base_iri=config.base_iri,
                    note="review:new",
                    base_id=base_id,
                    conn=conn,
                )
                cid = repo.upsert_concept(
                    iri,
                    component.kind,
                    dict([pref_label_of(component)]),
                    component.name,
                    embedding_model_id=model_id,
                    embedding_val=vec,
                    snapshot_id=vid,
                    conn=conn,
                )
                repo.insert_surface_forms(cid, _surface_entries(component), conn=conn)
                repo.mark_review_resolved(review_id, "applied", decision, conn=conn)
            logger.info("review: applied 'new' %s -> snapshot v%s", iri, vid)
            return ApplyResult(True, action="new", version_id=vid)

        if action == "relabel":
            # IRI-renewal Major Update: re-mint the IRI from the new official
            # term, migrate every reference old->new, deprecate the old IRI, and rename the
            # concept in the DB. Human-approved (this consumer) + LLM-advised.
            if not target_iri:
                return ApplyResult(
                    False, action="relabel", reason="relabel needs a target_iri"
                )
            t = URIRef(target_iri)
            cid = repo.get_concept_id(target_iri)
            if (t, RDF.type, None) not in graph or cid is None:
                return ApplyResult(
                    False,
                    action="relabel",
                    reason="stale target: concept no longer exists",
                )
            component = _component_from_payload(payload)
            new_label = (
                decision.get("label")
                or payload.get("correct_term")
                or representative_label(component)
            )
            # The reviewer may say what language they typed; otherwise the label inherits the
            # component's, which is `und` until a producer declares one. Not derived from the
            # string — a human typing a Korean label into a form is exactly the case that must be
            # asked rather than guessed, and the field is there to be filled in.
            new_lang = pref_lang_for(new_label)
            new_iri = mint(config.base_iri, component.kind, new_label)
            n = URIRef(new_iri)
            renamed = new_iri != target_iri
            if renamed and (
                (n, RDF.type, None) in graph or repo.get_concept_id(new_iri) is not None
            ):
                return ApplyResult(
                    False, action="relabel", reason="renewal target IRI already exists"
                )
            # Read before the migration moves them onto the new IRI. All of them, not the first:
            # a term may hold a label per language and only the one being relabelled is displaced.
            old_labels = [
                o for o in graph.objects(t, SKOS.prefLabel) if isinstance(o, Literal)
            ]
            if renamed:
                _migrate_iri(graph, t, n)  # move axioms + re-point references
                # Deprecated tombstone on the old IRI: an inert, auditable redirect.
                # NOT owl:sameAs — a reasoner would merge old/new and defeat the rename.
                graph.add((t, RDF.type, _OWL_TYPE[component.kind]))
                graph.add((t, OWL.deprecated, Literal(True)))
                graph.add((t, RDFS.seeAlso, n))
                # `lang="en"` is a fact here, not an assumption: this sentence is composed on the
                # line above it. The tags that had to go were the ones asserted over strings that
                # came from somewhere else.
                graph.add(
                    (t, SKOS.changeNote, Literal(f"renamed to {new_iri}", lang="en"))
                )
            # Promote the new official term as the representative label on the (possibly new) IRI,
            # replacing only the label in the language being relabelled. Clearing every language
            # would silently drop a term's other names because a reviewer edited one of them.
            for pred in (SKOS.prefLabel, RDFS.label):
                for o in list(graph.objects(n, pred)):
                    if getattr(o, "language", None) == new_lang:
                        graph.remove((n, pred, o))
                graph.add((n, pred, Literal(new_label, lang=new_lang)))
            passed, reasons = _validate(graph, config, run_reasoner, run_oops)
            if not passed:
                return ApplyResult(False, action="relabel", reason="; ".join(reasons))
            record = {**decision, "old_iri": target_iri, "new_iri": new_iri}
            with repo.transaction() as conn:
                vid = repo.save_ontology_version(
                    serialize_turtle(graph),
                    base_iri=config.base_iri,
                    note="review:relabel",
                    base_id=base_id,
                    conn=conn,
                )
                if renamed:
                    repo.rename_concept_iri(
                        cid, new_iri, conn=conn
                    )  # vectors/CQ survive
                # After a renewal the concept lives at the new IRI and the old one is a tombstone,
                # so labels and demoted surfaces must land on the new IRI — writing them to `cid`
                # would decorate the tombstone and leave the live term with the stale label.
                live_iri = new_iri if renamed else cid
                repo.set_pref_labels(live_iri, {new_lang: new_label}, conn=conn)
                # Only the displaced label becomes an alias, and it keeps its own tag. A label in
                # another language was not replaced and is still this term's PREF there — demoting
                # it would strip a name the reviewer never touched.
                demoted: list[tuple[str, str, str]] = [
                    (str(o), new_lang, "relabel-demoted")
                    for o in old_labels
                    if getattr(o, "language", None) == new_lang
                ]
                demoted += _surface_entries(component)
                repo.insert_surface_forms(live_iri, demoted, conn=conn)
                repo.mark_review_resolved(review_id, "applied", record, conn=conn)
            logger.info(
                "review: applied renewal %s -> %s (%r, snapshot v%s)",
                target_iri,
                new_iri,
                new_label,
                vid,
            )
            return ApplyResult(True, action="relabel", version_id=vid)

        if action == "delete":
            # Human-only removal, reachable only through this consumer. Deprecate-in-place:
            # if the concept is still referenced, mark it owl:deprecated and KEEP its
            # triples so nothing dangles; hard-remove only when unreferenced.
            if not target_iri:
                return ApplyResult(
                    False, action="delete", reason="delete needs a target_iri"
                )
            t = URIRef(target_iri)
            if (t, RDF.type, None) not in graph:
                return ApplyResult(
                    False,
                    action="delete",
                    reason="stale target: concept no longer exists",
                )
            cid = repo.get_concept_id(target_iri)
            # References = other subjects pointing at this IRI (domain/range, child subClassOf).
            references = sorted(
                {str(p) for s, p, o in graph.triples((None, None, t)) if s != t}
            )
            # ...plus the ones the graph cannot show. An individual attached to this class leaves
            # no triple in the TTL, so a graph-only check called such a class unreferenced and sent
            # it down the hard-delete path, where the RESTRICT FK raised — the reviewer was told
            # "no references" and then handed a database error. This does not make the delete
            # lossless: deprecating retires the anchor, and the loader resolves ACTIVE only, so
            # individuals naming this class stop resolving on the next load either way. What it
            # changes is that the reviewer sees which of the two they chose.
            n_individuals = repo.count_external_dependents(target_iri)
            deprecated = bool(references) or n_individuals > 0
            if deprecated:
                graph.add((t, OWL.deprecated, Literal(True)))
                note_text = (
                    "deprecated in place (still referenced)"
                    if references
                    else f"deprecated in place ({n_individuals} individual(s) attached)"
                )
                graph.add((t, SKOS.changeNote, Literal(note_text, lang="en")))
            else:
                for triple in list(graph.triples((t, None, None))):
                    graph.remove(triple)
                for triple in list(graph.triples((None, None, t))):
                    graph.remove(triple)
            passed, reasons = _validate(graph, config, run_reasoner, run_oops)
            if not passed:
                return ApplyResult(False, action="delete", reason="; ".join(reasons))
            note = "review:deprecate" if deprecated else "review:delete"
            with repo.transaction() as conn:
                vid = repo.save_ontology_version(
                    serialize_turtle(graph),
                    base_iri=config.base_iri,
                    note=note,
                    base_id=base_id,
                    conn=conn,
                )
                if deprecated:
                    if cid is not None:
                        repo.deprecate_concept(cid, conn=conn)
                else:
                    repo.delete_concept(target_iri, conn=conn)
                repo.mark_review_resolved(
                    review_id,
                    "applied",
                    {**decision, "deprecated": deprecated},
                    conn=conn,
                )
            logger.warning(
                "review: applied 'delete' %s -> %s snapshot v%s (human-gated)",
                target_iri,
                "deprecated" if deprecated else "removed",
                vid,
            )
            return ApplyResult(
                True,
                action="delete",
                version_id=vid,
                # n_individuals separately: 'references' holds predicate IRIs, so a
                # delete that turned into a deprecation purely because of attached
                # individuals would otherwise report an empty reference list.
                detail={
                    "deprecated": deprecated,
                    "references": references,
                    "n_individuals": n_individuals,
                },
            )
    except StaleOntologyError as exc:
        return ApplyResult(
            False, action=action, reason=f"concurrent update conflict: {exc}"
        )
    except pg_errors.UniqueViolation:
        # A concurrent change inserted the target IRI between the pre-check and the commit
        # (rename has no ON CONFLICT). Report a clean conflict, not an uncaught 500.
        return ApplyResult(
            False, action=action, reason="target IRI already exists (concurrent change)"
        )
    except pg_errors.ForeignKeyViolation:
        # count_external_dependents runs before _validate (a full HermiT pass), so an ABox load can
        # attach an individual inside that window and the RESTRICT FK then rejects the delete. Same
        # shape as the UniqueViolation above: a clean 409, not an uncaught 500 that loses the
        # reviewer's decision with no explanation.
        return ApplyResult(
            False,
            action=action,
            reason="concept acquired a dependent reference since the check "
            "(concurrent change)",
        )

    return ApplyResult(False, action=action, reason=f"unknown action: {action!r}")
