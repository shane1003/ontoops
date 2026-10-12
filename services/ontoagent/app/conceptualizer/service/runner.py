"""Stage-1 runner: chunk fan-out/fan-in, reintegration, artifact writing.

Flow:
 1. Load the CQ file as a CQDocument.
 2. Split the CQList into chunks of settings.extraction.chunk_size.
 3. For each chunk (concurrently): extract components + individuals, and run
    DomainSpecificClassifier on that chunk's components to attach the domain layer +
    TBox/ABox verdict.
    (Individuals are extracted after the chunk's verdicts are settled, so each one's
    class_name equals a component name.)
 4. Cross-chunk reintegration: merge components by (kind, name) (domain layer union)
    and concatenate individuals. When the per-chunk TBox/ABox verdicts for a
    same-named component conflict, the merged verdict becomes `conflict` and a
    `DisagreementFlag(kind="verdict_conflict")` holding the competing verdicts and
    source chunk ids is attached.
 5. Write two artifacts to disk: annotated components (TBox) and individuals (ABox).
    Individuals never go into the components artifact, nor components into the
    individuals artifact.

Chunks have no data dependencies on each other (reintegration runs after all of them
finish) and each is a single LLM call, so run sequentially the wall-clock time grows
in direct proportion to the chunk count. So they are submitted concurrently to a
thread pool (`CONCEPTUALIZER_CHUNK_CONCURRENCY`, default 8) and **put back in their
original index order** for reintegration: collected in completion order, the sources
in `disagreement_flags` (`chunk3` etc.) would vary per run and the same input would
not give the same output.

The verdict (tbox/abox/conflict) is a 'mark' on the component, not an output gate:
whatever the verdict, the component goes out unchanged. Individuals reference
components by name, so filtering here would leave those individuals with
nowhere to attach.

No TTL is generated. No semantic merging ("different names, same concept") either:
choosing representatives and dedup belong to the downstream that owns the global TBox
(tboxcraft). Merging here would keep discarded names from ever reaching downstream,
so they could not remain in surface_form, and individuals carrying those names would
fail to resolve.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from pathlib import Path

from app.core.config import Settings
from app.core.llm.tracing import traced
from app.conceptualizer.schemas import (
    AnnotatedComponent,
    CQDocument,
    Individual,
    ExtractionOutput,
)
from app.conceptualizer.service.domain_specific_classifier import (
    ClassifierStats,
    DomainSpecificClassifier,
)
from app.conceptualizer.service.extract import (
    extract_components,
    extract_individuals,
    load_cq_document,
)
from app.conceptualizer.service.merge_util import (
    flag_hierarchy_contradictions,
    reconcile_verdict,
    union_component_fields,
)

log = logging.getLogger("conceptualizer.runner")


# =========================================================================
# Chunk splitting
# =========================================================================


def _chunk(items: list, size: int) -> list[list]:
    """Splits items into chunks of size (a single chunk of everything if size<=0)."""
    if size <= 0:
        return [list(items)] if items else []
    return [list(b) for b in itertools.batched(items, size)]


def _chunk_id(index: int) -> str:
    """Returns the chunk source id (used as the source of disagreement flag candidates)."""
    return f"chunk{index}"


# =========================================================================
# Chunk fan-out (concurrent execution)
# =========================================================================

# Concurrency cap. Unlimited, a large document floods the LLM server queue and times
# out.
_CONCURRENCY_ENV = "CONCEPTUALIZER_CHUNK_CONCURRENCY"
_DEFAULT_CONCURRENCY = 8


def _chunk_concurrency(n_chunks: int) -> int:
    """Decides the chunk concurrency: env > default 8, never more than the chunk count.

    0, negative or unparsable values fold to **1 (serial)** or the default, not to
    'unlimited': a reading that removes the cap directly defeats the reason for this
    setting (preventing queue floods). Give 1 when serial execution (= the same path
    as in the sequential days) is needed.
    """
    raw = (os.getenv(_CONCURRENCY_ENV) or "").strip()
    limit = _DEFAULT_CONCURRENCY
    if raw:
        try:
            limit = int(raw)
        except ValueError:
            log.warning(
                "[extract] could not read %s=%r as an integer → using default %d",
                _CONCURRENCY_ENV,
                raw,
                _DEFAULT_CONCURRENCY,
            )
    return max(1, min(limit, max(n_chunks, 1)))


def _process_chunk(
    chunk: list,
    index: int,
    n_chunks: int,
    classifier: DomainSpecificClassifier,
    stats: ClassifierStats,
) -> tuple[list[AnnotatedComponent], list[Individual]]:
    """Processes one chunk: component extraction → classifier read → individual extraction.

    Concurrent execution is safe only if chunks share no state. The classifier is
    shared by several chunks but holds no state between calls (the batch label is
    passed as an argument too), and each log line contains `chunk i/n`, so it is
    readable which chunk a line belongs to even when completion order interleaves.
    `stats` is the observation accumulator the chunks fill jointly (self-locking).
    """
    label = f"chunk {index + 1}/{n_chunks}"

    # 1) Extract this chunk's component structure.
    comps = extract_components(chunk)
    log.info("[extract] %s: %d CQs → %d components extracted", label, len(chunk), len(comps))

    # 2) Classifier read: domain layer + verdict (logged with label).
    comps = classifier.run(comps, label=label, stats=stats)

    # 3) Individual extraction (class_name must equal the component name).
    individuals = extract_individuals(chunk)

    verdicts = Counter(c.verdict for c in comps)
    log.info(
        "[extract] %s: done — verdict(tbox=%d abox=%d conflict=%d), %d individuals",
        label,
        verdicts.get("tbox", 0),
        verdicts.get("abox", 0),
        verdicts.get("conflict", 0),
        len(individuals),
    )
    return comps, individuals


def _fan_out_chunks(
    chunks: list[list],
    classifier: DomainSpecificClassifier,
    stats: ClassifierStats,
    workers: int,
) -> tuple[list[list[AnnotatedComponent]], list[list[Individual]]]:
    """Processes chunks concurrently in a thread pool and puts results back in order.

    Results go back **at their original index**. Collected in completion order, the
    sources (chunk ids) in disagreement_flags would vary per run and the same input
    would not give the same output (order stability → reproducible chunk ids).
    """
    n_chunks = len(chunks)
    per_chunk_components: list[list[AnnotatedComponent]] = [[] for _ in chunks]
    per_chunk_individuals: list[list[Individual]] = [[] for _ in chunks]

    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="extract-chunk"
    ) as pool:
        futures = {
            # The Langfuse trace is a ContextVar and is not inherited by new threads.
            # The context must be copied and passed so the chunk's LLM calls are
            # grouped under this extract trace.
            # (copy_context() is created anew per submit: two threads running the
            #  same Context object at once raise RuntimeError.)
            pool.submit(
                copy_context().run,
                _process_chunk,
                chunk,
                i,
                n_chunks,
                classifier,
                stats,
            ): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                comps, individuals = future.result()
            except Exception as exc:  # noqa: BLE001 — one chunk failing does not kill the whole run
                # In the sequential days an exception killed the whole run, but in the
                # fan-out the other chunks have already paid their LLM cost. Empty only
                # the failed chunk and continue (same intent as the classifier's
                # internal degrade contract). ERROR + stack so it does not pass
                # silently.
                log.error(
                    "[extract] chunk %d/%d failed → emptying this chunk and continuing: %s",
                    i + 1,
                    n_chunks,
                    exc,
                    exc_info=True,
                )
                continue
            per_chunk_components[i] = comps
            per_chunk_individuals[i] = individuals

    return per_chunk_components, per_chunk_individuals


# =========================================================================
# Reintegration (merge by name + verdict conflict rules)
# =========================================================================


def _reintegrate_components(
    per_chunk: list[list[AnnotatedComponent]],
) -> list[AnnotatedComponent]:
    """Reintegrates the per-chunk components into one list by (kind, name).

    - Keeps the order of first appearance.
    - Structure/domain layer/provenance are unioned with union_component_fields.
    - The verdict follows the reconcile_verdict rules (conflict on a clash of decisive
      values or when conflict is observed).
    - The hierarchy is unioned, but when the same name is observed as both broader
      and narrower a hierarchy_conflict flag is left (both edges are kept).

    Observations can disagree even in fields that are only unioned. The verdict has a
    single slot, so a clash shows; the hierarchy is a list, so opposite-direction
    observations silently sit side by side, and they really meet as the number of
    chunks a name spans grows (a common name can appear in hundreds of CQs). Passing on
    the contradiction without evidence leaves downstream unable to decide which edge
    to drop.

    NOTE: only surface (exact (kind,name)) merging is done: same names only, so no
    name is discarded. Semantic duplicate merging is not done here. That is the judgment of the downstream that sees the global TBox;
    done here, the discarded names would never reach downstream and individuals
    referencing them would fail to resolve.
    """
    order: list[tuple[str, str]] = []
    merged: dict[tuple[str, str], AnnotatedComponent] = {}
    # (kind, name) -> {verdict: [chunk_id, ...]}  : per-chunk verdict observations.
    verdicts: dict[tuple[str, str], dict[str, list[str]]] = {}
    # (kind, name) -> {other name: {view: [chunk_id, ...]}}  : per-chunk hierarchy
    # observations. Merged away, which chunk saw which direction would be lost and no
    # evidence for a contradiction could be built.
    hierarchy: dict[tuple[str, str], dict[str, dict[str, list[str]]]] = {}

    for chunk_idx, comps in enumerate(per_chunk):
        cid = _chunk_id(chunk_idx)
        for c in comps:
            key = (c.kind, c.name)
            verdicts.setdefault(key, {}).setdefault(c.verdict, []).append(cid)
            obs = hierarchy.setdefault(key, {})
            for view, names in c.domain_layer.hierarchy.items():
                for n in names:
                    sources = obs.setdefault(n, {}).setdefault(view, [])
                    if cid not in sources:
                        sources.append(cid)
            base = merged.get(key)
            if base is None:
                # First appearance: store a deep copy so later merges do not alter the original.
                merged[key] = c.model_copy(deep=True)
                order.append(key)
                continue
            union_component_fields(base, c)

    # Verdict reconciliation + hierarchy contradiction flags (both with chunk ids as
    # source).
    contradictions: list[str] = []
    for key in order:
        reconcile_verdict(
            merged[key], verdicts[key], note="청크 간 TBox/ABox 판정이 갈림."
        )
        for name in flag_hierarchy_contradictions(merged[key], hierarchy.get(key, {})):
            contradictions.append(f"{merged[key].name}↔{name}")

    if contradictions:
        # WARNING so this does not become a silent defect: it leaves a trace on the API
        # (uvicorn) path too.
        log.warning(
            "[extract] %d hierarchy contradictions — same name observed as both "
            "broader/narrower: %s",
            len(contradictions),
            ", ".join(contradictions[:5]),
        )

    return [merged[k] for k in order]


# =========================================================================
# Output contract check (individual anchors)
# =========================================================================


def _orphan_individuals(
    components: list[AnnotatedComponent], individuals: list[Individual]
) -> list[Individual]:
    """Selects the individuals whose class_name matches no class component (no anchor).

    An individual names its component through class_name. If that name is not in
    the component list, the output contradicts itself and downstream, unable to find
    the anchor, experiences it as a miss.

    The typical cause is the input: when a property's domain/range points to a name
    not declared in that CQ's `classes`, component extraction does not make that name
    a class (it only sees the declared classes) while individual extraction uses the
    domain/range as the anchor (a range name absent from classes). The individuals are not removed here: the values are real
    ABox facts, and discarding them would leave those facts nowhere. Instead the
    contract violation is exposed.
    """
    class_names = {c.name for c in components if c.kind == "class"}
    return [i for i in individuals if i.class_name not in class_names]


# =========================================================================
# Runner core
# =========================================================================


@traced("extract")
def build_extraction(
    doc: CQDocument,
    settings: Settings,
    *,
    chunk_size: int | None = None,
    language: str | None = None,
) -> ExtractionOutput:
    """Builds an ExtractionOutput (reintegrated components + individuals) from a CQDocument.

    Uses no disk. Performs chunk fan-out → classifier read → reintegration and RETURNS
    the ExtractionOutput. No artifacts are written (file-less in-memory core, for the
    API runner); run_extraction wraps file loading/writing.

    Args:
        doc: Parsed CQ input document.
        settings: Source of the classifier LLM and chunk_size.
        chunk_size: If given, overrides settings.extraction.chunk_size (for the API
            query parameter).
        language: **Fallback** language of the domain-layer text (correct_term,
            notes). Used only for values the classifier gave no language for;
            without it, `und`. The language is not guessed from character ranges
            (that breaks on mixed scripts).

    Returns:
        The reintegrated ExtractionOutput.
    """
    cqs = doc.competency_questions

    size = settings.extraction.chunk_size if chunk_size is None else chunk_size
    chunks = _chunk(cqs, size)
    n_chunks = len(chunks)

    # The document's ontology.domain comes first; otherwise the deployment fallback
    # (CONCEPTUALIZER_DOMAIN(env) > settings.agents.domain). **It never falls back to
    # prefix**: prefix is the document name, and the moment CQ Gen started filling it
    # from the title, "you are a {document title} domain expert classifier" went into
    # the prompt (a silent defect that only lowers judging quality, with no exception
    # or warning). A deployment default someone chose beats a document name.
    domain_used = doc.ontology.domain or settings.agents.domain
    if doc.ontology.domain:
        domain_source = "document"
    else:
        domain_source = "env" if os.getenv("CONCEPTUALIZER_DOMAIN") else "settings"
    classifier = DomainSpecificClassifier(
        settings, domain=doc.ontology.domain, language=language
    )
    workers = _chunk_concurrency(n_chunks)
    # Observations the chunks fill jointly: read from the response instead of the logs
    # interleaved by concurrent execution.
    stats = ClassifierStats()

    log.info(
        "[extract] %d input CQs → %d chunks (chunk_size=%d, concurrency %d, domain=%s[%s])",
        len(cqs),
        n_chunks,
        size,
        workers,
        domain_used,
        domain_source,
    )

    per_chunk_components, per_chunk_individuals = _fan_out_chunks(
        chunks, classifier, stats, workers
    )

    # Individuals are also concatenated in chunk index order (deterministic regardless
    # of completion order).
    all_individuals: list[Individual] = [
        ind for chunk_inds in per_chunk_individuals for ind in chunk_inds
    ]

    # 4) Cross-chunk reintegration (exact (kind,name) merge + verdict conflict rules;
    #    individuals are concatenated).
    components = _reintegrate_components(per_chunk_components)

    # 5) The verdict is a 'mark', not a gate.
    #    Components judged abox are passed on unchanged too. Individuals reference
    #    component 'names' through class_name and property keys, so dropping one
    #    here leaves the individuals referencing that name with nowhere to attach and
    #    becomes a downstream resolution failure (miss). Whether to mint, keep only as
    #    surface_form, or send to the review queue is decided by the downstream that
    #    owns the global TBox after reading the verdict.
    # Output contract: every individual's class_name must exist as a class component.
    orphans = _orphan_individuals(components, all_individuals)

    output = ExtractionOutput(
        components=components,
        individuals=all_individuals,
        n_orphan_individuals=len(orphans),
        n_synthesized_components=sum(1 for c in components if c.synthesized),
        # Observations are carried as per-chunk sums (before reintegration): chunks
        # run concurrently and their logs interleave, so they can be read from the
        # response instead of by counting log lines.
        n_unmatched=stats.n_unmatched,
        n_promoted_conflict=stats.n_promoted_conflict,
        n_degraded_batches=stats.n_degraded_batches,
        # The domain this batch was actually judged with and its origin, so that a
        # fallback because the document did not carry one is visible outside the
        # response (document | env | settings).
        domain=domain_used,
        domain_source=domain_source,
    )
    # The verdict distribution goes only to the log (not into the response contract):
    # it is a diagnostic for whether the classifier leans one way, and the disposition
    # is up to the downstream reading the verdict.
    counts = Counter(c.verdict for c in components)
    log.info(
        "extract build done: components=%d (tbox=%d abox=%d conflict=%d, synthesized=%d) "
        "individuals=%d unmatched=%d conflict_promoted=%d degraded_chunks=%d/%d",
        len(components),
        counts.get("tbox", 0),
        counts.get("abox", 0),
        counts.get("conflict", 0),
        output.n_synthesized_components,
        len(all_individuals),
        stats.n_unmatched,
        stats.n_promoted_conflict,
        stats.n_degraded_batches,
        n_chunks,
    )
    if orphans:
        # A contract violation passing silently surfaces only later as a downstream
        # miss → WARNING. Record which CQ caused it (a missing classes declaration in
        # the input CQ is the typical cause).
        log.warning(
            "[extract] %d individuals without an anchor — class_name is not a class "
            "component (check whether a property's domain/range is undeclared in that "
            "CQ's classes): %s",
            len(orphans),
            ", ".join(f"{o.class_name}(cq={o.source_cq_id})" for o in orphans[:5]),
        )
    # Input present but empty output is a silent failure that "succeeds" with an empty
    # result and no exception. The completion log (INFO) alone is invisible on the API
    # (uvicorn) path because root is WARNING, so only this case is raised to WARNING to
    # leave a trace in the container log.
    if cqs and not all_individuals:
        log.warning(
            "[extract] %d input CQs but 0 individuals — no items hold a value, or "
            "extraction failed.",
            len(cqs),
        )
    if cqs and not components:
        log.warning(
            "[extract] %d input CQs but 0 components — check the CQ schema/extraction.", len(cqs)
        )
    return output


def run_extraction(
    cq_path: str | Path,
    settings: Settings,
    *,
    components_out: str | Path | None = None,
    individuals_out: str | Path | None = None,
    language: str | None = None,
) -> ExtractionOutput:
    """Builds an ExtractionOutput (reintegrated components + individuals) from a CQList.

    Also writes the artifacts, in the order load_cq_document → build_extraction →
    artifact writing. Writes to components_out/individuals_out when given, otherwise
    to the default paths under settings.paths.intermediate_dir. `language` means the
    same as in build_extraction (fallback language of the domain-layer text).
    """
    doc: CQDocument = load_cq_document(cq_path)
    output = build_extraction(doc, settings, language=language)

    # Write the two artifacts to disk (separate routing of individuals/components).
    comp_path, ind_path = _resolve_artifact_paths(
        settings, components_out, individuals_out
    )
    _write_artifacts(output, comp_path, ind_path)
    log.info(
        "extract done: components=%d individuals=%d",
        len(output.components),
        len(output.individuals),
    )
    return output


def _resolve_artifact_paths(
    settings: Settings,
    components_out: str | Path | None,
    individuals_out: str | Path | None,
) -> tuple[Path, Path]:
    """Decides the component/individual artifact output paths (arguments first, else settings)."""
    if components_out is not None:
        comp_path = Path(components_out)
    else:
        comp_path = Path(settings.paths.intermediate_dir) / "components.json"

    if individuals_out is not None:
        ind_path = Path(individuals_out)
    else:
        # Put individuals.json in the same directory as the components artifact.
        ind_path = comp_path.parent / "individuals.json"

    return comp_path, ind_path


def _write_artifacts(
    output: ExtractionOutput,
    components_path: Path,
    individuals_path: Path,
) -> None:
    """Writes the two artifacts; no individuals in the components file and vice versa."""
    components_path.parent.mkdir(parents=True, exist_ok=True)
    individuals_path.parent.mkdir(parents=True, exist_ok=True)

    # Components artifact (TBox): components only.
    components_payload = {
        "components": [c.model_dump(mode="json") for c in output.components],
    }
    components_path.write_text(
        json.dumps(components_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Individuals artifact (ABox): individuals only.
    individuals_payload = {
        "individuals": [i.model_dump(mode="json") for i in output.individuals],
    }
    individuals_path.write_text(
        json.dumps(individuals_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
