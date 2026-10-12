"""Standalone validation module + report contract.

Runnable on any TBox (a TTL path or an in-memory rdflib graph). Syntax and HermiT failures
block a pipeline return (see :func:`gate`); OOPS! pitfalls are advisory. Run standalone, the
module only reports.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rdflib import Graph

from app.tboxcraft.emit.tbox import serialize_rdfxml
from app.tboxcraft.validate.oops import SEVERITIES, scan_oops
from app.tboxcraft.validate.reasoner import run_hermit_on_rdfxml
from app.tboxcraft.validate.syntax import dl_profile_check, parse_ttl

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    syntax_ok: bool = False
    syntax_error: str | None = None
    dl_profile_ok: bool = True
    dl_violations: list[str] = field(default_factory=list)
    consistent: bool | None = None
    unsatisfiable_classes: list[str] = field(default_factory=list)
    reasoner_error: str | None = None
    # OOPS! (populated by the OOPS! scan). ``oops_error`` says why a scan produced nothing —
    # unreachable, or never attempted because no endpoint is configured. Without it "unavailable" is
    # a bare false and a reader cannot tell a network failure from a deliberate skip.
    oops_available: bool = False
    oops_pitfalls: list[dict] = field(default_factory=list)
    oops_error: str | None = None


def _load_graph(
    ttl_path: str | Path | None, graph: Graph | None
) -> tuple[Graph | None, str | None]:
    if graph is not None:
        return graph, None
    result = parse_ttl(ttl_path)
    return result.graph, result.error


def validate_tbox(
    ttl_path: str | Path | None = None,
    *,
    graph: Graph | None = None,
    run_reasoner: bool = True,
    run_oops: bool = False,
    java_exe: str | None = None,
    oops_endpoint: str | None = None,
    oops_session=None,
) -> ValidationReport:
    """Validate a TBox. Provide either ``ttl_path`` or an in-memory ``graph``."""
    report = ValidationReport()

    loaded, error = _load_graph(ttl_path, graph)
    if loaded is None:
        report.syntax_ok = False
        report.syntax_error = error
        logger.warning("validate: syntax FAILED: %s", error)
        return report
    report.syntax_ok = True
    logger.info("validate: syntax OK (%d triples)", len(loaded))

    dl = dl_profile_check(loaded)
    report.dl_profile_ok = dl.ok
    report.dl_violations = dl.violations
    if dl.ok:
        logger.info("validate: OWL 2 DL profile OK")
    else:
        logger.warning("validate: OWL 2 DL punning: %s", ", ".join(dl.violations))

    rdfxml = serialize_rdfxml(loaded)

    # Do not reason on an OWL-Full (punning) ontology — behavior is undefined.
    if run_reasoner and report.dl_profile_ok:
        logger.info("validate: running HermiT reasoner...")
        reasoner = run_hermit_on_rdfxml(rdfxml, java_exe=java_exe)
        report.consistent = reasoner.consistent
        report.unsatisfiable_classes = reasoner.unsatisfiable_classes
        report.reasoner_error = reasoner.error
        if reasoner.error:
            logger.warning("validate: HermiT error: %s", reasoner.error)
        else:
            logger.info(
                "validate: HermiT consistent=%s, unsatisfiable_classes=%d",
                reasoner.consistent,
                len(reasoner.unsatisfiable_classes),
            )
    elif run_reasoner:
        # The reasoner was asked for and could not run — consistency is unproven, not confirmed.
        # WARNING so it survives the API path, where the root logger stays at WARNING.
        logger.warning("validate: HermiT skipped (OWL 2 DL profile violated)")
    else:
        logger.info(
            "validate: HermiT skipped (run_reasoner=False)"
        )  # an explicit choice

    if run_oops and not oops_endpoint:
        # Asked for, but nowhere to ask. The scan posts the whole ontology in the request body,
        # so falling back to a built-in endpoint would pick a recipient on the operator's behalf —
        # the one decision this must never make for them. Skipped and said out loud: OOPS! never
        # blocks the gate (see `gate`), so refusing here costs no verification strength.
        report.oops_error = "no OOPS! endpoint configured (TBOXCRAFT_OOPS_ENDPOINT)"
        logger.warning(
            "validate: OOPS! scan skipped — %s. The scan sends the whole ontology to that "
            "endpoint, so it is not run against a default one.",
            report.oops_error,
        )
    elif run_oops:
        logger.info("validate: OOPS! scan -> %s", oops_endpoint)
        oops = scan_oops(rdfxml, endpoint=oops_endpoint, session=oops_session)
        report.oops_available = oops.available
        report.oops_pitfalls = [asdict(p) for p in oops.pitfalls]
        report.oops_error = oops.error
        if oops.available:
            logger.info("validate: OOPS! returned %d pitfalls", len(oops.pitfalls))
        else:
            logger.warning("validate: OOPS! unavailable: %s", oops.error)
    else:
        logger.info("validate: OOPS! scan skipped (run_oops=False)")

    return report


def gate(report: ValidationReport) -> tuple[bool, list[str]]:
    """Decide whether a pipeline may return this TBox. OOPS! never blocks."""
    reasons: list[str] = []
    if not report.syntax_ok:
        reasons.append(f"syntax: {report.syntax_error}")
    if not report.dl_profile_ok:
        reasons.append(f"OWL 2 DL punning: {', '.join(report.dl_violations)}")
    if report.reasoner_error:
        # HermiT is a hard gate: if it could not run we cannot prove consistency, so fail
        # closed rather than committing an unchecked ontology (a JVM/reasoner failure otherwise
        # slips through because consistent is unknown, not False).
        reasons.append(f"reasoner failed to run: {report.reasoner_error}")
    if report.consistent is False:
        reasons.append("ontology is inconsistent (HermiT)")
    if report.unsatisfiable_classes:
        reasons.append(
            f"unsatisfiable classes: {', '.join(report.unsatisfiable_classes)}"
        )
    return (not reasons, reasons)


def _by_severity(pitfalls: list[dict]) -> dict[str, int]:
    """Pitfall counts per grade. Always all three keys, plus "Other" only when it is non-zero.

    The buckets must sum to the total, or the breakdown is worse than the total it replaced — a
    reader who adds them up and lands short has no way to tell whether the gap is a bug here or a
    grade OOPS! invented. Anything outside the documented three is counted rather than dropped,
    and the key appears only when there is something in it so a normal response stays three keys.
    """
    counts = {level: 0 for level in SEVERITIES}
    for p in pitfalls:
        level = p.get("severity")
        bucket = level if level in counts else "Other"
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def summarize(report: ValidationReport) -> dict:
    """The gate's own account of how it decided — small enough to travel in an API response.

    ``ok`` alone cannot be audited: a run with the reasoner switched off leaves ``consistent``
    None, which gate() has no reason to refuse, so an unchecked TBox and a proven-consistent one
    return the same thing. ``reasoner_ran`` is what separates them, and it is the field a consumer
    treating PASS as "verified" has to read.

    OOPS! contributes two facts rather than one: ``oops_available`` false and zero pitfalls are
    not the same state, and collapsing them reads "clean" for a scan that never happened.
    """
    return {
        "syntax_ok": report.syntax_ok,
        "dl_profile_ok": report.dl_profile_ok,
        "dl_violations": report.dl_violations,
        # None = the reasoner did not run: either it was switched off, or the DL profile check
        # already failed and reasoning over an OWL-Full ontology is undefined.
        "reasoner_ran": report.consistent is not None
        or report.reasoner_error is not None,
        "consistent": report.consistent,
        "unsatisfiable_classes": report.unsatisfiable_classes,
        "reasoner_error": report.reasoner_error,
        "oops_available": report.oops_available,
        # Why a scan produced nothing. `oops_available: false` alone reads the same
        # for a network failure and for a scan nobody configured an endpoint for,
        # and only one of those is somebody's to fix.
        "oops_error": report.oops_error,
        "n_oops_pitfalls": len(report.oops_pitfalls),
        # OOPS! grades every pitfall Critical / Important / Minor, and a flat total throws that
        # away: 27 pitfalls could be 27 cosmetic ones or 3 modelling errors buried in 24. Nobody
        # can act on the total, and now that the scan runs on every batch the total is exactly the
        # number that will be ignored. Always all three keys, so a consumer reads a zero rather
        # than a missing key.
        "oops_by_severity": _by_severity(report.oops_pitfalls),
        # The rows, not just the counts — same reason unresolved_ranges and withheld_kind_conflicts
        # carry theirs. "P08: missing annotations, 41 elements" is something a curator can act on;
        # "3 Important" is not. Small: OOPS returns one row per pitfall type, not per element.
        "oops_pitfalls": report.oops_pitfalls,
    }
