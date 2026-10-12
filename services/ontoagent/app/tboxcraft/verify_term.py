"""Domain-term verifier.

Asks the LLM whether a candidate is the official English term used in this domain, grounded on
the existing ontology (nearest concepts' labels) plus the model's own knowledge — no external
glossary. Returns a :class:`~tboxcraft.llm.base.Verdict`; any transport/parse failure degrades
to ``Verdict.unavailable`` so the human review is never blocked.
"""

from __future__ import annotations

import logging

from app.tboxcraft.llm.base import LLMProvider, Verdict
from app.tboxcraft.models import Component

logger = logging.getLogger(__name__)

_ALLOWED_ACTIONS = {"merge", "new", "relabel", "delete", "reject"}

# The domain the verifier judges "official term" against is not baked in — it is named per
# deployment via config.domain_description and interpolated here. "Official" is domain-relative,
# so the model has to be told which domain; but which domain is a config choice, not this module's.
# The default is neutral, so tboxcraft is general-purpose; a deployment sets its own.
_DEFAULT_DOMAIN = "specialized-domain"

_FIELDS = (
    "is_official_term (boolean), is_standard_form (boolean), recommended_action (one of merge, "
    "new, relabel, delete, reject), proposed_term (string or null — the official term if a relabel "
    "is warranted), rationale (string), confidence (number 0..1)"
)


def _task(domain: str) -> str:
    return (
        f"You are a terminology verifier for a {domain} ontology. "
        "Given a candidate term and the nearest existing ontology concepts, judge whether the "
        "candidate is the official English term used in this domain. Ground your judgment on the "
        "provided context and your knowledge. Also judge is_standard_form: whether the candidate "
        "is a well-formed, standard term for its concept (an acceptable synonym/abbreviation, kept "
        "as an AlterTerm) as opposed to a non-standard document artifact — the right concept but a "
        "messy or colloquial surface form (e.g. a sentence-like phrasing) that should be a "
        "HiddenTerm kept only for text mapping, not a standard synonym."
    )


def _system(domain: str) -> str:
    return f"{_task(domain)} Respond ONLY with a JSON object with fields: {_FIELDS}."


def _batch_system(domain: str) -> str:
    # Differs from the single-candidate system prompt in two places only: the reply is a keyed
    # list, and the neighbours are context rather than a group to judge collectively. The judging
    # task itself (``_task``) is word-for-word identical, which is what makes a batched run
    # comparable to an unbatched one — otherwise a verdict change could not be told apart from a
    # prompt change.
    return (
        f"{_task(domain)} You are given several candidates at once, each with an id. Judge every "
        "candidate separately, on its own merits: the others are shown so you can see the "
        "neighbourhood a term sits in, never so that one candidate's judgment carries over to "
        'another. Respond ONLY with a JSON object {"verdicts": [...]} holding exactly one entry '
        f"per candidate. Each entry is the candidate's id (integer) plus the fields: {_FIELDS}. "
        "Answer every id you were given, each exactly once, and invent no others."
    )


def _candidate_block(component: Component, gloss: str | None) -> str:
    parts = [f"Candidate term: {component.name}", f"Kind: {component.kind}"]
    if component.domain:
        parts.append(f"Domain: {component.domain}")
    if component.range_raw:
        parts.append(f"Range: {component.range_raw}")
    if gloss:
        parts.append(f"Gloss: {gloss}")
    elif component.comment:
        parts.append(f"Note: {component.comment}")
    return "\n".join(parts)


def _nearest_lines(nearest: list[dict]) -> list[str]:
    """The neighbourhood block. Shared so the batched and unbatched prompts state it identically —
    a verdict that changed because the wording differed could not be told from one that changed
    because the judgment did."""
    lines = ["Nearest existing ontology concepts:"]
    if nearest:
        for n in nearest:
            score = n.get("score")
            score_s = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
            lines.append(
                f"- {n.get('label') or n.get('iri')} "
                f"(iri={n.get('iri')}, similarity={score_s})"
            )
    else:
        lines.append("- (none)")
    return lines


def build_prompt(
    component: Component,
    gloss: str | None,
    nearest: list[dict],
    *,
    domain: str = _DEFAULT_DOMAIN,
) -> tuple[str, str]:
    """Return ``(system, user)``. ``nearest`` items carry ``iri``, ``label``, and ``score``."""
    lines = [_candidate_block(component, gloss), ""] + _nearest_lines(nearest)
    lines += [
        "",
        "Is the candidate the official English domain term? Respond as the JSON object.",
    ]
    return _system(domain), "\n".join(lines)


def build_batch_prompt(
    components: list[Component],
    glosses: dict[str, str] | None = None,
    nearest_by_uid: dict[str, list[dict]] | None = None,
    *,
    domain: str = _DEFAULT_DOMAIN,
) -> tuple[str, str]:
    """Return ``(system, user)`` for judging several candidates in one call.

    Every candidate is numbered, and the reply has to carry the same ids back. The id is the only
    thing tying a verdict to the term it was written for: matching on position would silently
    mis-attribute every verdict after the first one the model skipped or reordered.

    Each candidate carries its own gloss and neighbourhood, exactly as the unbatched prompt does.
    This block used to be omitted here, so the bootstrap asked a strictly poorer question than the
    update path did about the same term. What masked it is that batch-mates are themselves a
    neighbourhood — but that is a side effect of how batches are grouped, not the neighbourhood
    being stated: it disappears the moment ``verify_cluster_size`` is set to 1, which is a live
    A/B lever, and it never carried similarity scores at all.
    """
    lines = [
        f"Judge each of the following {len(components)} candidate terms, independently."
    ]
    for i, component in enumerate(components, start=1):
        lines += [
            "",
            f"--- id: {i} ---",
            _candidate_block(component, (glosses or {}).get(component.uid)),
        ]
        lines += _nearest_lines((nearest_by_uid or {}).get(component.uid) or [])
    lines += [
        "",
        'Respond as the JSON object: {"verdicts": [...]}, one entry per id above.',
    ]
    return _batch_system(domain), "\n".join(lines)


def _coerce(raw: dict) -> Verdict:
    try:
        action = str(raw.get("recommended_action", "reject")).strip().lower()
        if action not in _ALLOWED_ACTIONS:
            action = "reject"
        try:
            conf = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
        except (TypeError, ValueError):
            conf = 0.0
        standard_form = raw.get("is_standard_form")
        return Verdict(
            is_official_term=bool(raw.get("is_official_term", False)),
            recommended_action=action,
            proposed_term=(raw.get("proposed_term") or None),
            rationale=str(raw.get("rationale", "")),
            confidence=conf,
            # Absent OR explicit null -> True: treat an unstated form as a standard AlterTerm
            # (safe default). Only an explicit false marks it a non-standard HiddenTerm.
            is_standard_form=True if standard_form is None else bool(standard_form),
        )
    except Exception as exc:  # noqa: BLE001 - a malformed verdict must not raise into review
        return Verdict.unavailable(f"malformed verdict: {exc}")


def verify_term(
    component: Component,
    provider: LLMProvider,
    *,
    gloss: str | None = None,
    nearest: list[dict] | None = None,
    domain: str = _DEFAULT_DOMAIN,
) -> Verdict:
    """Judge one candidate. Never raises — returns ``Verdict.unavailable`` on any failure."""
    system, user = build_prompt(component, gloss, nearest or [], domain=domain)
    try:
        raw = provider.judge(system, user)
    except Exception as exc:  # noqa: BLE001 - degrade, human still decides
        logger.warning(
            "verify_term: provider unavailable for %s: %s", component.uid, exc
        )
        return Verdict.unavailable(str(exc))
    return _coerce(raw)


def _index_by_id(raw: object, n: int) -> dict[int, dict]:
    """Map a batch reply back onto the ids it was asked about.

    Only ids that were given and were answered exactly once survive. An id answered twice is
    dropped rather than resolved by first-wins: two verdicts for one term means the model lost
    track of which candidate it was on, and picking either would attach a judgment possibly
    written about its neighbour. The caller re-asks for whatever does not survive.
    """
    entries = raw.get("verdicts") if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        return {}
    by_id: dict[int, dict] = {}
    answered_twice: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            eid = int(entry["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 1 <= eid <= n:
            continue  # an id nobody asked about says nothing about the candidates that exist
        if eid in by_id:
            answered_twice.add(eid)
            continue
        by_id[eid] = entry
    for eid in answered_twice:
        del by_id[eid]
    return by_id


def verify_terms(
    components: list[Component],
    provider: LLMProvider,
    *,
    glosses: dict[str, str] | None = None,
    nearest_by_uid: dict[str, list[dict]] | None = None,
    domain: str = _DEFAULT_DOMAIN,
) -> tuple[list[Verdict], int]:
    """Judge a batch of candidates in one call. Never raises.

    Returns ``(verdicts positionally aligned to components, LLM calls actually made)``. The call
    count is returned rather than logged because it is the measure of whether batching paid off,
    and a log line cannot be asserted on.

    The two ways a batch can fail are handled differently, on purpose:

    * **the provider raises** — the server is down or refusing. Re-asking once per candidate would
      multiply a dead server's traffic by the batch size, so every candidate in the batch degrades
      to ``unavailable`` off that single call.
    * **the server answered, but the reply does not map onto the ids** — the server is alive and
      only the model's bookkeeping failed. The unanswered candidates are re-asked one at a time,
      which costs calls; that cost shows up in the returned count instead of hiding as a batch of
      spurious "unavailable" verdicts.
    """
    if len(components) <= 1:
        # A single-candidate batch is the un-batched prompt verbatim, so batching can always be
        # switched off without changing what the verifier is asked.
        return [
            verify_term(
                c,
                provider,
                gloss=(glosses or {}).get(c.uid),
                nearest=(nearest_by_uid or {}).get(c.uid),
                domain=domain,
            )
            for c in components
        ], len(components)

    system, user = build_batch_prompt(
        components, glosses, nearest_by_uid, domain=domain
    )
    try:
        raw = provider.judge(system, user)
    except Exception as exc:  # noqa: BLE001 - degrade, human still decides
        logger.warning(
            "verify_terms: provider unavailable for a batch of %d: %s",
            len(components),
            exc,
        )
        return [Verdict.unavailable(str(exc)) for _ in components], 1

    by_id = _index_by_id(raw, len(components))
    verdicts: list[Verdict | None] = [
        _coerce(by_id[i]) if i in by_id else None for i in range(1, len(components) + 1)
    ]
    calls = 1
    unanswered = [i for i, v in enumerate(verdicts) if v is None]
    if unanswered:
        logger.warning(
            "verify_terms: batch of %d came back with %d candidate(s) unmatched (missing or "
            "duplicated id) — re-asking those individually",
            len(components),
            len(unanswered),
        )
        for i in unanswered:
            verdicts[i] = verify_term(
                components[i],
                provider,
                gloss=(glosses or {}).get(components[i].uid),
                nearest=(nearest_by_uid or {}).get(components[i].uid),
                domain=domain,
            )
            calls += 1
    return verdicts, calls
