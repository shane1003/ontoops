"""Routing on the upstream TBox/ABox verdict, shared by ``generate`` and ``update``.

Its own module because both pipelines need it and ``update_apply`` imports ``generate`` (a cold
``/update`` bootstraps through it), so the rule cannot live in either without a cycle. It lived in
``update_apply`` alone, which meant a bootstrapping run — the first ingest into an empty store,
i.e. exactly the common case — routed nothing at all.
"""

from __future__ import annotations

import logging
from collections import Counter

logger = logging.getLogger(__name__)

VERDICTS = ("tbox", "conflict", "abox")


def route_by_verdict(
    components: list, *, stage: str = "ingest"
) -> tuple[dict[str, str], Counter]:
    """Read the upstream TBox/ABox verdict. Returns ``({uid: verdict} for the flagged ones, tally)``.

    ``Component.verdict`` was parsed but never read, so the distinction the upstream classifier
    exists to draw reached nothing.

    **Every component is minted ACTIVE, whatever the verdict** — including ``abox``. That is
    deliberate, and it is not the ontologically tidy choice:

    An individual names its class with a string byte-identical to a component name, and the ABox
    loader resolves it through ``surface_form`` restricted to ACTIVE terms. Drop the component and
    every individual anchored on that name misses; and a component judged ``abox`` is *more* likely
    to be such an anchor, not less ("this is an instance" usually means it appears as one). There
    is no middle state to park it in: the store allows DRAFT, but the loader resolves ACTIVE only,
    so DRAFT preserves nothing.

    So the choice is: mint it (an instance is modelled as a class — wrong, but correctable) or drop
    it (the individuals are lost — silent and not correctable). Both flagged verdicts are minted
    and queued for review, where merge/relabel/delete can fix the modelling with the anchor intact.

    This tallies whatever it is given, so it is only honest if nothing was filtered upstream. The
    loader used to keep "tbox" only, which made this report ``n_abox=0`` on a batch that had
    them — the counter meant to expose the drop asserted it had not happened.
    """
    flagged: dict[str, str] = {}
    tally: Counter = Counter()
    for c in components:
        verdict = (getattr(c, "verdict", None) or "tbox").strip().lower()
        if verdict not in VERDICTS:
            logger.warning(
                "%s: unknown verdict %r on %s; treating it as tbox",
                stage,
                verdict,
                c.name,
            )
            verdict = "tbox"
        tally[verdict] += 1
        if verdict != "tbox":
            flagged[c.uid] = verdict
    if tally["abox"]:
        logger.warning(
            "%s: %d component(s) were classified as individuals (verdict='abox') but are "
            "minted as TBox terms to keep them resolvable as anchors; each is queued for review",
            stage,
            tally["abox"],
        )
    if tally["conflict"]:
        logger.info(
            "%s: %d component(s) carry verdict='conflict'; minting and queueing for review",
            stage,
            tally["conflict"],
        )
    return flagged, tally


def flagged_reason(verdict: str) -> str:
    """Why this component is in the queue, in the reviewer's terms."""
    if verdict == "conflict":
        return "upstream verdict=conflict (chunks disagreed)"
    return (
        "upstream verdict=abox (classified as an individual, minted as a term to keep it "
        "resolvable as an anchor)"
    )


def surface_role(verdict, floor: float) -> str:
    """The surface role for a term form: ``'alter'`` (a standard synonym) or ``'hidden'`` (kept for
    ABox mapping only).

    HiddenTerm only when the verifier is *confidently* sure the form is non-standard; otherwise a
    safe AlterTerm. A low-confidence "non-standard" verdict, or an unavailable one, is not trusted
    enough to bury a term — the verdict is advisory and can hallucinate on niche jargon.

    Shared rather than duplicated: the update path decided this per attach while the bootstrap
    wrote every surface as 'alter', so ``hidden_confidence_floor`` had no effect at all on the
    first ingest. Two copies of the rule would drift the same way again.
    """
    if (
        verdict is not None
        and not verdict.is_standard_form
        and verdict.confidence >= floor
    ):
        return "hidden"
    return "alter"
