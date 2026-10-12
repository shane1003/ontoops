"""Deterministic gloss assembly.

Each component becomes one gloss string with no LLM. Because ``comment`` is null and
``correct_term`` sparse in the batch while ``domain_layer.notes`` is Korean-dominant, the
gloss must be viable from ``name`` + ``notes`` alone — so ``name`` (de-split from CamelCase)
always anchors it. For properties, ``domain``/``range`` are folded in.
"""

from __future__ import annotations

import re

from app.tboxcraft.io.cq import CQIndex
from app.tboxcraft.models import Component

# Insert a space at lowercase→uppercase and acronym→Word boundaries.
_CAMEL_1 = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_2 = re.compile(r"([A-Z]+)([A-Z][a-z])")

# A generous character cap; the embedding provider enforces its own token limit on top.
DEFAULT_MAX_CHARS = 2000


def desplit_camel(name: str) -> str:
    """``InfantryBrigade`` → ``Infantry Brigade``; ``hasAreaOfInterest`` → ``has Area Of Interest``.

    Runs of capitals (acronyms) are preserved as single tokens.
    """
    spaced = _CAMEL_2.sub(r"\1 \2", _CAMEL_1.sub(r"\1 \2", name))
    return spaced.strip()


def build_gloss(
    component: Component,
    cq_index: CQIndex | None = None,
    *,
    max_cq_sample: int = 5,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Assemble the deterministic gloss for one component.

    Ordering is fixed so the same input always yields the same string.
    """
    parts: list[str] = [desplit_camel(component.name)]

    if component.correct_term:
        parts.append(component.correct_term)

    # domain_layer.notes is the dense (Korean) field; linguistic notes are usually sparser.
    if component.dom_notes:
        parts.append(component.dom_notes)

    if component.is_property:
        if component.domain:
            parts.append(f"domain: {component.domain}")
        ranges = component.range_values()
        if ranges:
            parts.append(f"range: {', '.join(ranges)}")

    if cq_index is not None and component.source_cq_ids:
        sample_ids = component.source_cq_ids[:max_cq_sample]
        cq_texts = cq_index.texts_for(sample_ids)
        parts.extend(cq_texts)

    gloss = ". ".join(part.rstrip(". ") for part in parts if part).strip()
    if len(gloss) > max_chars:
        gloss = gloss[:max_chars].rstrip()
    return gloss
