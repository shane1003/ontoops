"""Extracts a document's top-level skeleton (division/chapter/appendix) from ODL blocks.

Why an LLM:
  Typography alone cannot capture the appendix hierarchy: appendix titles have the same
  font and ODL heading level as chapter titles, and there are no divider pages. The
  document carries no signal, so only semantic knowledge can answer. In exchange the
  LLM handles only the skeleton, and the inside (sections, paragraphs) is filled as is
  by odl_tree's heading level + marker + x0: the skeleton is exactly where typography
  fails, and the inside is where typography does well.

Why this shape:
  - Two separate steps: asking for identification and hierarchy at once buries the
    hierarchy judgment in the long input. Asking again with only the identified units
    as an outline gets it right.
  - Windowing + passing on confirmed units: given each window without context, the
    model does not know "which division is open now", promotes body section titles to
    divisions and mistakes table-of-contents entries for parts.
  - TSV with id/fs/text only: the result is the same without type/lvl/page. But one
    rank signal (fs) is required: without it, divisions collapse.
  - N per window: extracting one at a time ends the loop the moment the model emits
    null midway, missing everything after it.

Limitations:
  - The frame is built from ODL heading neighbors, so a document without headings (all
    top-level sections are list items) gets an empty skeleton. Such documents are
    handled without structure by odl_tree's marker+x0 path, which is harmless.
  - If validation fails, the whole structure is discarded. No skeleton is better than a
    wrong one.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import settings
from src.client import structure_llm
from src.models import IdentifiedUnits, StructuralUnitKind, UnitDepths
from src.odl_tree import Block, flatten_blocks
from src.prompt import build_odl_identify_user_prompt, load_prompt

# Text length per frame line; enough to recognize a title.
FRAME_TEXT_CHARS = 48
LABEL_CHARS = 44
PREFIX_CHARS = 20


class StructureUnit(BaseModel):
    title_id: int
    prefix_id: int | None = None
    kind: StructuralUnitKind
    depth: int
    # Text of the prefix + title blocks joined by the system; not a string the LLM made.
    title: str


class Structure(BaseModel):
    block_count: int
    window_count: int
    units: list[StructureUnit] = Field(default_factory=list)


# =============================================================================
# 1. Frame: choosing the candidate lines shown to the LLM
# =============================================================================


def frame_lines(blocks: list[Block]) -> list[str]:
    """Keeps only headings and the one block before and after each.

    The neighbors are needed because ODL tags prefixes like 'PART TWO'/'Chapter 3' as
    paragraphs, not headings. Keeping only headings would drop the prefix and the title
    would lose its number.
    """
    keep: set[int] = set()
    for i, b in enumerate(blocks):
        if b["type"] == "heading":
            keep.update({i - 1, i, i + 1})
    return [
        f"{blocks[i]['id']}\tfs={blocks[i]['font_size']}\t{blocks[i]['text'][:FRAME_TEXT_CHARS]}"
        for i in sorted(keep)
        if 0 <= i < len(blocks)
    ]


def chunk_lines(lines: list[str], max_chars: int) -> list[list[str]]:
    """Splits lines into windows at line boundaries, keeping each within max_chars."""
    out: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for ln in lines:
        if cur and size + len(ln) > max_chars:
            out.append(cur)
            cur, size = [], 0
        cur.append(ln)
        size += len(ln) + 1
    if cur:
        out.append(cur)
    return out


# =============================================================================
# 2. Step 1 (identification) + step 3 (hierarchy)
# =============================================================================


class StructureLLMError(RuntimeError):
    """The LLM produced no usable output.

    A server-side failure, so it is reported as 5xx, separate from input errors
    (ValueError, 422).
    """


def _ids_in_document(unit: Any, by_id: dict[int, Block]) -> bool:
    """Whether title_id and prefix_id are both blocks of this document.

    prefix_id is checked too because _label and _joined_title index by_id without a
    guard.
    """
    return unit.title_id in by_id and (
        unit.prefix_id is None or unit.prefix_id in by_id
    )


def _label(unit: Any, by_id: dict[int, Block]) -> str:
    title = by_id[unit.title_id]["text"][:LABEL_CHARS]
    if unit.prefix_id is None:
        return title
    return f"{by_id[unit.prefix_id]['text'][:PREFIX_CHARS]} - {title}"


def identify(
    windows: list[list[str]], by_id: dict[int, Block], callbacks: list | None = None
) -> list[Any]:
    """Collects structural units window by window.

    Confirmed units are carried into the next window as context.
    """
    llm = structure_llm.with_structured_output(IdentifiedUnits, method="json_mode")  # type: ignore[arg-type]  # supported at runtime (only the type stub is missing)
    system = load_prompt(settings.odl_structure_identify_prompt_path)
    acc: list[Any] = []

    for i, window in enumerate(windows, start=1):
        confirmed = "\n".join(
            f"{u.title_id}\t{u.kind.value}\t{_label(u, by_id)}" for u in acc
        )
        user = build_odl_identify_user_prompt(
            confirmed, "\n".join(window), i, len(windows)
        )
        result = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)],
            config={"callbacks": callbacks or [], "run_name": "structure_identify"},
        )
        # Silently skipping a window lets a skeleton missing only that window's units
        # pass validation, and the missing chapter's body is attributed to the previous
        # chapter in the provenance. json_mode returns dict|BaseModel, hence isinstance.
        if not isinstance(result, IdentifiedUnits):
            raise StructureLLMError(
                f"No structured output received for window {i}/{len(windows)}."
            )
        known = {u.title_id for u in acc}
        fresh = [u for u in result.units if u.title_id not in known]
        # A unit pointing to a missing id is a failure that drops its chapter from the
        # skeleton, but it is only counted, not raised. Rejecting without knowing the
        # rate could reject many sound documents, so this log measures the frequency.
        dropped = [u for u in fresh if not _ids_in_document(u, by_id)]
        if dropped:
            logging.warning(
                f"[odl-structure] window {i}/{len(windows)}: dropped "
                f"{len(dropped)} unit(s) with ids not in the document - "
                + ", ".join(
                    f"title_id={u.title_id}/prefix_id={u.prefix_id}" for u in dropped
                )
            )
        acc.extend(u for u in fresh if _ids_in_document(u, by_id))
    return sorted(acc, key=lambda u: u.title_id)


def hierarchize(
    units: list[Any], by_id: dict[int, Block], callbacks: list | None = None
) -> dict[int, int]:
    """Assigns depths from an outline of the identified units only.

    The input is ~1KB, so the context does not get diluted.
    """
    outline = "\n".join(
        f"{u.title_id}\tp{by_id[u.title_id]['page']}\t{_label(u, by_id)}" for u in units
    )
    llm = structure_llm.with_structured_output(UnitDepths, method="json_mode")  # type: ignore[arg-type]
    # The parser exception (OutputParserException) is a ValueError: left alone, api.py
    # maps it to 422 and the full model response in str(e) ends up in the response body.
    # The original is kept only in the log, via __cause__.
    try:
        result = llm.invoke(
            [
                SystemMessage(
                    content=load_prompt(settings.odl_structure_hierarchy_prompt_path)
                ),
                HumanMessage(content=outline),
            ],
            config={"callbacks": callbacks or [], "run_name": "structure_hierarchize"},
        )
    except ValueError as e:  # pydantic ValidationError is a subclass too
        raise StructureLLMError(
            "No structured output received for the depth judgment."
        ) from e
    if not isinstance(result, UnitDepths):
        raise StructureLLMError("No structured output received for the depth judgment.")
    return {u.id: u.depth for u in result.units}


# =============================================================================
# 3. Validation: check against a closed set. Any failure discards everything.
# =============================================================================


def validation_errors(
    units: list[Any], depths: dict[int, int], by_id: dict[int, Block]
) -> list[str]:
    """Checks the units against the closed set of the block list.

    The LLM only picked ids, so they are checked against the block list. If any check
    fails, the skeleton is discarded and the coordinate/marker tree is used as fallback.

    Notes:
        On the generation path, identify() passes only existing ids, sorted, so some
        checks never fire. They are there for the upload path (validate_uploaded), which
        receives hand-edited files.
    """
    errors: list[str] = []
    ids = [u.title_id for u in units]
    if not units:
        return ["no structural units"]
    if len(set(ids)) != len(ids):
        errors.append("duplicate title_id")
    if ids != sorted(ids):
        errors.append("reading order violated")
    if any(u.title_id not in by_id for u in units):
        errors.append("nonexistent title_id")
    if any(u.prefix_id is not None and u.prefix_id not in by_id for u in units):
        errors.append("nonexistent prefix_id")
    if any(u.prefix_id is not None and u.prefix_id >= u.title_id for u in units):
        errors.append("prefix after title")
    # A prefix can only be the block right before the title (frame_lines shows only
    # heading±1). Allowing a distant prefix would make _joined_title promote the body in
    # between to the title. prefix/title blocks are merge_wrapped barriers, so their ids
    # stay adjacent after merging.
    if any(u.prefix_id is not None and u.prefix_id < u.title_id - 1 for u in units):
        errors.append("prefix not right before the title")
    missing = [i for i in ids if i not in depths]
    if missing:
        errors.append(f"{len(missing)} unit(s) without depth")
    elif min(depths[i] for i in ids) != 0:
        errors.append("no top level (depth 0)")
    elif any(
        depths[b] - depths[a] > 1 for a, b in zip(ids, ids[1:])
    ):  # a child is at most parent+1
        errors.append("depth jump")
    return errors


# =============================================================================
# 4. Public entry points
# =============================================================================


def validate_uploaded(odl: dict[str, Any], structure: Structure) -> list[str]:
    """Checks whether an uploaded skeleton belongs to this document.

    Hand-edited files and files from other documents arrive here.

    Notes:
        Block ids are positions in the flatten_blocks result, so for another document
        they silently point to the wrong blocks. If block_count differs, none of the ids
        can be trusted.
    """
    blocks = flatten_blocks(odl)
    by_id = {b["id"]: b for b in blocks}
    errors: list[str] = []
    if structure.block_count != len(blocks):
        errors.append(
            f"skeleton of another document (blocks {structure.block_count} vs "
            f"{len(blocks)})"
        )
    errors += validation_errors(
        structure.units, {u.title_id: u.depth for u in structure.units}, by_id
    )
    # title is recomputed from the block text, so it can be checked. Without the check,
    # a title not in the document would go out as provenance (section_path). Units with
    # missing ids were already caught by validation_errors and would raise KeyError
    # when recomputed here, so they are skipped.
    if any(
        u.title_id in by_id
        and (u.prefix_id is None or u.prefix_id in by_id)
        and u.title != _joined_title(u, by_id)
        for u in structure.units
    ):
        errors.append("title differs from the document's block text")
    return errors


def build_structure(odl: dict[str, Any], callbacks: list | None = None) -> Structure:
    """Extracts the document skeleton from ODL JSON.

    Raises:
        ValueError: The input has a shape no skeleton can be built from.
        StructureLLMError: The LLM produced no usable output.
    """
    blocks = flatten_blocks(odl)
    if not blocks:
        raise ValueError("No text blocks found in the ODL output.")

    lines = frame_lines(blocks)
    if not lines:
        raise ValueError(
            "Cannot build a skeleton: no block was judged a heading. "
            "Process this document without structure, via the coordinate/marker path."
        )

    by_id = {b["id"]: b for b in blocks}
    windows = chunk_lines(lines, settings.odl_structure_window_chars)
    # identify() calls the LLM sequentially per window, so the window count = how long
    # one request holds the server. The size cap (cq_max_json_bytes) alone cannot
    # prevent it: even a small document gets more windows when headings are dense.
    if len(windows) > settings.odl_structure_max_windows:
        raise ValueError(
            f"{len(windows)} windows exceed the limit of "
            f"{settings.odl_structure_max_windows}. Split the document."
        )
    units = identify(windows, by_id, callbacks)
    if not units:
        raise ValueError("No structural units were identified.")

    depths = hierarchize(units, by_id, callbacks)
    errors = validation_errors(units, depths, by_id)
    if errors:
        raise ValueError(f"Skeleton validation failed: {', '.join(errors)}")

    logging.info(
        f"[odl-structure] {len(blocks)} blocks / {len(lines)} frame lines / "
        f"{len(windows)} windows -> {len(units)} units"
    )
    return Structure(
        block_count=len(blocks),
        window_count=len(windows),
        units=[
            StructureUnit(
                title_id=u.title_id,
                prefix_id=u.prefix_id,
                kind=u.kind,
                depth=depths[u.title_id],
                title=_joined_title(u, by_id),
            )
            for u in units
        ],
    )


def _joined_title(unit: Any, by_id: dict[int, Block]) -> str:
    title = by_id[unit.title_id]["text"]
    if unit.prefix_id is None:
        return title
    return f"{by_id[unit.prefix_id]['text']} {title}"
