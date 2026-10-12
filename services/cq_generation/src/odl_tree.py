"""Assembles opendataloader-pdf output (JSON) into a section tree for MCS chunking.

ODL gives paragraph-level segmentation and heading levels but no hierarchy: every
heading is listed as a top-level sibling, and in numbered documents (manuals, orders)
the hierarchy lives in the numbering of list items, not in headings. So the tree is
built by combining two signals:

  1. heading level = the outer skeleton. A monotonic stack runs on the absolute level.
     The levels themselves are noisy, but only their relative order is used, so the
     noise is mostly absorbed.
  2. marker sequence = the hierarchy inside a heading. An 'a.' after '1.' is a new
     family, so a child; 'b.' continues the ongoing LOWER_DOT, so a sibling; '2.'
     returns to ARABIC_DOT, so the 'a.' level is popped.

When the two signals conflict, the marker wins. ODL sometimes tags list items as
headings, so a heading whose marker continues the current sequence is demoted to a list
item (section 4, tree assembly).

Headers/footers are not detected: ODL already removes most of them, and applying the
"repeated at the same vertical position means footer" rule to the remaining pieces
deletes more body headings than real footers, because chapter opening pages share the
same layout. Detecting the leftovers was split off as a separate task.
"""

from __future__ import annotations

import re
from statistics import median
from typing import Any, Iterator, Protocol, Sequence

from src.markers import MARKER_PRIORITY_ORDER, MarkerMatch, match_markers
from src.mcs_chunker import Node

# =============================================================================
# 1. Flattening: only blocks with text, in reading order
# =============================================================================

TEXT_TYPES = {"paragraph", "heading", "list item", "text block"}
# Tables/images/captions are excluded from chunking (captions describe figures and
# tables, so they go with them). 'list' is only a container without text of its own,
# so it is filtered out naturally.

SECTION_PATH_MAX_CHARS = 80  # length cap of one breadcrumb step


Block = dict[str, Any]


class SpineUnit(Protocol):
    """Shape satisfied by odl_structure.StructureUnit.

    Taken structurally only, to avoid a reverse import.
    """

    title_id: int
    prefix_id: int | None
    depth: int
    title: str


Spine = Sequence[SpineUnit]


def _walk(nodes: list[Block]) -> Iterator[Block]:
    """Flattens the ODL tree in reading order (pre-order).

    kids and list items are both children.
    """
    for n in nodes:
        yield n
        yield from _walk((n.get("kids") or []) + (n.get("list items") or []))


# Filler text of blank pages. ODL tags it as a heading, so left alone it pops the stack
# and swallows the real section that follows as its child. Filler is detected by text
# only: "is this the only thing on the page" cannot tell it apart from a page that holds
# a single body block.
_FILLER = re.compile(r"intentionally\s+left\s+blank", re.I)


def flatten_blocks(odl: dict[str, Any]) -> list[Block]:
    """Extracts only the text blocks from ODL JSON, in reading order.

    id is the position in this list. It is exactly the id odl_structure asks the LLM to
    pick, so the same ODL JSON must yield the same ids (hence it is assigned here, and
    merge_wrapped keeps the id of the first piece).
    """
    out: list[Block] = []
    for n in _walk(odl.get("kids") or []):
        content = (n.get("content") or "").strip()
        bbox = n.get("bounding box")
        page = n.get("page number")
        if not content or n.get("type") not in TEXT_TYPES or not bbox:
            continue
        if _FILLER.search(content):
            continue
        out.append(
            {
                "id": len(out),
                "type": str(n["type"]),
                "level": n.get("heading level"),
                "page": page,
                "x0": float(bbox[0]),
                "x1": float(bbox[2]),
                "font_size": n.get("font size"),
                "text": content,
            }
        )
    return out


# =============================================================================
# 2. Rejoining pieces split by line wraps
# =============================================================================

# Endings that count as a finished sentence: if a block ends here, the next block is a
# new paragraph.
_SENTENCE_END = re.compile(r'[.。?!:;”")\]]\s*$')

RIGHT_MARGIN_TOLERANCE = 2.0


def _right_margin(blocks: list[Block]) -> float:
    """Returns the body's right edge; a line reaching it was wrapped by overflow."""
    return max((float(b["x1"]) for b in blocks), default=0.0)


def _join_wrapped(a: str, b: str) -> str:
    """Joins two pieces split by a line wrap.

    Korean breaks mid-word, so the pieces are joined directly; ASCII words break at word
    boundaries, so a space is inserted.
    """
    if a and b and (a[-1].isascii() or b[0].isascii()):
        return f"{a} {b}"
    return a + b


def merge_wrapped(blocks: list[Block], barriers: set[int] | None = None) -> list[Block]:
    """Appends the next block to a block that reaches the right margin mid-sentence.

    ODL sometimes emits one paragraph as several blocks. Both end up on the same node
    anyway, so chunk boundaries do not cut the paragraph, but the text keeps mid-word
    line breaks that make it hard to read.

    Limitation: right-aligned cover text has every line touching the margin and gives
    false positives. Geometry alone cannot tell right alignment from overflow.

    ids in barriers are neither absorbed nor absorb the following block. If a block the
    skeleton points to (a prefix like 'PART TWO' is a paragraph too) were absorbed, its
    id would vanish and the segment could not be cut; if that block absorbed the next
    one, the swallowed body would vanish: _segments drops the title block from the
    segment and build_tree uses the pre-merge unit.title as the node text, so it has
    nowhere to go.
    """
    if not blocks:
        return []
    barriers = barriers or set()
    margin = _right_margin(blocks) - RIGHT_MARGIN_TOLERANCE
    out: list[Block] = [dict(blocks[0], pages=[blocks[0]["page"]])]
    for b in blocks[1:]:
        prev = out[-1]
        prev_text: str = prev["text"]
        wrapped = (
            float(prev["x1"]) >= margin
            and not _SENTENCE_END.search(prev_text)
            and b["type"] == prev["type"] == "paragraph"
            and b["id"] not in barriers
            and prev["id"] not in barriers
        )
        if wrapped:
            prev["text"] = _join_wrapped(prev_text, b["text"])
            if b["page"] not in prev["pages"]:
                prev["pages"].append(b["page"])
            # Keep x0 of the first piece (= the line with the marker): continuation
            # pieces are pushed right by hanging indent and would misread the depth.
            prev["x1"] = b["x1"]  # next overflow check uses the last piece's right end
        else:
            out.append(dict(b, pages=[b["page"]]))
    return out


# =============================================================================
# 3. Depth of list items: x0 decides, the marker assists
# =============================================================================

_PRIORITY = {name: i for i, name in enumerate(MARKER_PRIORITY_ORDER)}

X0_TOLERANCE_FACTOR = 0.6  # x0 gaps under this × font size are the same indent

MarkerScope = tuple[float, str, int]  # (x0, family, last value)


def x0_tolerance(blocks: list[Block]) -> float:
    """Returns the tolerance for judging equal indentation.

    Derived from the font size so it is not hardcoded per document. x0 jitters even
    within one list, so the tolerance must exceed that jitter but stay below the width
    of a real indent step. X0_TOLERANCE_FACTOR places it in between.
    """
    sizes = [float(b["font_size"]) for b in blocks if b.get("font_size")]
    return X0_TOLERANCE_FACTOR * (median(sizes) if sizes else 10.0)


def resolve_marker_depth(
    matches: list[MarkerMatch], stack: list[MarkerScope], x0: float, tolerance: float
) -> tuple[int, MarkerMatch]:
    """Decides the depth of a block with a marker and which marker to adopt.

    Depth is decided by x0, not by the marker family. Deciding by family breaks two
    things:
      - when a list ends and a new list starts further left, it becomes a child because
        it is a new family, and everything after it is dragged along;
      - when the same family restarts at another depth, it concludes there is nowhere
        to put it because that family is already open.
    x0 answers both cases correctly.

    The marker's role is (a) to say this block is an item, not body text, and (b) to
    choose among ambiguous candidates. Continuing the family a sibling at the same
    position was using takes precedence over the priority table: '(c)' is both
    LOWER_PAREN #3 (ongoing) and the first ROMAN_LOWER_PAREN (C=100), and the table
    alone would pick the latter and make the list jump to Roman numerals.
    """
    # A scope opened at or to the right of this block cannot be its ancestor.
    depth = len(stack)
    while depth > 0 and stack[depth - 1][0] >= x0 - tolerance:
        depth -= 1

    replaced = stack[depth] if depth < len(stack) else None
    if replaced is not None and abs(replaced[0] - x0) < tolerance:
        continuing = [
            m for m in matches if m.family == replaced[1] and m.value > replaced[2]
        ]
        if continuing:
            return depth, continuing[0]
    return depth, min(matches, key=lambda m: _PRIORITY.get(m.family, 999))


# =============================================================================
# 4. Tree assembly: heading stack + marker stack
# =============================================================================


def _breadcrumb(text: str) -> str:
    head = text.split("\n", 1)[0].strip()
    if len(head) > SECTION_PATH_MAX_CHARS:
        return head[: SECTION_PATH_MAX_CHARS - 1] + "…"
    return head


def _build_interior(root: Node, blocks: list[Block]) -> None:
    """Assembles a block list into a hierarchy under root (heading level + marker + x0).

    With a skeleton (odl_structure), only the segment of one structural unit comes in;
    without one, the whole document does. Either way this inner logic is the same: the
    skeleton decides only the outside and does not touch the inside.
    """
    heading_stack: list[tuple[int, Node]] = []  # (heading level, node)
    marker_stack: list[MarkerScope] = []
    marker_nodes: list[Node] = []  # nodes at the same depth as marker_stack
    open_node = root  # where body text without a marker attaches
    tolerance = x0_tolerance(blocks)

    def current_heading() -> Node:
        return heading_stack[-1][1] if heading_stack else root

    for b in blocks:
        text: str = b["text"]
        matches = match_markers(text)
        placed = (
            resolve_marker_depth(matches, marker_stack, float(b["x0"]), tolerance)
            if matches
            else None
        )
        level = b["level"] if b["type"] == "heading" else None

        # Even a heading is a list item if its marker lands inside an already open
        # list. ODL sometimes tags items in the middle of a list as headings; trusting
        # that would pop the stack wrongly and attach what follows to the wrong parent.
        if level is not None and placed is not None and placed[0] < len(marker_stack):
            level = None

        if level is not None:
            level = int(level)
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            node = Node(title=_breadcrumb(text), text=text, pages=list(b["pages"]))
            current_heading().children.append(node)
            heading_stack.append((level, node))
            marker_stack, marker_nodes = (
                [],
                [],
            )  # a new heading restarts the marker context
            open_node = node

        elif placed is not None:
            depth, chosen = placed
            del marker_stack[depth:]
            del marker_nodes[depth:]
            parent = marker_nodes[-1] if marker_nodes else current_heading()
            node = Node(title=_breadcrumb(text), text=text, pages=list(b["pages"]))
            parent.children.append(node)
            marker_stack.append((float(b["x0"]), chosen.family, chosen.value))
            marker_nodes.append(node)
            open_node = node

        else:
            # Neither marker nor heading = own content of the currently open node
            open_node.text += ("\n" if open_node.text else "") + text
            for p in b["pages"]:
                if p not in open_node.pages:
                    open_node.pages.append(p)


def _wrap_loose_content(root: Node) -> None:
    """Wraps content piled directly on the root (cover, front matter) in one subtree.

    Left as is, when the root is over budget and split, it would only be copied as
    context and belong to no chunk (= lost). Wrapped, it gets the same standing as the
    other top-level sections.
    """
    if root.text:
        front = Node(title=_breadcrumb(root.text), text=root.text, pages=root.pages)
        root.children.insert(0, front)
        root.text, root.pages = "", []


def spine_block_ids(spine: Spine) -> set[int]:
    """Returns every block id the skeleton points to (merge_wrapped's barriers)."""
    ids = {u.title_id for u in spine}
    ids.update(u.prefix_id for u in spine if u.prefix_id is not None)
    return ids


def _segments(
    blocks: list[Block], spine: Spine
) -> list[tuple[SpineUnit | None, Block | None, list[Block]]]:
    """Cuts the block list at the start of each skeleton unit.

    Each segment is (unit, the unit's title block, the blocks inside the unit). The
    unit's own prefix and title blocks become the node title, so they are left out of
    the inside; the title block is returned separately because the node must inherit
    its page as provenance. Blocks before the first unit come first as (None, None,
    blocks).
    """
    index_of = {b["id"]: i for i, b in enumerate(blocks)}
    # Compare on the first element only. With duplicate title_ids, tuple comparison
    # falls through to the second element and comparing SpineUnits raises TypeError;
    # an uploaded skeleton can have duplicates.
    marks = sorted(
        ((index_of[u.title_id], u) for u in spine if u.title_id in index_of),
        key=lambda m: m[0],
    )

    out: list[tuple[SpineUnit | None, Block | None, list[Block]]] = []
    head_end = marks[0][0] if marks else len(blocks)
    first = marks[0][1] if marks else None
    first_prefix = index_of.get(first.prefix_id) if first is not None else None
    if first_prefix is not None and first_prefix == head_end - 1:
        head_end = first_prefix
    if head_end > 0:
        out.append((None, None, blocks[:head_end]))

    for k, (title_idx, unit) in enumerate(marks):
        if k + 1 >= len(marks):
            end = len(blocks)
        else:
            nxt_idx, nxt = marks[k + 1]
            # The next unit's prefix block belongs to that unit and must not go into
            # this segment, but only when the prefix sits right before the title. A real
            # prefix is always adjacent (frame_lines shows the LLM only heading±1); a
            # distant prefix_id passes validation (it checks prefix<title, not the
            # distance) but would pull end before the segment start and delete that
            # segment's blocks from the tree entirely.
            nxt_prefix = index_of.get(nxt.prefix_id, nxt_idx)
            end = nxt_prefix if nxt_prefix == nxt_idx - 1 else nxt_idx
        out.append((unit, blocks[title_idx], blocks[title_idx + 1 : end]))
    return out


def build_tree(blocks: list[Block], spine: Spine | None = None) -> Node:
    """Assembles a block list into a section tree.

    With a spine, the top-level skeleton comes from it and only the inside of each unit
    is filled from coordinates and markers. Without one, the whole document is
    assembled from coordinates and markers.

    Returns:
        A synthetic root without a title.
    """
    root = Node()
    if not spine:
        _build_interior(root, blocks)
        _wrap_loose_content(root)
        return root

    stack: list[Node] = [root]  # stack[d] = node of depth d-1 unit; stack[0] = root
    for unit, title_block, segment in _segments(blocks, spine):
        if unit is None or title_block is None:
            _build_interior(root, segment)
            _wrap_loose_content(root)
            continue
        # Inherit the title block's pages: the node's text (its title) is its own
        # content and thus one provenance entry, whose page would be an empty list if
        # pages were empty.
        node = Node(
            title=_breadcrumb(unit.title), text=unit.title, pages=list(title_block["pages"])
        )
        parent_idx = min(unit.depth, len(stack) - 1)
        stack[parent_idx].children.append(node)
        del stack[parent_idx + 1 :]
        stack.append(node)
        _build_interior(node, segment)
    return root


def parse_odl_to_tree(odl: dict[str, Any], spine: Spine | None = None) -> Node:
    """Public entry point: assembles ODL JSON into a section tree for MCS chunking."""
    raw = flatten_blocks(odl)
    blocks = merge_wrapped(raw, barriers=spine_block_ids(spine) if spine else None)
    if not blocks:
        raise ValueError("No text blocks found in the ODL output.")
    return build_tree(blocks, spine)
