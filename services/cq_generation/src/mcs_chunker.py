"""MCS (Maximal Complete-Subtree) chunking: splits a section tree into whole subtrees.

chunk(v) = the whole subtree of the shallowest ancestor a of v whose total subtree size
is <= budget. Size increases monotonically going up, so this definition is equivalent to
finding, by top-down recursion from the root, the node with "size(node) <= budget <
size(parent)"; that is why chunk_tree's walk goes from the top down.

The budget is a target, not a cap. A childless node over budget is emitted whole, not
split: not cutting a complete unit takes priority over meeting the budget. Splitting by
paragraph breaks chunks down to single sentences that yield no useful CQs, and it also
breaks the provenance claim "this CQ was derived from this section". A chunk over budget
still stays at a size the LLM can digest in one pass.

A parent's own content (heading line, preamble before the first child) belongs to no
child's section when the parent is split into its children. That content is then
duplicated into every child chunk as context but 'belongs' to none of them (it is not
put in the provenance list); removing duplicate CQs caused by the duplication is a
later task's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


@dataclass
class Node:
    """Node of a section tree; MCS's input contract, independent of the parser."""

    title: str = ""  # heading line; one step of the provenance breadcrumb
    text: str = ""  # own content (incl. heading line): body not belonging to children
    pages: list[int] = field(
        default_factory=list
    )  # pages the own content spans (ascending)
    children: list["Node"] = field(default_factory=list)


class ParagraphRef(BaseModel):
    """One provenance entry; section_path stays a list (not joined into a string)."""

    page: list[int]
    section_path: list[str]


class Chunk(BaseModel):
    index: int
    chars: int
    section_path: list[str]  # path to the chunk root = name of the covered subtree
    context: str = ""  # copy of the ancestors' own content (not owned)
    text: str
    paragraphs: list[ParagraphRef] = Field(default_factory=list)


def subtree_size(node: Node) -> int:
    """Returns the size that is compared with the budget."""
    return len(node.text) + sum(subtree_size(c) for c in node.children)


def _subtree_text(node: Node) -> list[str]:
    out = [node.text] if node.text else []
    for c in node.children:
        out.extend(_subtree_text(c))
    return out


def _subtree_paragraphs(node: Node, path: list[str]) -> list[ParagraphRef]:
    """Flattens every node with own content in the subtree into one provenance entry.

    Nodes with a heading but no body are included too, because they are also sections
    the chunk covers.
    """
    node_path = path + [node.title] if node.title else path
    out: list[ParagraphRef] = []
    if node.text:
        out.append(ParagraphRef(page=list(node.pages), section_path=node_path))
    for c in node.children:
        out.extend(_subtree_paragraphs(c, node_path))
    return out


def _make_chunk(node: Node, ancestors: list[Node], index: int) -> Chunk:
    path = [a.title for a in ancestors if a.title]
    context = "\n".join(a.text for a in ancestors if a.text)
    text = "\n".join(_subtree_text(node))
    return Chunk(
        index=index,
        chars=len(text),
        section_path=path + ([node.title] if node.title else []),
        context=context,
        text=text,
        paragraphs=_subtree_paragraphs(node, path),
    )


def chunk_tree(root: Node, budget: int) -> list[Chunk]:
    """Splits a section tree into MCS chunks.

    The same tree and the same budget always give the same result.
    """
    out: list[Chunk] = []

    def walk(node: Node, ancestors: list[Node]) -> None:
        # Without children it cannot be split even over budget: paragraph integrity
        # takes priority over the budget.
        if subtree_size(node) <= budget or not node.children:
            out.append(_make_chunk(node, ancestors, len(out)))
            return
        # Over budget + has children -> split only into whole child subtrees.
        # This node's own content is copied into the child chunks' context
        # (ancestors in _make_chunk).
        for c in node.children:
            walk(c, ancestors + [node])

    walk(root, [])
    return out
