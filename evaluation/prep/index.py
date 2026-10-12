"""Index extraction for the three-column Army layout.

The index is set in three columns, so words are split into columns before
they are grouped into lines. Inside a column the indent carries the
structure (measured on ATP 2-01.3): a heading starts flush
left and wraps with a hanging indent of about 10 points; a sub-entry starts
about 13 points in and wraps at about 23. A heading carried over from the
previous column is repeated as "heading (continued)".
"""

import re
from dataclasses import dataclass

from evaluation.prep.layout import (
    FOOTER_BAND,
    HEADER_BAND,
    Document,
    Line,
    Word,
    group_lines,
)
from evaluation.scoring.assets import IndexEntry

COLUMNS = 3
_CONTINUED = re.compile(r"^(.*?)\s*\(\s*(continued|cont\.?)\s*\)\s*$", re.IGNORECASE)
_LETTER = re.compile(r"^[A-Z](-[A-Z])*$")
_REF = r"(?:[A-Z]{1,2}-\d+|\d+-\d+)(?:\s?[–—-]\s?(?:[A-Z]{1,2}-\d+|\d+-\d+))?"
_REFS = re.compile(rf"^(.*?),\s*({_REF}(?:,\s*{_REF})*)\s*\.?$")
# Non-breaking and figure hyphens are hyphens; an em dash inside a
# reference range is an en dash.
_DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2212": "-"})
# When the regular pattern fails (a comma missing between two references,
# "1-543-1"), everything after the first ", <digit>" is the reference.
_LOOSE_REFS = re.compile(r"^(.*?),\s*(\d[\d\-–—,\s]*)$")
_DASH_SPACE = re.compile(r"\s*–\s*")
# ADP 3-90 prints where a term is defined on the heading line: "term, defined".
_DEFINED = re.compile(r"^(.*?),\s*(defined)$")
_SEE = re.compile(r"^(.*?)\.?\s+(See(?: also)?\s.*)$")
COLUMN_SLACK = 4.0
SIZE_TOLERANCE = 0.3
MIN_COLUMN_WIDTH = 120.0
LINE_START_GAP = 30.0
LINE_GAP = 14.0
# Indent thresholds in points, from the column's left edge.
HEADING_WRAP = 8.0
SUB_START = 11.5
SUB_WRAP = 19.0


@dataclass
class _Entry:
    text: str
    level: int
    page: int


def _line_starts(words: list[Word]) -> list[float]:
    """x positions where lines begin: a word with no close left neighbour."""
    starts: dict[int, int] = {}
    for word in words:
        before = [
            o.x1 for o in words if abs(o.top - word.top) < 2.5 and o.x1 <= word.x0
        ]
        if not before or word.x0 - max(before) > LINE_START_GAP:
            key = round(word.x0)
            starts[key] = starts.get(key, 0) + 1
    return sorted(float(x) for x in starts)


def _column_edges(words: list[Word]) -> list[float]:
    """Left edge of each column, measured on the page.

    The pitch differs between manuals (162 points in ATP 2-01.3 and ADP 3-90,
    156 in ADRP 3-0), so it is not a constant.
    """
    starts = _line_starts(words)
    edges = [starts[0]] if starts else [min(w.x0 for w in words)]
    for x in starts:
        if len(edges) < COLUMNS and x > edges[-1] + MIN_COLUMN_WIDTH:
            edges.append(x)
    return edges


def _columns(doc: Document, page: int) -> list[list[Line]]:
    height = doc.page_height(page)
    inside = [w for w in doc.words(page) if HEADER_BAND < w.top < height - FOOTER_BAND]
    if not inside:
        return []
    # Entries are set in one size; the letter headings ("A", "X-Y-Z") and the
    # title are larger and would otherwise merge into the entry on their line.
    sizes: dict[float, int] = {}
    for w in inside:
        sizes[round(w.size, 1)] = sizes.get(round(w.size, 1), 0) + 1
    body = max(sizes, key=lambda size: sizes[size])
    words = [w for w in inside if w.size <= body + SIZE_TOLERANCE]
    # The note under the title ("Entries are by paragraph number ...") runs
    # across all columns; it goes before the columns are cut.
    notes = [w.top for w in words if w.text == "Entries"]
    if notes:
        words = [w for w in words if w.top > max(notes) + LINE_GAP]
    if not words:
        return []
    edges = _column_edges(words)
    buckets: list[list[Word]] = [[] for _ in edges]
    for word in words:
        # A word that starts a hair left of a column edge still belongs to it.
        slot = (
            max(i for i, edge in enumerate(edges) if word.x0 >= edge - COLUMN_SLACK)
            if word.x0 >= edges[0] - COLUMN_SLACK
            else 0
        )
        buckets[slot].append(word)
    return [group_lines(bucket, page) for bucket in buckets]


def _join(before: str, after: str) -> str:
    # A line broken after a dash (an em dash in a heading, a hyphen or en
    # dash inside a reference such as "3-42–3-" / "55") continues without
    # a space.
    return (
        f"{before}{after}" if before.endswith(("—", "–", "-")) else f"{before} {after}"
    )


def _parse(text: str) -> tuple[str, str]:
    """Splits an entry into its words and its reference."""
    # Text layers mix dash characters and leave stray soft-hyphen marks.
    text = text.replace("¬", "").translate(_DASHES)
    text = _DASH_SPACE.sub("–", text)
    see = _SEE.match(text)
    if see:
        return see.group(1).rstrip(".,"), see.group(2)
    refs = _REFS.match(text) or _LOOSE_REFS.match(text)
    if refs:
        return refs.group(1), refs.group(2)
    return text.rstrip(","), ""


def _entries(doc: Document, pages: tuple[int, int]) -> list[_Entry]:
    entries: list[_Entry] = []
    for page in range(pages[0], pages[1] + 1):
        for column in _columns(doc, page):
            if not column:
                continue
            edge = min(line.x0 for line in column)
            for line in column:
                text = line.text
                if line.starts_bold and _LETTER.match(text):
                    continue  # the letter heading of a group
                offset = line.x0 - edge
                if offset < HEADING_WRAP:
                    entries.append(_Entry(text, 1, page))
                elif offset < SUB_START and entries:
                    entries[-1].text = _join(entries[-1].text, text)
                elif offset < SUB_WRAP or not entries:
                    entries.append(_Entry(text, 2, page))
                else:
                    entries[-1].text = _join(entries[-1].text, text)
    return entries


def extract_index(
    doc: Document, pages: tuple[int, int], *, run_in_first_sub: bool = False
) -> list[IndexEntry]:
    """Reads a manual's index.

    Args:
        doc: The opened manual.
        pages: The index's 1-based inclusive page range.
        run_in_first_sub: The manual prints a heading's first sub-entry on the
            heading line ("heading, first sub-entry, 6-80"), as some manuals
            do. The split happens only when the heading line ends in a
            reference and sub-entries follow, so a heading whose own name
            has commas stays whole.

    Returns:
        Headings (level 1) and sub-entries (level 2) in reading order; a
        sub-entry names its heading in ``term``.
    """
    rows: list[IndexEntry] = []
    heading = ""
    entries = _entries(doc, pages)
    for i, entry in enumerate(entries):
        carried = _CONTINUED.match(entry.text) if entry.level == 1 else None
        if carried:
            heading = carried.group(1)  # carried over from the previous column
            continue
        words, reference = _parse(entry.text)
        if entry.level == 1:
            followed = i + 1 < len(entries) and entries[i + 1].level == 2
            first_sub = ""
            defined = _DEFINED.match(words)
            if defined:
                words, first_sub = defined.group(1), defined.group(2)
            elif run_in_first_sub and reference and followed and ", " in words:
                words, first_sub = words.split(", ", 1)
            if not words:
                continue
            heading = words
            rows.append(
                IndexEntry(
                    term=words,
                    sub_entry="",
                    level=1,
                    reference="" if first_sub else reference,
                    pdf_page=entry.page,
                )
            )
            if first_sub:
                rows.append(
                    IndexEntry(
                        term=words,
                        sub_entry=first_sub,
                        level=2,
                        reference=reference,
                        pdf_page=entry.page,
                    )
                )
        else:
            sub = "" if words.startswith("See also") else words
            if not sub and not reference:
                reference = entry.text
            rows.append(
                IndexEntry(
                    term=heading,
                    sub_entry=sub,
                    level=2,
                    reference=reference,
                    pdf_page=entry.page,
                )
            )
    return rows
