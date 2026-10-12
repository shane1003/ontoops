"""Glossary extraction for the two layouts doctrine manuals use.

Army layout (ATP, ADP, ADRP, FM): acronyms in bold with the expansion in a
fixed column; terms as a bold heading line with the definition indented
underneath and the proponent publication in parentheses at its end.

Joint layout (JP): acronyms and expansions on one line in the same weight;
each term in bold at the start of a paragraph with its definition following
on the same line, and the source as ``(DOD Dictionary. Source: JP 3-0)``.

Entries whose definition is only "None." (terms the dictionary approved for
removal) are not terms of the manual and are left out, with a count.
"""

import re
from dataclasses import dataclass, field

from evaluation.prep.layout import Document, Line
from evaluation.scoring.assets import GlossaryEntry

_ACRONYMS = re.compile(r"^\s*(SECTION|PART)\s+I\b(?!I)")
_TERMS = re.compile(r"^\s*(SECTION|PART)\s+II\b")
_SECTION_HEADING = re.compile(r"^\s*(SECTION|PART)\s+I{1,3}\b")
# The proponent at the end of an Army definition: "(ADP 3-90)", "(JP 3-09.3)".
# Any publication short title followed by its number, e.g. "CJCSM 5120.01A".
_ARMY_SOURCE = re.compile(r"\(([A-Z][A-Z0-9]{1,7}\s\d[^()]*)\)\.?\s*$")
# The joint form: "(DOD Dictionary. Source: JP 3-0)", "(JP 1-02. SOURCE: JP 2-01.3)".
_JOINT_SOURCE = re.compile(r"\(([^()]*?)SOURCE:\s*([^()]*)\)\s*$", re.IGNORECASE)
_EDITORIAL = re.compile(r"\((Approved for|Upon approval)[^()]*\)\s*$", re.IGNORECASE)
_QUALIFIER = re.compile(r"^\((DOD|Army|Joint|NATO|DOD/NATO)\)")
_REMOVED = re.compile(r"^None\.\s*(\(.*\))?$")
_NUMBERED = re.compile(r"(?<![\w.])(\d{1,2})\.(?=[A-Z])")
_FIRST_SENSE = re.compile(r"^(.*?)\.\s+(\d{1,2}\.)$")
_GAP = 8.0  # points of white space between an acronym and its expansion


@dataclass
class GlossaryResult:
    """Entries plus what was deliberately not kept.

    Attributes:
        entries: The glossary in reading order.
        layout: ``army`` or ``joint``.
        removed: Terms listed only as approved for removal.
    """

    entries: list[GlossaryEntry]
    layout: str
    removed: list[str] = field(default_factory=list)


def _split_acronym(line: Line) -> tuple[str, str] | None:
    # Bold acronym then regular expansion (Army); otherwise the first wide gap.
    head, tail = line.bold_prefix()
    if head and tail:
        return head, tail
    words = line.words
    for i in range(1, len(words)):
        if words[i].x0 - words[i - 1].x1 > _GAP:
            left = " ".join(w.text for w in words[:i])
            right = " ".join(w.text for w in words[i:])
            return left, right
    return None


def _acronyms(lines: list[Line]) -> list[GlossaryEntry]:
    entries: list[GlossaryEntry] = []
    column: float | None = None
    for line in lines:
        split = _split_acronym(line)
        starts_entry = split is not None and (column is None or line.x0 < column - 2)
        if starts_entry and split:
            term, expansion = split
            column = next(w.x0 for w in line.words if w.text == expansion.split()[0])
            entries.append(
                GlossaryEntry(
                    term=term,
                    definition=expansion,
                    source="",
                    pdf_page=line.page,
                    entry_type="abbreviation",
                )
            )
        elif entries:
            # A wrapped expansion continues in the expansion column.
            last = entries[-1]
            entries[-1] = last.model_copy(
                update={"definition": f"{last.definition} {line.text}"}
            )
    return entries


def _finish(term: str, definition: str, page: int, source_re: str) -> GlossaryEntry:
    # The text layer drops the space after a numbered sense ("1.The").
    definition = _NUMBERED.sub(r"\1. ", definition.strip())
    source = ""
    if source_re == "joint":
        # A term this manual adds carries an editorial note instead of a
        # source; like an Army proponent term, it gets no source.
        definition = _EDITORIAL.sub("", definition).strip()
        match = _JOINT_SOURCE.search(definition)
        if match:
            source = match.group(2).strip()
            definition = definition[: match.start()].strip()
    else:
        match = _ARMY_SOURCE.search(definition)
        if match:
            source = match.group(1).strip()
            definition = definition[: match.start()].strip()
    return GlossaryEntry(
        term=term.strip(),
        definition=definition,
        source=source,
        pdf_page=page,
        entry_type="term",
    )


def _page_left(lines: list[Line]) -> dict[int, float]:
    """The flush-left edge of term headings on each page.

    Facing pages of a printed manual have different margins, so the edge is
    per page, not per document.
    """
    left: dict[int, float] = {}
    for line in lines:
        if line.starts_bold:
            left[line.page] = min(left.get(line.page, line.x0), line.x0)
    return left


def _army_terms(lines: list[Line]) -> list[GlossaryEntry]:
    # A term is a bold line flush left; its definition is the indented lines.
    left = _page_left(lines)
    entries: list[GlossaryEntry] = []
    term, page, parts = "", 0, list[str]()
    for line in lines:
        head, tail = line.bold_prefix()
        heading = (
            line.starts_bold
            and abs(line.x0 - left.get(line.page, line.x0)) < 3
            # "(DOD)" or "(Army)" may follow a term on its heading line.
            and (not tail or _QUALIFIER.match(tail) is not None)
        )
        if heading:
            if term and not parts:
                # A term that wraps continues on the next bold line.
                term = f"{term} {head}"
                if tail:
                    parts.append(tail)
                continue
            if term:
                entries.append(_finish(term, " ".join(parts), page, "army"))
            term, page, parts = head, line.page, [tail] if tail else []
        elif term:
            parts.append(line.text)
    if term and parts:
        entries.append(_finish(term, " ".join(parts), page, "army"))
    return entries


def _joint_terms(lines: list[Line]) -> tuple[list[GlossaryEntry], list[str]]:
    # Continuation lines are indented, so a flush-left line opens a term. The
    # position is the signal, not the weight: some text layers
    # lose the bold on term headings.
    left: dict[int, float] = {}
    for line in lines:
        if not _SECTION_HEADING.match(line.text):
            left[line.page] = min(left.get(line.page, line.x0), line.x0)
    entries: list[GlossaryEntry] = []
    removed: list[str] = []
    term, page, parts = "", 0, list[str]()

    def flush() -> None:
        if not term:
            return
        definition = " ".join(parts).strip()
        if _REMOVED.match(definition):
            removed.append(term)
            return
        entries.append(_finish(term, definition, page, "joint"))

    for line in lines:
        if _SECTION_HEADING.match(line.text):
            continue
        if abs(line.x0 - left[line.page]) < 3:
            head, tail = line.bold_prefix()
            if not head.endswith("."):
                head, _, tail = line.text.partition(". ")
                head += "."
            sense = _FIRST_SENSE.match(head)
            if sense:
                # The bold run can reach into the first sense number: "objective. 1."
                head, tail = sense.group(1) + ".", f"{sense.group(2)} {tail}"
            flush()
            term, page, parts = head[:-1], line.page, [tail]
        elif term:
            parts.append(line.text)
    flush()
    return entries, removed


def extract_glossary(doc: Document, pages: tuple[int, int]) -> GlossaryResult:
    """Reads a manual's glossary.

    Args:
        doc: The opened manual.
        pages: The glossary's 1-based inclusive page range.

    Returns:
        Acronyms first, then terms, each in reading order.
    """
    lines = list(doc.lines(range(pages[0], pages[1] + 1)))
    layout = (
        "joint"
        if any(line.text.lstrip().startswith("PART I") for line in lines)
        else "army"
    )
    acro_at = next(
        (i for i, line in enumerate(lines) if _ACRONYMS.match(line.text)), None
    )
    terms_at = next(
        (i for i, line in enumerate(lines) if _TERMS.match(line.text)), None
    )
    if acro_at is None and terms_at is None:
        # Older manuals list acronyms only, without section titles.
        return GlossaryResult(_acronyms(_after_title(lines)), layout)
    acro_lines = lines[
        (acro_at or 0) + 1 : terms_at if terms_at is not None else len(lines)
    ]
    term_lines = lines[terms_at + 1 :] if terms_at is not None else []
    acronyms = _acronyms(
        [ln for ln in acro_lines if not _SECTION_HEADING.match(ln.text)]
    )
    if layout == "joint":
        terms, removed = _joint_terms(term_lines)
    else:
        terms, removed = _army_terms(term_lines), []
    return GlossaryResult(acronyms + terms, layout, removed)


def _after_title(lines: list[Line]) -> list[Line]:
    # Drop the page title ("Glossary") and anything before the first entry.
    return [line for line in lines if not re.match(r"^\s*Glossary\b", line.text, re.I)]
