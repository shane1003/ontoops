"""Where a manual's contents, body, glossary and index are.

The body is what the services may read; everything else is either front
matter or a scorer asset. So the boundaries are the firewall, and they are
found from the page headings the manuals print, then written down in
``report.json`` for a person to check.
"""

import re
from dataclasses import dataclass

from evaluation.prep.layout import Document

# A contents line ends in a leader or a page reference ("1-1", "II-3", "A-2",
# "vi"); some text layers drop the dots but keep the reference.
_CONTENTS_LINE = re.compile(
    r"(\.{6,}|(\. ){4,})|\b([IVXL]+|[A-Z]{1,2}|\d{1,2})\s?-\s?\d{1,3}$|\b[ivx]{1,5}$"
)
_GLOSSARY = re.compile(r"^\s*(Glossary|GLOSSARY)\b")
_INDEX = re.compile(r"^\s*(Index|INDEX)\b")
_BACK_MATTER = re.compile(
    r"^\s*(Glossary|GLOSSARY|Index|INDEX|References|REFERENCES|Bibliography|BIBLIOGRAPHY)\b"
)
# A second glossary part may open a page without the running header.
_PART = re.compile(r"^\s*(PART|SECTION)\s+I{1,3}\b")
# Back matter starts in the last part of a manual; a heading earlier on is a
# contents entry or a cross-reference.
BACK_MATTER_FROM = 0.6
FRONT_MATTER_UNTIL = 0.2
CONTENTS_MIN_LINES = 5


@dataclass(frozen=True)
class Sections:
    """1-based inclusive page ranges; None when the manual lacks the part.

    Attributes:
        contents: The table of contents and the lists of figures and tables.
        body: What the services see.
        glossary: The glossary.
        index: The index.
    """

    contents: tuple[int, int] | None
    body: tuple[int, int]
    glossary: tuple[int, int] | None
    index: tuple[int, int] | None

    def as_dict(self) -> dict[str, list[int] | None]:
        """Plain form for ``report.json``."""
        return {
            name: list(value) if value else None
            for name, value in (
                ("contents", self.contents),
                ("body", self.body),
                ("glossary", self.glossary),
                ("index", self.index),
            )
        }


_BLANK = re.compile(r"intentionally\s+(left\s+)?blank", re.IGNORECASE)


def is_blank(doc: Document, page: int) -> bool:
    """Whether a page carries nothing but a "left blank" notice."""
    lines = doc.body_lines(page)
    return not lines or all(_BLANK.search(line.text) for line in lines)


def _first_line(doc: Document, page: int) -> str:
    lines = doc.body_lines(page)
    return lines[0].text if lines else ""


def _running(doc: Document, page: int, word: str) -> bool:
    # Continuation pages carry the section name in the running header.
    return word.lower() in doc.head_text(page, band=60.0).lower()


def _back_matter_starts(doc: Document) -> list[int]:
    first = int(doc.n_pages * BACK_MATTER_FROM)
    return [
        p
        for p in range(first, doc.n_pages + 1)
        if _BACK_MATTER.match(_first_line(doc, p))
    ]


def _span(
    doc: Document, pattern: re.Pattern[str], word: str, starts: list[int]
) -> tuple[int, int] | None:
    start = next((p for p in starts if pattern.match(_first_line(doc, p))), None)
    if start is None:
        return None
    later = [p for p in starts if p > start]
    if later:
        end = later[0] - 1
        # Blank verso pages before the next section are not part of this one.
        while end > start and is_blank(doc, end):
            end -= 1
        return start, end
    end = start
    while end + 1 <= doc.n_pages and (
        _running(doc, end + 1, word) or _PART.match(_first_line(doc, end + 1))
    ):
        end += 1
    return start, end


def _contents(doc: Document) -> tuple[int, int] | None:
    last = int(doc.n_pages * FRONT_MATTER_UNTIL) or 1
    pages = [
        p
        for p in range(1, last + 1)
        if sum(bool(_CONTENTS_LINE.search(line.text)) for line in doc.body_lines(p))
        >= CONTENTS_MIN_LINES
    ]
    if not pages:
        return None
    # The contents is one run of pages; stop at the first gap.
    end = pages[0]
    while end + 1 in pages:
        end += 1
    return pages[0], end


def find_sections(doc: Document) -> Sections:
    """Locates the parts of a manual.

    The body runs from the page after the contents to the page before the
    first back-matter section (glossary, references, bibliography or index).
    """
    contents = _contents(doc)
    starts = _back_matter_starts(doc)
    glossary = _span(doc, _GLOSSARY, "glossary", starts)
    index = _span(doc, _INDEX, "index", starts)
    body_end = (starts[0] - 1) if starts else doc.n_pages
    body_start = contents[1] + 1 if contents else 1
    return Sections(contents, (body_start, body_end), glossary, index)
