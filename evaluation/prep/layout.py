"""Text lines with position and weight, rebuilt from a PDF's words.

Every extractor works on lines, not on raw text: where a line starts and
whether it is bold is what tells a glossary term from its definition, and an
index heading from a sub-entry.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber
from pdfplumber.utils.exceptions import PdfminerException

# Running headers and footers sit in these bands on every doctrine page.
HEADER_BAND = 60.0
FOOTER_BAND = 60.0
# A line in the outer bands that is only a publication number, a section
# page number or a page number is a running header or footer; some manuals
# print theirs lower than the fixed bands.
RUNNING_BAND = 110.0
_RUNNING = re.compile(
    r"^((FM|ATP|ADP|ADRP|JP|TC)\s[\d.\-]+(\s\([\d.\-]+\))?"
    r"|(Glossary|Index|References)(-\d+)?|\d{1,3})$"
)
# Words whose tops differ by less than this belong to one line.
LINE_TOLERANCE = 2.5


@dataclass(frozen=True)
class Word:
    """One word as the PDF places it."""

    text: str
    x0: float
    x1: float
    top: float
    bold: bool
    size: float


@dataclass(frozen=True)
class Line:
    """Words sharing a baseline, left to right.

    Attributes:
        page: 1-based PDF page number.
        top: Distance from the top of the page.
        words: The words in reading order.
    """

    page: int
    top: float
    words: tuple[Word, ...]

    @property
    def x0(self) -> float:
        """Where the line starts."""
        return self.words[0].x0

    @property
    def text(self) -> str:
        """The words joined by single spaces."""
        return " ".join(word.text for word in self.words)

    @property
    def starts_bold(self) -> bool:
        """Whether the first word is bold."""
        return self.words[0].bold

    def bold_prefix(self) -> tuple[str, str]:
        """Splits the line into its leading bold words and the rest."""
        cut = 0
        while cut < len(self.words) and self.words[cut].bold:
            cut += 1
        head = " ".join(w.text for w in self.words[:cut])
        tail = " ".join(w.text for w in self.words[cut:])
        return head, tail


def _word(raw: dict[str, Any]) -> Word:
    font = str(raw.get("fontname", ""))
    return Word(
        text=str(raw["text"]),
        x0=float(raw["x0"]),
        x1=float(raw["x1"]),
        top=float(raw["top"]),
        bold="Bold" in font or "Black" in font,
        size=float(raw.get("size", 0.0)),
    )


def group_lines(words: list[Word], page: int) -> list[Line]:
    """Groups words into lines by their top coordinate."""
    lines: list[Line] = []
    current: list[Word] = []
    for word in sorted(words, key=lambda w: (round(w.top, 1), w.x0)):
        if current and abs(word.top - current[0].top) > LINE_TOLERANCE:
            lines.append(
                Line(page, current[0].top, tuple(sorted(current, key=lambda w: w.x0)))
            )
            current = []
        current.append(word)
    if current:
        lines.append(
            Line(page, current[0].top, tuple(sorted(current, key=lambda w: w.x0)))
        )
    return lines


class Document:
    """A PDF opened for line-level reading.

    Attributes:
        path: The source PDF.
        n_pages: Page count.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._pdf = pdfplumber.open(path)
        self.n_pages = len(self._pdf.pages)
        self._cache: dict[int, list[Word]] = {}
        self.unreadable_pages: set[int] = set()

    def close(self) -> None:
        """Closes the PDF."""
        self._pdf.close()

    def __enter__(self) -> "Document":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def words(self, page: int) -> list[Word]:
        """All words of a 1-based page, headers and footers included."""
        if page not in self._cache:
            try:
                raw = self._pdf.pages[page - 1].extract_words(
                    extra_attrs=["fontname", "size"], keep_blank_chars=False
                )
            except PdfminerException:
                # A malformed page (one front-matter page of JP 3-0) reads as
                # empty; the report lists it so the gap is visible.
                self.unreadable_pages.add(page)
                raw = []
            self._cache[page] = [_word(w) for w in raw]
        return self._cache[page]

    def page_height(self, page: int) -> float:
        """Height of a 1-based page."""
        return float(self._pdf.pages[page - 1].height)

    def page_width(self, page: int) -> float:
        """Width of a 1-based page."""
        return float(self._pdf.pages[page - 1].width)

    def head_text(self, page: int, band: float = 130.0) -> str:
        """Text in the top band of a page, where headings and headers sit."""
        return " ".join(w.text for w in self.words(page) if w.top < band)

    def body_lines(self, page: int) -> list[Line]:
        """The page's lines without its running header and footer."""
        height = self.page_height(page)
        bottom = height - FOOTER_BAND
        kept = [w for w in self.words(page) if HEADER_BAND < w.top < bottom]
        return [
            line
            for line in group_lines(kept, page)
            if not (
                (line.top < RUNNING_BAND or line.top > height - RUNNING_BAND)
                and _RUNNING.match(line.text)
            )
        ]

    def lines(self, pages: range) -> Iterator[Line]:
        """Body lines of several pages, in order."""
        for page in pages:
            yield from self.body_lines(page)
