"""The scorer assets a manual carries: its glossary and its index.

These CSV files are the gold standard of the coverage metrics, so their shape
is a contract: the extraction in ``evaluation prep`` must produce exactly
these columns for every manual. The files are written with a UTF-8 byte order
mark, as spreadsheet tools expect, and read back without it.
"""

import csv
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

GLOSSARY_COLUMNS: tuple[str, ...] = (
    "term",
    "definition",
    "source",
    "raw_text",
    "pdf_page",
    "entry_type",
)
INDEX_COLUMNS: tuple[str, ...] = (
    "term",
    "sub_entry",
    "level",
    "reference",
    "raw_line",
    "pdf_page",
)


class AssetFormatError(ValueError):
    """A glossary or index file does not follow the column contract."""


class GlossaryEntry(BaseModel):
    """One glossary line.

    Attributes:
        term: The defined term, or the abbreviation.
        definition: The definition, or the expansion of an abbreviation.
        source: Publication the definition is taken from; empty when the
            manual defines the term itself.
        pdf_page: Page of the PDF the entry is printed on.
        entry_type: ``term`` for a definition, ``abbreviation`` for an
            acronym and its expansion.
    """

    term: str
    definition: str
    source: str
    pdf_page: int
    entry_type: Literal["term", "abbreviation"]


class IndexEntry(BaseModel):
    """One index line.

    Attributes:
        term: The heading the line belongs to.
        sub_entry: The sub-entry text; empty on a heading line.
        level: 1 for a heading, 2 for a sub-entry.
        reference: Paragraph numbers or a ``See ...`` cross-reference.
        pdf_page: Page of the PDF the line is printed on.
    """

    term: str
    sub_entry: str
    level: Literal[1, 2]
    reference: str
    pdf_page: int


def _rows(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != columns:
            raise AssetFormatError(
                f"{path.name}: columns {reader.fieldnames} != {list(columns)}"
            )
        return list(reader)


def load_glossary(path: Path) -> list[GlossaryEntry]:
    """Reads a manual's glossary.

    Args:
        path: ``corpus/<doc>/glossary.csv``.

    Returns:
        The entries in file order.

    Raises:
        AssetFormatError: If the header differs from ``GLOSSARY_COLUMNS``.
        pydantic.ValidationError: If a row has an unknown type or a bad page.
    """
    return [GlossaryEntry.model_validate(row) for row in _rows(path, GLOSSARY_COLUMNS)]


def load_index(path: Path) -> list[IndexEntry]:
    """Reads a manual's index.

    Args:
        path: ``corpus/<doc>/index.csv``.

    Returns:
        The lines in file order.

    Raises:
        AssetFormatError: If the header differs from ``INDEX_COLUMNS``.
        pydantic.ValidationError: If a row has an unknown level or a bad page.
    """
    # A literal type does not coerce, and every CSV value is a string.
    return [
        IndexEntry.model_validate({**row, "level": _int_or_raw(row["level"])})
        for row in _rows(path, INDEX_COLUMNS)
    ]


def _int_or_raw(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def index_headings(entries: list[IndexEntry]) -> list[str]:
    """Returns the distinct level-1 headings, the denominator of index coverage."""
    seen: dict[str, None] = {}
    for entry in entries:
        if entry.level == 1:
            seen.setdefault(entry.term, None)
    return list(seen)
