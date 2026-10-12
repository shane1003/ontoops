"""Preparing one manual end to end: verify, locate, cut, extract, report.

Outputs for a manual with slug ``S``:

* ``corpus/prepared/S.pdf``: the body pages only, what the services
  see (not tracked).
* ``corpus/S/glossary.csv`` and ``index.csv``: the scorer
  assets in the contract format, when the manual has them.
* ``corpus/S/report.json``: page ranges, counts, the pages a
  parser could not read, the entries deliberately left out, and the hashes of
  the source and the prepared PDF.

Assets marked ``hand_checked`` in ``SOURCES.yaml`` were verified by a person
against the rendered pages and are never overwritten; the report only gains
the machine-readable page ranges.
"""

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pypdf import PdfReader, PdfWriter

from evaluation.prep.glossary import extract_glossary
from evaluation.prep.index import extract_index
from evaluation.prep.layout import Document
from evaluation.prep.sections import Sections, find_sections
from evaluation.scoring.assets import (
    GLOSSARY_COLUMNS,
    INDEX_COLUMNS,
    GlossaryEntry,
    IndexEntry,
)

CORPUS = Path("corpus")


class SourceDocument(BaseModel):
    """One manual of the corpus, as ``SOURCES.yaml`` lists it."""

    slug: str
    file: str
    title: str
    edition: str
    sha256: str
    url: str | None = None
    index_run_in: bool = False
    hand_checked: bool = False


class Sources(BaseModel):
    """Contents of ``SOURCES.yaml``."""

    documents: list[SourceDocument]

    def get(self, slug: str) -> SourceDocument:
        """The manual with this slug.

        Raises:
            KeyError: If no manual has it.
        """
        for doc in self.documents:
            if doc.slug == slug:
                return doc
        raise KeyError(slug)


class HashMismatchError(RuntimeError):
    """The file on disk is not the edition the corpus names."""


def load_sources(path: Path) -> Sources:
    """Reads ``SOURCES.yaml``."""
    return Sources.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def sha256(data: bytes) -> str:
    """Hex SHA-256 of some bytes."""
    return hashlib.sha256(data).hexdigest()


def cut_pages(pdf: bytes, pages: tuple[int, int]) -> bytes:
    """Keeps the 1-based inclusive page range of a PDF."""
    reader = PdfReader(io.BytesIO(pdf))
    writer = PdfWriter()
    for number in range(pages[0], pages[1] + 1):
        writer.add_page(reader.pages[number - 1])
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _write_glossary(path: Path, entries: list[GlossaryEntry]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(GLOSSARY_COLUMNS)
        for e in entries:
            raw = " ".join(
                part
                for part in (e.term, e.definition, f"({e.source})" if e.source else "")
                if part
            )
            writer.writerow(
                [e.term, e.definition, e.source, raw, e.pdf_page, e.entry_type]
            )


def _write_index(path: Path, entries: list[IndexEntry]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(INDEX_COLUMNS)
        for e in entries:
            text = e.sub_entry if e.level == 2 else e.term
            raw = ", ".join(part for part in (text, e.reference) if part)
            writer.writerow(
                [e.term, e.sub_entry, e.level, e.reference, raw, e.pdf_page]
            )


def prepare(source: SourceDocument, root: Path = CORPUS) -> dict[str, Any]:
    """Prepares one manual; returns the report written next to its assets.

    Raises:
        FileNotFoundError: If the PDF is not under ``root/pdf``.
        HashMismatchError: If the PDF is not the edition ``SOURCES.yaml`` names.
    """
    pdf_path = root / "pdf" / source.file
    pdf = pdf_path.read_bytes()
    digest = sha256(pdf)
    if digest != source.sha256:
        raise HashMismatchError(
            f"{source.slug}: {pdf_path.name} has {digest}, expected {source.sha256}"
        )

    out = root / source.slug
    out.mkdir(parents=True, exist_ok=True)
    with Document(pdf_path) as doc:
        sections: Sections = find_sections(doc)
        glossary = (
            extract_glossary(doc, sections.glossary) if sections.glossary else None
        )
        index = (
            extract_index(doc, sections.index, run_in_first_sub=source.index_run_in)
            if sections.index
            else None
        )
        unreadable = sorted(doc.unreadable_pages)

    prepared = cut_pages(pdf, sections.body)
    prepared_dir = root / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    (prepared_dir / f"{source.slug}.pdf").write_bytes(prepared)

    report_path = out / "report.json"
    report: dict[str, Any] = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    extracted: dict[str, Any] = {
        "pages": sections.as_dict(),
        "unreadable_pages": unreadable,
        "source_sha256": digest,
        "prepared_sha256": sha256(prepared),
        "prepared_pages": sections.body[1] - sections.body[0] + 1,
        "hand_checked": source.hand_checked,
    }
    if glossary is not None:
        extracted["glossary_layout"] = glossary.layout
        extracted["glossary_removed_terms"] = glossary.removed
        extracted["glossary_entries"] = len(glossary.entries)
    if index is not None:
        extracted["index_rows"] = len(index)
    if not source.hand_checked:
        if glossary is not None:
            _write_glossary(out / "glossary.csv", glossary.entries)
        if index is not None:
            _write_index(out / "index.csv", index)
    report.update(extracted)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
