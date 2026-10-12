"""OOPS! pitfall scan via the REST API.

Sends a full ``<OOPSRequest>`` envelope (a bare CDATA fragment is rejected) with the TBox
as RDF/XML, and parses pitfalls graded by severity. Network-optional: any request failure
sets ``available=False`` and returns no pitfalls rather than raising — OOPS! never blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.sax.saxutils import escape

from rdflib import Graph, Namespace

OOPS_NS = Namespace("http://www.oeg-upm.net/oops#")
SEVERITIES = ("Critical", "Important", "Minor")


@dataclass(frozen=True)
class Pitfall:
    code: str
    name: str
    severity: str
    num_affected: int | None = None


@dataclass
class OopsResult:
    available: bool
    pitfalls: list[Pitfall] = field(default_factory=list)
    error: str | None = None

    def by_severity(self, severity: str) -> list[Pitfall]:
        return [p for p in self.pitfalls if p.severity == severity]


def build_request(
    rdfxml: str, *, pitfalls: str = "", output_format: str = "RDF/XML"
) -> str:
    """Build the OOPS! request envelope. ``rdfxml`` goes inline in a CDATA block."""
    # CDATA cannot contain "]]>"; defensively split any occurrence.
    safe = rdfxml.replace("]]>", "]]]]><![CDATA[>")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<OOPSRequest>\n"
        "  <OntologyURI></OntologyURI>\n"
        f"  <OntologyContent><![CDATA[{safe}]]></OntologyContent>\n"
        f"  <Pitfalls>{escape(pitfalls)}</Pitfalls>\n"
        f"  <OutputFormat>{escape(output_format)}</OutputFormat>\n"
        "</OOPSRequest>"
    )


def parse_response(rdfxml_response: str) -> list[Pitfall]:
    """Parse the OOPS! RDF/XML response into pitfalls."""
    graph = Graph().parse(data=rdfxml_response, format="xml")
    pitfalls: list[Pitfall] = []
    for subject in set(graph.subjects(OOPS_NS.hasCode, None)):
        code = str(graph.value(subject, OOPS_NS.hasCode) or "").strip()
        name = str(graph.value(subject, OOPS_NS.hasName) or "").strip()
        severity = (
            str(graph.value(subject, OOPS_NS.hasImportanceLevel) or "").strip()
            or "Minor"
        )
        num = graph.value(subject, OOPS_NS.hasNumberAffectedElements)
        pitfalls.append(
            Pitfall(
                code=code,
                # Kept verbatim, including a level outside SEVERITIES. Coercing an unrecognised
                # grade into a known one would report a severity OOPS! never said; the summary
                # side buckets whatever it does not recognise instead, so the unknown stays
                # visible rather than being quietly renamed here.
                name=name,
                severity=severity,
                num_affected=int(num) if num is not None else None,
            )
        )
    return sorted(pitfalls, key=lambda p: p.code)


def scan_oops(
    rdfxml: str,
    *,
    endpoint: str,
    session=None,
    timeout: int = 120,
) -> OopsResult:
    """POST the ontology to OOPS! and parse the pitfall report. Never raises on network error."""
    if session is None:
        import requests

        session = requests.Session()

    body = build_request(rdfxml).encode("utf-8")
    try:
        response = session.post(
            endpoint,
            data=body,
            headers={"Content-Type": "application/xml"},
            timeout=timeout,
        )
        response.raise_for_status()
        pitfalls = parse_response(response.text)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully; OOPS is advisory
        return OopsResult(available=False, error=f"{type(exc).__name__}: {exc}")
    return OopsResult(available=True, pitfalls=pitfalls)
