"""Derivation of the FalkorDB graph name from a term IRI."""

import re

_IRI = re.compile(r"^https?://[^/]+/(?P<graph>.+)#(?P<local>[^#]*)$")


class UnroutableIriError(ValueError):
    """Raised when an IRI does not carry a derivable graph name."""


def graph_id_from_iri(iri: str) -> str:
    """Derive the FalkorDB graph name from a term IRI.

    ``http://example.org/military/<document>#<Local>`` maps to the graph
    ``military/<document>``: everything between the host and the fragment.

    Args:
        iri: The full term IRI, which is also ``tbox_term.id``.

    Returns:
        The FalkorDB graph name.

    Raises:
        UnroutableIriError: If the IRI has no host-rooted path or no fragment.
            The upstream ``rdb_*`` source tables are empty in v1; if non-document
            sources ever appear, this raise is where that shows up.
    """
    match = _IRI.match(iri)
    if match is None:
        raise UnroutableIriError(f"IRI carries no derivable graph name: {iri!r}")
    return match.group("graph")
