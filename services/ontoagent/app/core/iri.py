"""Single point of base-IRI validation (shared by tboxcraft and aboxmapper).

A document's base IRI passes through both stages of the pipeline: tboxcraft mints the TBox
term IRIs under it, and aboxmapper mints individual IRIs in the same namespace. If the
rule differed per module, a namespace that passed one side could be rejected by the other,
a failure that only surfaces after the TBox IRIs were already issued; so the rule lives
here, in one place only.

Rejects, never repairs: a minted IRI is a stored, permanent identifier. A silently
"fixed" base becomes a namespace nobody ever asked for, and the 800 terms made with it are
hard to undo. A rejected request can be fixed.
"""

from __future__ import annotations

import os
import re
from urllib.parse import quote, urlsplit

from dotenv import load_dotenv

# Characters that cannot be in an IRI. Backtick/whitespace/control characters: every one
# a corruption upstream has actually sent. A trailing '#' or '/' is explicitly allowed as
# a namespace separator; it is not corruption, so it is not stripped. '%20' is a legal IRI
# sequence and is not treated as corruption.
_FORBIDDEN = re.compile(r"[`\s\x00-\x1f\x7f]")


class InvalidBaseIri(ValueError):
    """The given base IRI is malformed. Rejected, never silently repaired."""


def validate_base_iri(base: str) -> str:
    """Return ``base`` unchanged if it is a usable namespace IRI, else raise.

    Deliberately **not** normalising: a base that arrives with a backtick, a space, an
    empty path segment (``//``) or a stray ``%20`` is a sign the upstream is confused, and
    quietly repairing it hides that while minting IRIs nobody can reproduce. A trailing
    ``#`` or ``/`` is a separator, not an error — the minters read it.

    Args:
        base: The namespace IRI a document's terms/individuals are minted under.

    Returns:
        ``base``, unchanged.

    Raises:
        InvalidBaseIri: When ``base`` is empty, not absolute (``scheme://host/…``),
            carries a backtick/whitespace/control character, contains an empty path
            segment (``//``) — almost always two halves joined with a slash on each
            side — or carries a ``#`` anywhere but as its final character.
    """
    if not base or not isinstance(base, str):
        raise InvalidBaseIri("base_iri must be a non-empty string")
    parts = urlsplit(base)
    if not (parts.scheme and parts.netloc) or _FORBIDDEN.search(base):
        raise InvalidBaseIri(
            f"not a usable base IRI: {base!r}. Expected an absolute IRI "
            "(scheme://host/…) with no backticks, whitespace or control characters; "
            "a trailing '#' or '/' is fine."
        )
    if "//" in parts.path:
        raise InvalidBaseIri(
            f"base IRI carries an empty path segment ('//'): {base!r}. This is almost "
            "always two halves joined with a slash on both sides — fix the assembly; "
            "IRIs minted under it would be a namespace nobody intended."
        )
    if "#" in base and not (base.endswith("#") and base.count("#") == 1):
        # '#' comes only as the **last character** of a hash namespace: the minter detects
        # the hash form with endswith('#'), so a '#' in the middle is misread as the slash
        # form and permanently issues IRIs with a path inside the fragment
        # ({base}#{…}/class/{local}).
        raise InvalidBaseIri(
            f"'#' may only appear as the final character of a base IRI: {base!r}. "
            "Assemble path segments with '/' and put a single '#' at the very end "
            "(e.g. http://host/domain/document#)."
        )
    return base


# --- Piece assembly -----------------------------------------------------------
# If the assembly rule of a permanent identifier lives in the client (a string template in
# the pipeline YAML), neither tests nor review reach it; so the rule is implemented here,
# in one place, and the client sends only the pieces (root, domain, document). During the
# transition the client-assembled and server-assembled values are compared
# (resolve_base_iri), and only after measurement proves the rules equal does it move to
# pieces only.
#
# Rule: the document title is first normalized, whitespace→'_' (:func:`normalize_document`,
# the DBpedia convention: an IRI studded with '%20' is unreadable to people), then encoded
# equivalently to encodeURIComponent (:func:`encode_document`). Normalization is part of
# the canonical rule too, so it happens here rather than in the client; the same
# normalization applies to both the IRI and the graph name, leaving no room for the two
# to drift apart.

# Deployment default root (e.g. http://example.org). The request's iri_root wins.
ROOT_ENV_VAR = "ONTOLOGY_IRI_ROOT"

# Non-alphanumerics that JS encodeURIComponent leaves unencoded; equivalence is the
# contract, so kept as is.
_ENCODE_SAFE = "-_.!~*'()"

_WHITESPACE_RUN = re.compile(r"\s+")


def normalize_document(title: str) -> str:
    """Document title → identifier segment: strip, then whitespace runs become ``_``.

    Deterministic (same title → always the same segment); a title already using
    underscores is unchanged.
    """
    return _WHITESPACE_RUN.sub("_", title.strip())


def encode_document(title: str) -> str:
    """Percent-encode a document title exactly as JS ``encodeURIComponent`` does."""
    return quote(title, safe=_ENCODE_SAFE)


def assemble_base_iri(root: str, domain: str, document: str) -> str:
    """``{root}/{domain}/{encode(normalize(document))}#`` — the single assembly rule.

    Args:
        root: Scheme+host root, with or without a trailing slash (both yield the same
            result — the class of bug this function exists to end).
        domain: Path segment naming the corpus (e.g. ``military``). Must be a single
            bare segment: no ``/``, no ``#``, no characters the validator forbids.
        document: Document title as-is; spaces become ``_`` and the rest is
            percent-encoded here, so free-text titles and non-ASCII are fine.

    Returns:
        The assembled base IRI, already passed through :func:`validate_base_iri`.

    Raises:
        InvalidBaseIri: On a missing/unsound piece or an unusable assembled result.
    """
    if not root or not isinstance(root, str):
        raise InvalidBaseIri("iri root must be a non-empty string (e.g. http://host)")
    if not domain or "/" in domain or "#" in domain:
        raise InvalidBaseIri(
            f"domain must be a single bare path segment (no '/' or '#'): {domain!r}"
        )
    if not document or not normalize_document(document):
        raise InvalidBaseIri("document must be a non-empty string")
    segment = encode_document(normalize_document(document))
    return validate_base_iri(f"{root.rstrip('/')}/{domain}/{segment}#")


def assemble_graph_name(domain: str, document: str) -> str:
    """``{domain}/{normalize(document)}`` — the graph-name rule, owned next to the IRI rule.

    The graph name is a FalkorDB key, so it is not percent-encoded, but the document
    title normalization (whitespace→'_') goes through the **same function** as the IRI:
    the IRI segment and graph name of one document could drift apart by different rules
    only inside this file, and so they do not.
    """
    if not domain or not document:
        raise InvalidBaseIri("graph name needs both domain and document")
    return f"{domain}/{normalize_document(document)}"


def resolve_base_iri(
    base_iri: str | None,
    *,
    root: str | None = None,
    domain: str | None = None,
    document: str | None = None,
) -> tuple[str, str, dict[str, str] | None] | None:
    """Decide this request's base IRI from a full string, pieces, or both.

    ``document`` triggers the piece path. When a full IRI and pieces come **together**,
    the full IRI is compared with the server-assembled one and rejected on a mismatch:
    the contract that verifies, in one request, that the client assembly rule equals the
    server rule during the transition (dual-run check).

    Returns:
        ``(base_iri, source, pieces)``: ``source`` is ``request`` (a full IRI came) |
        ``assembled`` (the server assembled it from pieces); ``pieces`` is the
        ``{root, domain, document}`` used for assembly, for echoing (None if no pieces
        came). ``None`` if nothing came; the env default fallback is left to the caller
        (each module has its own env).

    Raises:
        InvalidBaseIri: When the pieces are incomplete (document only, no domain/root),
            the assembled/full IRI fails validation, or the full IRI does not match the
            assembled one.
    """
    if document is None:
        if base_iri is None:
            return None
        return validate_base_iri(base_iri), "request", None

    if not domain:
        raise InvalidBaseIri(
            "document was given without a domain — the pieces path needs both "
            "(?domain=…&document=…)"
        )
    load_dotenv()
    root_used = root or os.environ.get(ROOT_ENV_VAR)
    if not root_used:
        raise InvalidBaseIri(
            f"no IRI root: pass ?iri_root=… or set {ROOT_ENV_VAR} (e.g. http://example.org)"
        )
    assembled = assemble_base_iri(root_used, domain, document)
    pieces = {"root": root_used, "domain": domain, "document": document}
    if base_iri is not None:
        supplied = validate_base_iri(base_iri)
        if supplied != assembled:
            raise InvalidBaseIri(
                "base_iri disagrees with the server-assembled namespace — the caller's "
                f"assembly rule has drifted: sent {supplied!r}, assembled {assembled!r}"
            )
        return supplied, "request", pieces
    return assembled, "assembled", pieces
