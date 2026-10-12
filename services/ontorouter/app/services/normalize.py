"""Query-side normalization and separator-variant expansion.

The stored side is never normalized. Upstream owns the ``public`` schema, so no
expression index matching this pipeline can exist there; running the pipeline in
SQL would force a sequential scan over every surface form. Instead the query is
normalized here and expanded into the separator variants that the upstream
``lower(surface_text)`` index can serve directly.
"""

import itertools
import re
import unicodedata

SEPARATORS = (" ", "-", "_")
_SEPARATOR_RUN = re.compile(r"[\s_\-]+")


def normalize_term(term: str) -> str:
    """Normalize a raw term into the lookup key.

    The pipeline is NFC, lowercase, collapse every run of whitespace,
    underscores and hyphens into a single space, then strip the ends.
    Punctuation is preserved.

    Collapsing happens before stripping so that a leading or trailing separator
    does not survive as a space; stored surface forms never begin or end with a
    separator, so this cannot change any stored-side match.

    Args:
        term: The raw term supplied by the consumer.

    Returns:
        The normalized lookup key.
    """
    folded = unicodedata.normalize("NFC", term).lower()
    return _SEPARATOR_RUN.sub(" ", folded).strip()


def separator_variants(key: str, max_words: int) -> list[str] | None:
    """Expand a normalized key into every single-separator spelling of it.

    Each space in the key is replaced by a space, a hyphen or an underscore, so
    the result set is what ``lower(surface_text)`` could hold for this key. The
    count grows as ``3 ** (words - 1)``, hence the cap.

    Args:
        key: A normalized lookup key.
        max_words: Maximum word count to expand; beyond it the caller must fall
            back to normalizing in SQL.

    Returns:
        The variant spellings, or None when the key exceeds ``max_words``.
    """
    parts = key.split(" ")
    if len(parts) > max_words:
        return None
    if len(parts) == 1:
        return [key]
    variants: list[str] = []
    for seps in itertools.product(SEPARATORS, repeat=len(parts) - 1):
        buffer = [parts[0]]
        for separator, part in zip(seps, parts[1:], strict=True):
            buffer.append(separator)
            buffer.append(part)
        variants.append("".join(buffer))
    return variants
