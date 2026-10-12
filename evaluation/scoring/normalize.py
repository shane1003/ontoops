"""The one string normalization every string match in scoring goes through.

Strict on purpose: no stemming, no plural folding, no synonym
table, so every coverage figure is a lower bound. Steps, in order:

1. Unicode NFKC; private-use glyphs (bullets from PDF fonts) removed;
   typographic quotes and dashes made plain.
2. A leading proponent marker ``*`` removed.
3. CamelCase split, for labels that come from IRI local names.
4. Lower case; punctuation to spaces; spaces collapsed.
5. A leading article (a, an, the) removed.
"""

import re
import unicodedata

_PRIVATE_USE = re.compile("[-]")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_PUNCT = re.compile(r"[^\w\s]|_")
_SPACES = re.compile(r"\s+")
_ARTICLE = re.compile(r"^(a|an|the)\s+")
_PARENTHETICAL = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")
_PLAIN = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def normalize(text: str) -> str:
    """Normalizes one label, term or query for matching."""
    text = unicodedata.normalize("NFKC", text)
    text = _PRIVATE_USE.sub(" ", text).translate(_PLAIN).strip().lstrip("*").strip()
    text = _CAMEL.sub(" ", text)
    text = _PUNCT.sub(" ", text.lower())
    text = _SPACES.sub(" ", text).strip()
    return _ARTICLE.sub("", text)


def variants(text: str) -> set[str]:
    """The forms a heading may be matched by.

    ``"air defense artillery (ADA)"`` matches by the whole, by the words
    before the parenthesis, or by what is inside it; empty forms are dropped.
    """
    forms = {normalize(text)}
    match = _PARENTHETICAL.match(text)
    if match:
        forms.add(normalize(match.group(1)))
        forms.add(normalize(match.group(2)))
    return {form for form in forms if form}
