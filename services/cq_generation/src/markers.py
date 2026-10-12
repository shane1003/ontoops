"""Numbering marker vocabulary: reads "1." / "(1)" / "(a)" / Roman numerals.

Judging from one line alone, "is this a list number or a fixed tag" cannot be decided
here. That decision needs sibling context, which the caller
(odl_tree.resolve_marker_depth) makes.
"""

import re
from dataclasses import dataclass
from typing import Callable

_ROMAN_VALS = [
    (1000, "m"),
    (900, "cm"),
    (500, "d"),
    (400, "cd"),
    (100, "c"),
    (90, "xc"),
    (50, "l"),
    (40, "xl"),
    (10, "x"),
    (9, "ix"),
    (5, "v"),
    (4, "iv"),
    (1, "i"),
]


def _int_to_roman(n: int) -> str:
    res = []
    for v, sym in _ROMAN_VALS:
        while n >= v:
            res.append(sym)
            n -= v
    return "".join(res)


def _roman_to_int(s: str) -> int | None:
    vals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    s = s.lower()
    if not s or any(c not in vals for c in s):
        return None
    total, prev = 0, 0
    for c in reversed(s):
        v = vals[c]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    if _int_to_roman(total) != s:  # reject non-canonical (iiii etc.): round-trip check
        return None
    return total


_MARKER_ROOTS = {
    "ARABIC": (r"\d+", lambda s: int(s)),
    "LOWER": (r"[a-z]", lambda s: ord(s) - ord("a") + 1),
    "UPPER": (r"[A-Z]", lambda s: ord(s) - ord("A") + 1),
    "ROMAN_LOWER": (r"[ivxlcdm]+", lambda s: _roman_to_int(s)),
    "ROMAN_UPPER": (r"[IVXLCDM]+", lambda s: _roman_to_int(s.lower())),
}
_MARKER_WRAPPERS = {
    "DOT": lambda pat: rf"^({pat})\.\s",
    "PAREN": lambda pat: rf"^\(({pat})\)\s",
    "CLOSE": lambda pat: rf"^({pat})\)\s",
}

MARKER_FAMILIES: dict[str, tuple[re.Pattern, Callable[[str], int | None]]] = {}
for _root_name, (_pat, _extractor) in _MARKER_ROOTS.items():
    for _wrapper_name, _wrap_fn in _MARKER_WRAPPERS.items():
        MARKER_FAMILIES[f"{_root_name}_{_wrapper_name}"] = (
            re.compile(_wrap_fn(_pat)),
            _extractor,
        )

# Default priority for disambiguation (used only for the first item under a parent, when
# there is no sequence to continue). Covers cases such as "(i)" matching both
# ROMAN_LOWER_PAREN (=1) and LOWER_PAREN (=9).
MARKER_PRIORITY_ORDER = [
    "ARABIC_DOT",
    "ARABIC_PAREN",
    "ARABIC_CLOSE",
    "ROMAN_LOWER_PAREN",
    "ROMAN_LOWER_DOT",
    "ROMAN_LOWER_CLOSE",
    "ROMAN_UPPER_PAREN",
    "ROMAN_UPPER_DOT",
    "ROMAN_UPPER_CLOSE",
    "LOWER_DOT",
    "LOWER_PAREN",
    "LOWER_CLOSE",
    "UPPER_DOT",
    "UPPER_PAREN",
    "UPPER_CLOSE",
]


@dataclass(frozen=True)
class MarkerMatch:
    family: str
    value: int


def match_markers(text: str) -> list[MarkerMatch]:
    """Returns every marker candidate matching the start of text.

    Several are returned if ambiguous.
    """
    out = []
    for family, (pattern, extractor) in MARKER_FAMILIES.items():
        m = pattern.match(text)
        if not m:
            continue
        value = extractor(m.group(1))
        if value is not None:
            out.append(MarkerMatch(family=family, value=value))
    return out
