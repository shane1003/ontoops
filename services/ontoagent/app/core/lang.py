"""Language tag policy, shared by the modules.

Formerly owned by conceptualizer/schemas.py, with tboxcraft duplicating only the constants.
The wire contract's language constants (`und`/`mul`), tag normalization and script
detection live in one place. If the producer (conceptualizer) and the consumer
(tboxcraft) each define the same values, the contract silently diverges the moment only
one side changes.
"""

from __future__ import annotations

import re

# Undetermined language (RFC 5646). Better than asserting `en`/`ko` without grounds, and
# it can be updated later.
UNDETERMINED_LANG = "und"
# Two languages mixed in one literal (ISO 639-2 `mul`, an RFC 5646 registered code). Kept
# for strings mixing Hangul and Latin, which become false when folded to either
# side; folding them to `und` instead says "unknown" and throws away the **known fact**
# that it is mixed.
MIXED_LANG = "mul"

# Loose BCP-47 shape check: a 2-3 letter primary + optional subtags. No tag registry is
# kept, so only the **shape** is checked, not validity (without even the shape there is no
# reason to trust the value).
_LANG_TAG_RE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")

# Hangul: syllables + jamo (old Hangul jamo, compatibility jamo). Hangul is a **script
# used only for Korean**, so its presence alone is evidence of the language, unlike Latin
# letters, which are no evidence of English.
_HANGUL_RE = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
# Non-Hangul "language-carrying" letters: Latin, Cyrillic, kana. Hanja (CJK unified
# ideographs) are shared by Korean/Chinese/Japanese, so they do not count as evidence;
# nor do digits, punctuation and whitespace.
_OTHER_SCRIPT_RE = re.compile(r"[A-Za-zЀ-ӿ぀-ヿ]")


def normalize_lang(value: object) -> str | None:
    """Normalizes a language tag.

    Only the primary subtag is lowercased; the rest is kept as is (`ko`, `en-US`).

    Returns:
        The normalized tag, or None if the value is not tag-shaped (left to the fallback).
    """
    if not isinstance(value, str):
        return None
    tag = value.strip()
    if not tag or not _LANG_TAG_RE.match(tag):
        return None
    primary, *rest = tag.split("-")
    return "-".join([primary.lower(), *rest])


def script_lang(text: str | None) -> str | None:
    """Returns the language that the script alone **settles**.

    - Hangul is the **majority** of letters → `ko`. Hangul is written only for Korean, so
      this is an observation, not a guess, and it takes precedence over the tag the LLM
      gave: even if the model calls a Hangul term `en`, the string says otherwise. A single
      mixed letter does not make it mixed, because Korean sentences often quote foreign
      terms (such a sentence is Korean, not bilingual).
    - Hangul present but not the majority → `mul` (mixed). A label mixing Hangul and
      Latin is half false when folded to either side. Nor is it folded
      to `und`: being mixed is a known fact, not an unknown one.
    - No Hangul → None. **Latin letters are no evidence of English** (romanized Korean,
      Spanish and German are all Latin). Insisting on `en` here was the original defect,
      so this case is left to the tag of whoever produced the text (the classifier) or to
      the request fallback.

    Digits, punctuation, whitespace and Hanja count toward neither side (Hanja are shared
    by Korean/Chinese/Japanese and cannot be evidence). So a Hangul term with digits is
    `ko`, and a string of digits and punctuation is not judged at all.

    Returns:
        The settled language, or None if the script cannot settle it.
    """
    if not text:
        return None
    hangul = len(_HANGUL_RE.findall(text))
    if not hangul:
        return None
    other = len(_OTHER_SCRIPT_RE.findall(text))
    return "ko" if hangul >= other else MIXED_LANG
