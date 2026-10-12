"""In-memory index of every normalized surface-form key.

The lexical channel used to generate candidates with a ``LIKE '<key>%'``
prefetch. That could only ever find strings the query is a prefix of, so it
never produced a candidate for a misspelling — a typo anywhere in the key left
the prefetch empty and rapidfuzz was handed nothing to score. What looked like
fuzzy matching was autocomplete.

Scanning every distinct key with rapidfuzz instead finds them: single-character
deletions, substitutions, insertions, transpositions and two-character
combinations all land in the top five, in a few milliseconds.

The cost is that the keys live in memory and go stale between refreshes. Only
the suggestion channel is affected: the exact tier still reads live SQL on
every request, so a just-ingested document is matched exactly straight away
and is merely absent from suggestions until the next refresh.
"""

import logging

from rapidfuzz import fuzz, process

from app.services.repository import Repository

logger = logging.getLogger(__name__)


class KeyIndex:
    """Every distinct normalized key, held for fuzzy candidate generation.

    Not thread-safe by locking, and it does not need to be: ``refresh`` builds
    a new list and rebinds it in one assignment, which is atomic under CPython,
    so a concurrent reader sees either the old list or the new one and never a
    partially built one.
    """

    def __init__(self) -> None:
        self._keys: list[str] = []

    def refresh(self, repo: Repository) -> int:
        """Rebuild the key list from the store.

        Args:
            repo: The database repository.

        Returns:
            The number of keys loaded.
        """
        keys = repo.distinct_keys()
        self._keys = keys
        return len(keys)

    def nearest(self, key: str, limit: int, min_score: float) -> list[str]:
        """Return the keys most similar to ``key``, best first.

        Args:
            key: A normalized lookup key.
            limit: Maximum keys to return.
            min_score: Admission threshold on the [0, 1] similarity scale.

        Returns:
            Matching keys, or an empty list when the index has not been
            refreshed yet — a cold index degrades the channel rather than
            failing the request.
        """
        if not self._keys:
            return []
        matches = process.extract(
            key,
            self._keys,
            scorer=fuzz.ratio,
            limit=limit,
            score_cutoff=min_score * 100,
        )
        return [match for match, _score, _index in matches]
