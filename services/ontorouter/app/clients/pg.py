"""pgvector literal formatting.

pgvector round-trips vectors as strings unless the pgvector-python adapter is
registered. Only the write direction is needed here — the service never reads a
vector back out — so one two-line function beats the extra dependency.
"""

from collections.abc import Iterable


def to_literal(vec: Iterable[float]) -> str:
    """Format a vector as a pgvector literal: ``[1,2,3]``."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"
