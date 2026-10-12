"""Request and response DTOs for the lookup endpoints."""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

TERM_MAX_LENGTH = 256
BATCH_MAX_ITEMS = 100


class TierDepth(StrEnum):
    """How deep to run the tier pipeline. Each value includes the ones above it."""

    EXACT = "exact"
    LEXICAL = "lexical"
    SEMANTIC = "semantic"


class PointerEntry(BaseModel):
    """A confirmed pointer into a graph. No score: exactness is the guarantee.

    ``label_form`` describes the surface form that matched; ``pref_label`` is
    the term's preferred name, which is usually a different string. It is null
    only when upstream holds no preferred form for the term.
    """

    graph_id: str
    iri: str
    term_type: str
    label_form: str
    pref_label: str | None


class ScoredEntry(PointerEntry):
    """A suggestion pointer, scored on its own channel's scale."""

    score: float


class Suggestions(BaseModel):
    """The two suggestion channels. Their score scales are never merged."""

    lexical: list[ScoredEntry] = Field(default_factory=list)
    semantic: list[ScoredEntry] = Field(default_factory=list)


class LookupMeta(BaseModel):
    """Response metadata."""

    tiers_executed: list[str]


class LookupResponse(BaseModel):
    """The answer for one term."""

    term: str
    normalized_key: str
    results: list[PointerEntry]
    suggestions: Suggestions
    meta: LookupMeta


class BatchLookupRequest(BaseModel):
    """A batch of terms; single-term logic is repeated per item."""

    terms: Annotated[list[str], Field(min_length=1, max_length=BATCH_MAX_ITEMS)]

    @field_validator("terms")
    @classmethod
    def reject_blank_terms(cls, terms: list[str]) -> list[str]:
        """Reject empty or whitespace-only items.

        Args:
            terms: The submitted terms.

        Returns:
            The terms unchanged.

        Raises:
            ValueError: If any item is blank or longer than the term limit.
        """
        for term in terms:
            if not term.strip():
                raise ValueError("terms must not be empty or whitespace-only")
            if len(term) > TERM_MAX_LENGTH:
                raise ValueError(f"terms must be at most {TERM_MAX_LENGTH} characters")
        return terms


class BatchLookupResponse(BaseModel):
    """One answer per submitted term, in submission order."""

    items: list[LookupResponse]
