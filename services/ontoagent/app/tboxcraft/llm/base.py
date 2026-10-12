"""LLM provider interface for the domain-term verifier.

Mirrors the embedding provider abstraction: a config-selected provider, local so text stays on-host.
The provider owns transport and structured-output enforcement; callers get a validated
``dict``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Verdict:
    """A domain-term verifier judgment. Advisory only — the human decides."""

    is_official_term: bool
    recommended_action: str  # merge | new | relabel | delete | reject
    proposed_term: str | None
    rationale: str
    confidence: float
    # Whether the candidate is a standard/official *term form* for its concept. Drives the
    # automatic AlterTerm (True) vs HiddenTerm (False) split for a confidently-attached word: a
    # non-standard document artifact (right concept, wrong form) is kept HiddenTerm for ABox
    # mapping. Defaults True so a missing/unavailable verdict keeps the word as a safe AlterTerm.
    is_standard_form: bool = True
    status: str = "ok"  # "ok" | "unavailable" (LLM could not be reached / parsed)

    @classmethod
    def unavailable(cls, reason: str) -> Verdict:
        """A non-blocking placeholder when the LLM is unreachable — the human still decides."""
        return cls(
            is_official_term=False,
            recommended_action="reject",
            proposed_term=None,
            rationale=f"verifier unavailable: {reason}",
            confidence=0.0,
            status="unavailable",
        )

    def to_dict(self) -> dict:
        return {
            "is_official_term": self.is_official_term,
            "recommended_action": self.recommended_action,
            "proposed_term": self.proposed_term,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "is_standard_form": self.is_standard_form,
            "status": self.status,
        }


class LLMProvider(ABC):
    """Judges a prompt and returns a validated structured object."""

    #: Provider identifier (for logging / provenance).
    name: str = "base"

    @abstractmethod
    def judge(self, system: str, user: str) -> dict:
        """Return a JSON object with the Verdict fields. Providers enforce the shape and raise
        on a malformed response rather than returning a partial/None object."""
