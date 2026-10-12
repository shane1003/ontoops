"""Index competency questions by id for gloss resolution (DB-backed in production)."""

from __future__ import annotations

from app.tboxcraft.models import CompetencyQuestion


class CQIndex:
    """Lookup of competency questions by id (``source_cq_ids`` join target)."""

    def __init__(self, questions: list[CompetencyQuestion]):
        self._by_id = {q.id: q for q in questions}

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, cq_id: str) -> bool:
        return cq_id in self._by_id

    def get(self, cq_id: str) -> CompetencyQuestion | None:
        return self._by_id.get(cq_id)

    def texts_for(self, cq_ids: list[str] | tuple[str, ...]) -> list[str]:
        """Resolve a list of ids to their question strings, skipping unknown ids."""
        return [q.question for cid in cq_ids if (q := self._by_id.get(cid))]


def load_cqs_from_db(repo) -> CQIndex:
    """Build a :class:`CQIndex` from the DB ``cq`` table.

    The pipeline reads competency questions (and their expected answers) that were
    persisted upstream by CQ generation. ``repo`` is any
    object exposing ``fetch_competency_questions() -> list[(cq_uuid, question, answer)]``.
    """
    return CQIndex(
        [
            CompetencyQuestion(id=uuid, question=question, answer=answer)
            for uuid, question, answer in repo.fetch_competency_questions()
        ]
    )


def resolve_cqs(repo=None) -> CQIndex:
    """Competency questions come from the store, keyed by the id a component cites in
    ``source_cq_ids``.

    "Upstream" is concretely ``POST /cq`` (router.py), which must run before an ingest that
    cites those ids: a component whose CQ is absent is stored without provenance, and the ABox
    loader then fails on the foreign key one stage later.
    An empty index is valid: glosses fall back to name + notes alone. ``load_cqs`` (CSV)
    remains a standalone utility for tests and one-off loads."""
    if repo is not None:
        return load_cqs_from_db(repo)
    return CQIndex([])
