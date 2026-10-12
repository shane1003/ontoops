"""Incremental embedding backfill for the ontorouter vector table.

Only rows whose surface text has no current vector in the configured model
space are embedded. "Current" is deliberately stronger than "present":
ontorouter does not index ``public`` precisely because the upstream
pipeline may recreate its tables, and recreating ``surface_form`` reassigns
``surface_form.id``. A vector keyed on a reused id would then describe a
different term while still counting as embedded. Every stored vector therefore
carries the md5 of the text it was produced from, and a row whose hash no
longer matches is re-selected and re-embedded like an unembedded one.

A Postgres advisory lock keeps the periodic pass and a manual trigger from
overlapping. Because that lock is session-scoped in Postgres, the whole
lock/work/unlock span must run on one pinned connection — see
``BackfillRepository.backfill_session`` below.
"""

from contextlib import AbstractContextManager
from typing import Protocol

# Arbitrary but stable. Any other advisory-lock holder in this database must
# not reuse it.
BACKFILL_LOCK_KEY = 0x04707241


class BackfillSession(Protocol):
    """Batch operations bound to the one connection that holds the lock."""

    def unembedded(self, model_id: str, limit: int) -> list[tuple[int, str]]: ...
    def store_embeddings(
        self, model_id: str, rows: list[tuple[int, str, list[float]]]
    ) -> int: ...


class BackfillRepository(Protocol):
    """The database surface the backfill depends on."""

    def backfill_session(
        self, key: int
    ) -> AbstractContextManager[BackfillSession | None]:
        """Pin one connection across the lock, every batch and the unlock.

        Args:
            key: The advisory lock key.

        Returns:
            A context manager yielding a ``BackfillSession`` bound to one
            physical connection, or ``None`` if another pass holds the lock.
        """
        ...


class BatchEmbedder(Protocol):
    """Batch embedding surface."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def run_backfill(
    repo: BackfillRepository,
    embedder: BatchEmbedder,
    model_id: str,
    batch_size: int = 256,
) -> int:
    """Embed every surface form without a current vector in this model space.

    A surface form counts as current only while the stored md5 of its text
    still matches, so upstream id reuse self-heals on the next pass instead of
    leaving vectors silently attached to the wrong terms.

    Commits per batch, so an interrupted pass leaves complete batches behind and
    no partial rows. Returns immediately when another pass holds the lock. The
    lock, every batch read/write and the unlock all run on the single
    connection ``repo.backfill_session`` pins for the whole pass — lock release
    happens inside that context manager, so it fires on every exit path,
    including an exception raised by the embedder or the store.

    Args:
        repo: The backfill repository.
        embedder: The batch embedding client.
        model_id: The configured embedding model id.
        batch_size: Rows per batch.

    Returns:
        The number of rows embedded in this pass.
    """
    embedded = 0
    with repo.backfill_session(BACKFILL_LOCK_KEY) as session:
        if session is None:
            return 0
        while True:
            pending = session.unembedded(model_id, batch_size)
            if not pending:
                break
            vectors = embedder.embed([text for _, text in pending])
            session.store_embeddings(
                model_id,
                [
                    (row_id, text, vector)
                    for (row_id, text), vector in zip(pending, vectors, strict=True)
                ],
            )
            embedded += len(pending)
    return embedded
