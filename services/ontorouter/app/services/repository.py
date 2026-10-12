"""SQL access to the upstream surface forms and the ontorouter embeddings.

Every statement is parameterized and every statement filters deprecated terms.
The ``public`` schema is read-only here: ontorouter never issues DDL or DML
against it.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, NamedTuple, Protocol

from app.clients.pg import to_literal
from app.core.db import Database


class SurfaceFormRow(NamedTuple):
    """One matched surface form joined to its term.

    ``pref_label`` is the term's preferred surface form, which is usually not
    the one that matched: many stored surface forms are ALTER or HIDDEN, and
    the IRI is not a substitute — some local parts differ from the preferred
    label outright (``#AO`` for "Area of Operations"), and the rest need a
    CamelCase split that mangles acronyms.

    ``None`` only if upstream holds no PREF form for the term, which is an
    upstream data defect.
    """

    iri: str
    term_type: str
    form_type: str
    surface_text: str
    pref_label: str | None


class ScoredRow(NamedTuple):
    """A surface form row with a similarity score attached."""

    iri: str
    term_type: str
    form_type: str
    surface_text: str
    pref_label: str | None
    score: float


class Repository(Protocol):
    """The database surface the routing tiers depend on."""

    def exact(self, variants: list[str]) -> list[SurfaceFormRow]: ...

    def exact_by_normalized(self, key: str) -> list[SurfaceFormRow]: ...

    def distinct_keys(self) -> list[str]: ...

    def semantic(
        self, vector_literal: str, model_id: str, limit: int
    ) -> list[ScoredRow]: ...


class PostgresRepository:
    """Repository backed by the live Postgres store.

    Args:
        db: The connection pool wrapper.
        timeout_s: Per-query statement timeout in seconds.
    """

    # LATERAL with LIMIT 1, not a plain LEFT JOIN on form_type = 'PREF': the
    # upstream unique index is (term_id, lang) WHERE form_type = 'PREF', so a
    # term may hold one preferred form per language. A plain join would then
    # multiply every match row, and which pref_label survived the
    # (graph_id, IRI) dedup downstream would be arbitrary. Ordering by
    # "same language as the matched form first" makes the pick deterministic
    # and sensible in advance of any second language appearing.
    _PREF = (
        "LEFT JOIN LATERAL ("
        "  SELECT surface_text FROM surface_form"
        "  WHERE term_id = t.id AND form_type = 'PREF'"
        "  ORDER BY (lang = s.lang) DESC, lang LIMIT 1"
        ") p ON true "
    )

    _SELECT = (
        "SELECT t.id AS iri, t.term_type, s.form_type, s.surface_text, "
        "p.surface_text AS pref_label "
        "FROM surface_form s JOIN tbox_term t ON t.id = s.term_id " + _PREF
    )
    _ACTIVE = "AND t.status <> 'DEPRECATED'"

    EXACT_SQL = _SELECT + "WHERE lower(s.surface_text) = ANY(%s) " + _ACTIVE

    # btrim wraps regexp_replace, not the other way round: normalize_term also
    # collapses separator runs first and strips afterwards, so the two sides
    # are structurally identical rather than agreeing only because no stored
    # surface form happens to start or end with a separator.
    EXACT_NORMALIZED_SQL = (
        _SELECT + "WHERE btrim(regexp_replace(lower(normalize(s.surface_text, NFC)), "
        "'[\\s_-]+', ' ', 'g')) = %s " + _ACTIVE
    )

    # The lexical channel no longer prefetches by prefix: it scans an in-memory
    # index of these keys with rapidfuzz, then fetches the rows for the keys it
    # picked through EXACT_SQL. This statement only feeds that index.
    DISTINCT_KEYS_SQL = (
        "SELECT DISTINCT btrim(regexp_replace("
        "lower(normalize(s.surface_text, NFC)), '[\\s_-]+', ' ', 'g')) "
        "FROM surface_form s JOIN tbox_term t ON t.id = s.term_id "
        "WHERE t.status <> 'DEPRECATED'"
    )

    SEMANTIC_SQL = (
        "SELECT t.id AS iri, t.term_type, s.form_type, s.surface_text, "
        "p.surface_text AS pref_label, "
        "1 - (e.embedding <=> %s::vector) AS score "
        "FROM ontorouter.surface_form_embedding e "
        "JOIN surface_form s ON s.id = e.surface_form_id "
        "JOIN tbox_term t ON t.id = s.term_id " + _PREF + "WHERE "
        "e.embedding_model_id = %s " + _ACTIVE + " "
        "ORDER BY e.embedding <=> %s::vector LIMIT %s"
    )

    # A row counts as embedded only if a vector exists in this model space AND
    # its stored surface_text_md5 still matches the live text. Upstream owns
    # public and may recreate its tables, which reassigns surface_form.id; a
    # vector left pointing at a reused id would otherwise be reported as
    # embedded forever while describing a different term. Folding the hash
    # check into the same NOT EXISTS makes drift indistinguishable from
    # "never embedded", so the ordinary backfill repairs it.
    UNEMBEDDED_SQL = (
        "SELECT s.id, s.surface_text FROM surface_form s "
        "JOIN tbox_term t ON t.id = s.term_id "
        "WHERE NOT EXISTS ("
        "  SELECT 1 FROM ontorouter.surface_form_embedding e "
        "  WHERE e.surface_form_id = s.id AND e.embedding_model_id = %s "
        "    AND e.surface_text_md5 = md5(s.surface_text)) "
        + _ACTIVE
        + " ORDER BY s.id LIMIT %s"
    )

    # The hash is computed by Postgres from the same expression UNEMBEDDED_SQL
    # compares against, so the two can never disagree over encoding. DO UPDATE
    # rather than DO NOTHING: a drifted row is re-selected by UNEMBEDDED_SQL
    # but already occupies the primary key, and DO NOTHING would leave the
    # stale vector in place and re-select the row on every later pass forever.
    STORE_SQL = (
        "INSERT INTO ontorouter.surface_form_embedding "
        "(surface_form_id, embedding_model_id, embedding, surface_text_md5) "
        "VALUES (%s, %s, %s::vector, md5(%s)) "
        "ON CONFLICT (surface_form_id, embedding_model_id) DO UPDATE SET "
        "embedding = EXCLUDED.embedding, "
        "surface_text_md5 = EXCLUDED.surface_text_md5"
    )

    def __init__(self, db: Database, timeout_s: float) -> None:
        self._db = db
        self._timeout_s = timeout_s

    def _fetch(self, sql: str, params: tuple[object, ...]) -> list[SurfaceFormRow]:
        with self._db.connection(self._timeout_s) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [SurfaceFormRow(*row) for row in rows]

    def exact(self, variants: list[str]) -> list[SurfaceFormRow]:
        """Fetch rows whose lowercased surface text is one of the variants.

        Args:
            variants: Separator spellings of the normalized key.

        Returns:
            Matching rows, unordered.
        """
        return self._fetch(self.EXACT_SQL, (variants,))

    def exact_by_normalized(self, key: str) -> list[SurfaceFormRow]:
        """Fetch rows by normalizing in SQL. The fallback for over-long keys.

        Args:
            key: The normalized lookup key.

        Returns:
            Matching rows, unordered.
        """
        return self._fetch(self.EXACT_NORMALIZED_SQL, (key,))

    def distinct_keys(self) -> list[str]:
        """Return every distinct normalized key of an active term.

        Feeds the in-memory fuzzy index. A full scan by construction — it runs
        on a refresh interval, not on the request path.

        Returns:
            The normalized keys, unordered.
        """
        with self._db.connection(None) as conn:
            rows = conn.execute(self.DISTINCT_KEYS_SQL).fetchall()
        return [str(row[0]) for row in rows]

    def semantic(
        self, vector_literal: str, model_id: str, limit: int
    ) -> list[ScoredRow]:
        """Fetch the nearest embedded surface forms in the configured model space.

        Args:
            vector_literal: The query vector as a pgvector literal.
            model_id: Only vectors produced under this model id are considered.
            limit: Maximum rows to return.

        Returns:
            Rows ordered by cosine distance ascending.
        """
        with self._db.connection(self._timeout_s) as conn:
            rows = conn.execute(
                self.SEMANTIC_SQL, (vector_literal, model_id, vector_literal, limit)
            ).fetchall()
        return [ScoredRow(*row) for row in rows]

    @contextmanager
    def backfill_session(self, key: int) -> Iterator["_BackfillSession | None"]:
        """Pin one connection across the advisory lock, every batch and the unlock.

        A connection pool may hand back a *different* physical connection on
        each checkout. Because the advisory lock is session-scoped, taking it
        and releasing it through separate checkouts can silently unlock
        nothing (``pg_advisory_unlock`` just returns false) while the
        connection that actually holds the lock goes back into the pool and
        keeps holding it — for as long as that connection lives, every later
        pass sees the lock as taken and backs off with no error. Checking out
        one connection for the whole pass and handing back a session object
        bound to it avoids that.

        Args:
            key: The advisory lock key.

        Yields:
            A ``_BackfillSession`` bound to one physical connection, or
            ``None`` if another pass already holds the lock.
        """
        with self._db.session() as conn:
            got = conn.execute("SELECT pg_try_advisory_lock(%s)", (key,)).fetchone()
            if not (got and got[0]):
                yield None
                return
            try:
                yield _BackfillSession(conn)
            finally:
                # A batch that raised mid-insert can leave the transaction in
                # a failed state; roll it back before issuing more statements
                # so the unlock itself cannot be rejected as "in failed sql
                # transaction". Nothing committed survives this — by design,
                # only a batch that reached its own commit() persists.
                conn.rollback()
                conn.execute("SELECT pg_advisory_unlock(%s)", (key,))
                conn.commit()


class _BackfillSession:
    """Backfill batch operations bound to one physical connection.

    Args:
        conn: The pooled connection this session is pinned to, checked out
            for the whole backfill pass by ``PostgresRepository.backfill_session``.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def unembedded(self, model_id: str, limit: int) -> list[tuple[int, str]]:
        """Return active surface forms with no vector in this model space.

        Args:
            model_id: The configured embedding model id.
            limit: Maximum rows to return.

        Returns:
            ``(surface_form_id, surface_text)`` pairs, ordered by id.
        """
        rows = self._conn.execute(
            PostgresRepository.UNEMBEDDED_SQL, (model_id, limit)
        ).fetchall()
        return [(int(row[0]), str(row[1])) for row in rows]

    def store_embeddings(
        self, model_id: str, rows: list[tuple[int, str, list[float]]]
    ) -> int:
        """Insert one batch of vectors, with their text hashes, and commit it.

        Committing per batch — rather than once at the end of the pass, even
        though the whole pass shares one connection — is what keeps an
        interrupted pass resumable: only complete, committed batches survive,
        and the ``ON CONFLICT ... DO UPDATE`` rewrites an already-stored row
        with the same values, so re-running a committed batch is a no-op.

        Args:
            model_id: The configured embedding model id.
            rows: ``(surface_form_id, surface_text, vector)`` triples to store.
                The text is hashed by Postgres and stored alongside the vector
                so a later pass can detect that the row it describes changed.

        Returns:
            The number of rows submitted.
        """
        for surface_form_id, surface_text, vector in rows:
            self._conn.execute(
                PostgresRepository.STORE_SQL,
                (surface_form_id, model_id, to_literal(vector), surface_text),
            )
        self._conn.commit()
        return len(rows)
