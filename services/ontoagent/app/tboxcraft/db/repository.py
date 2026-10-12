"""Persistence over the relational TBox store (tbox_term & friends).

This layer is the **only** place that knows the storage vocabulary. The pipeline speaks its own
terms throughout — ``'class'``/``'object_property'``/``'data_property'`` for kinds, ``'alter'``/
``'hidden'`` for surface roles, ``'gray'``/``'relabel'``/``'delete'``/``'new'`` for review entries,
``'pending'``/``'applied'``/``'rejected'`` for review status — and the schema stores the uppercase
forms its CHECK constraints require. Translating at this boundary keeps the mapping in one file
instead of scattering ``.upper()`` calls through the pipeline.

Identity: a term's IRI *is* its primary key, so the "concept id" the pipeline passes around is the
IRI itself. :meth:`get_concept_id` therefore returns the IRI (or None when unknown) and doubles as
an existence check.

Vectors are stored inline on ``tbox_term`` and are NOT NULL — a term cannot exist without one, so
:meth:`upsert_concept` takes the model and vector rather than leaving a row to be filled in later.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime

import numpy as np
from psycopg.sql import SQL, Identifier
from psycopg.types.json import Json
from psycopg_pool import ConnectionPool

from app.tboxcraft.emit.topology import break_cycles
from app.tboxcraft.represent import local_of, namespace_prefix

logger = logging.getLogger(__name__)

# Session-wide advisory-lock key serializing the ontology snapshot read-modify-write, so two
# concurrent update/review runs can't both branch off the same snapshot and lose one's changes.
_ONTOLOGY_LOCK_KEY = 815_002

# --- pipeline vocabulary <-> storage vocabulary -------------------------------------------------

_KIND_TO_DB = {
    "class": "CLASS",
    "object_property": "OBJECT_PROP",
    "data_property": "DATA_PROP",
}
_KIND_FROM_DB = {v: k for k, v in _KIND_TO_DB.items()}

_ROLE_TO_DB = {"pref": "PREF", "alter": "ALTER", "hidden": "HIDDEN"}

# The pipeline calls a brand-new concept 'new'; the queue calls that proposal a CREATE.
# kind_conflict: one name minted under two kinds. Its own entry type rather than another
# GRAY because the decision is different in kind — not "is this the same concept as that
# one" but "is this term a relation or an attribute" — and because the unique key is
# (candidate_uid, entry_type): filing it as GRAY would collide with a real gray item on
# the same candidate and one of the two would be dropped.
_ENTRY_TO_DB = {
    "gray": "GRAY",
    "relabel": "RELABEL",
    "delete": "DELETE",
    "new": "CREATE",
    "kind_conflict": "KIND_CONFLICT",
}
_ENTRY_FROM_DB = {v: k for k, v in _ENTRY_TO_DB.items()}

_STATUS_TO_DB = {"pending": "PENDING", "applied": "APPROVED", "rejected": "REJECTED"}
_STATUS_FROM_DB = {v: k for k, v in _STATUS_TO_DB.items()}


def _vec_to_np(value) -> np.ndarray | None:
    """Coerce a fetched pgvector column to a float32 ndarray, version-agnostically.

    Depending on the installed pgvector-python version, ``register_vector`` yields either a numpy
    array (older) or a ``pgvector.Vector`` object carrying ``.to_numpy()`` (newer). Passing the
    latter straight to ``np.asarray(..., dtype=np.float32)`` raises ``TypeError: float() argument
    must be ... not 'Vector'``; handle both shapes plus ``None``.
    """
    if value is None:
        return None
    to_numpy = getattr(value, "to_numpy", None)
    if callable(to_numpy):
        value = to_numpy()
    return np.asarray(value, dtype=np.float32)


def _model_key(name: str, revision: str) -> str:
    """A stable, <=50 char id for an embedding model row (the column is VARCHAR(50), and HF ids
    plus a revision routinely exceed that — hash the pair instead of truncating it)."""
    return hashlib.sha256(f"{name}@{revision}".encode()).hexdigest()[:32]


class StaleOntologyError(RuntimeError):
    """The ontology advanced since the caller loaded it — an optimistic-concurrency conflict."""


class Repository:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    @contextmanager
    def _connection(self, conn):
        """Yield the supplied connection, or a fresh (auto-committing) one."""
        if conn is not None:
            yield conn
        else:
            with self.pool.connection() as fresh:
                yield fresh

    @contextmanager
    def transaction(self):
        """A single connection wrapped in one transaction for atomic multi-write persistence."""
        with self.pool.connection() as conn, conn.transaction():
            yield conn

    # --- provenance ---

    def upsert_embedding_model(
        self,
        name: str,
        revision: str,
        dim: int,
        *,
        normalized: bool = True,  # kept for call-site compatibility; not stored
        mean: np.ndarray | None = None,
        conn=None,
    ) -> str:
        """Register an embedding model + its whitening mean, returning the model id.

        The mean is write-once: it defines the space every vector stored against this model lives
        in, so a second bootstrap must NOT overwrite it (a DB trigger refuses anyway). An existing
        (name, revision) therefore wins and its id is returned — later batches are expected to
        centre their queries with the stored mean, not to install a new one.
        """
        if mean is None:
            raise ValueError(
                "whitening mean is required — vectors cannot be centred without it"
            )
        mean_vec = np.asarray(mean, dtype=np.float32)
        model_id = _model_key(name, revision)
        with self._connection(conn) as c:
            c.execute(
                "INSERT INTO embedding_model (id, model_name, revision, dimensions, whitening_mean) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (model_name, revision) DO NOTHING",
                (model_id, name, revision, dim, mean_vec),
            )
            row = c.execute(
                "SELECT id FROM embedding_model WHERE model_name = %s AND revision = %s",
                (name, revision),
            ).fetchone()
            return row[0]

    def get_model_mean(self, name: str, revision: str) -> np.ndarray | None:
        """The whitening mean stored for a model, or None if unknown."""
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT whitening_mean FROM embedding_model WHERE model_name = %s AND revision = %s",
                (name, revision),
            ).fetchone()
        return _vec_to_np(row[0]) if row else None

    def get_model_id(self, name: str, revision: str) -> str | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT id FROM embedding_model WHERE model_name = %s AND revision = %s",
                (name, revision),
            ).fetchone()
            return row[0] if row else None

    # --- ontology snapshot ---

    def save_ontology_version(
        self,
        ttl: str,
        *,
        base_iri: str,
        note: str | None = None,
        base_id: str | None = None,
        reasoner_status: str = "VALIDATED",
        conn=None,
    ) -> str:
        """Append a new canonical TBox snapshot to ``base_iri``'s chain; its latest is authoritative.

        ``base_iri`` is required rather than defaulted. One document is one KG, so a snapshot only
        means something inside a namespace, and a default would silently file this batch under
        whichever namespace happened to be configured — the failure mode this scoping exists to
        remove.

        When ``base_id`` is given (the snapshot the caller mutated from), take a transaction
        advisory lock and verify the latest snapshot *of this chain* is still ``base_id`` before
        appending; a concurrent writer that advanced it raises :class:`StaleOntologyError` so the
        caller rolls back and retries rather than silently dropping the other run's changes.

        The lock is keyed by namespace as well, so two documents ingesting at once do not serialize
        against each other — they touch different chains and can never make each other stale.

        Snapshots are only written past the validation gate, hence the VALIDATED default.
        """
        with self._connection(conn) as c:
            if base_id is not None:
                c.execute(
                    "SELECT pg_advisory_xact_lock(%s, hashtext(%s))",
                    (_ONTOLOGY_LOCK_KEY, base_iri),
                )
                row = c.execute(
                    "SELECT id FROM ontology_snapshot WHERE base_iri = %s "
                    "ORDER BY created_at DESC, id DESC LIMIT 1",
                    (base_iri,),
                ).fetchone()
                current = row[0] if row else None
                if current != base_id:
                    raise StaleOntologyError(
                        f"ontology advanced from {base_id} to {current}"
                    )
            # Time-ordered id so the chain reads chronologically; the suffix keeps two snapshots
            # committed in the same second distinct.
            snap_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
            c.execute(
                "INSERT INTO ontology_snapshot "
                "(id, parent_snapshot_id, serialized_turtle, file_hash, reasoner_status, note, "
                " base_iri) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    snap_id,
                    base_id,
                    ttl,
                    hashlib.sha256(ttl.encode()).hexdigest(),
                    reasoner_status,
                    note,
                    base_iri,
                ),
            )
            return snap_id

    def get_latest_ontology(self, base_iri: str) -> tuple[str, str] | None:
        """The most recent (id, ttl) snapshot of ``base_iri``'s chain, or None when it has none.

        None means *this namespace* has no ontology, not that the store is empty — a second
        document arriving at a populated store still starts cold, which is the point.
        """
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT id, serialized_turtle FROM ontology_snapshot WHERE base_iri = %s "
                "ORDER BY created_at DESC, id DESC LIMIT 1",
                (base_iri,),
            ).fetchone()
            return (row[0], row[1]) if row else None

    # --- review queue ---

    def enqueue_review(
        self,
        *,
        candidate_uid: str,
        kind: str,
        entry_type: str,
        base_iri: str,
        gloss: str | None = None,
        candidate_vector: np.ndarray | None = None,
        nearest: list | None = None,
        payload: dict | None = None,
        conn=None,
    ) -> str | None:
        """Queue one item for human review. Idempotent per (candidate_uid, entry_type): a repeat
        enqueue for the same candidate is a no-op and returns None (single-pipeline dedup).

        The candidate vector is stored, not recomputed at decision time: re-embedding later could
        land in a different vector space, so the queue freezes what was actually routed. It is
        required for anything that can become a term; a ``delete`` candidate is exempt because a
        dangling orphan is an IRI referenced without a term behind it — there is nothing to embed.
        ``kind`` travels inside the payload: the queue holds candidates that are not terms yet, so
        there is no term row to carry it.
        """
        if candidate_vector is None and entry_type != "delete":
            raise ValueError(
                f"candidate_vector is required for a {entry_type!r} item — it must freeze its vector"
            )
        vec = (
            None
            if candidate_vector is None
            else np.asarray(candidate_vector, dtype=np.float32)
        )
        review_id = uuid.uuid4().hex
        body = {**(payload or {}), "kind": kind}
        with self._connection(conn) as c:
            row = c.execute(
                "INSERT INTO review_queue "
                "(id, candidate_uid, entry_type, base_iri, gloss, candidate_vector, nearest, "
                " diff_payload) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (candidate_uid, entry_type) DO NOTHING RETURNING id",
                (
                    review_id,
                    candidate_uid,
                    _ENTRY_TO_DB.get(entry_type, entry_type.upper()),
                    base_iri,
                    gloss,
                    vec,
                    Json(nearest or []),
                    Json(body),
                ),
            ).fetchone()
            return row[0] if row else None

    def _review_row(self, row, cols) -> dict:
        """Shape a review row into the pipeline's vocabulary (payload carries ``kind``)."""
        item = dict(zip(cols, row))
        payload = item.get("payload") or {}
        item["kind"] = payload.get("kind")
        item["entry_type"] = _ENTRY_FROM_DB.get(item["entry_type"], item["entry_type"])
        if "status" in item:
            item["status"] = _STATUS_FROM_DB.get(item["status"], item["status"])
        return item

    def list_pending_reviews(
        self, limit: int = 100, *, base_iri: str | None = None
    ) -> list[dict]:
        """Pending review items, oldest first; one namespace's when ``base_iri`` is given.

        A reviewer works on one document. Unscoped, the listing interleaves two documents' items
        with nothing on the row saying which is which, and the decisions are not interchangeable —
        "is this the same concept as that one" has a different answer in a different graph.

        ``None`` still lists everything, because an operator looking at the whole queue is a real
        need and this is a read. The write side has no such option: every enqueue names its
        namespace, so a row filed under no namespace can only be one written before the column
        existed.
        """
        cols = (
            "id",
            "candidate_uid",
            "entry_type",
            "gloss",
            "nearest",
            "payload",
            "llm_verdict",
            "created_at",
        )
        sql = (
            "SELECT id, candidate_uid, entry_type, gloss, nearest, diff_payload, verdict, "
            "created_at FROM review_queue WHERE status = 'PENDING'"
        )
        params: list = []
        if base_iri is not None:
            sql += " AND base_iri = %s"
            params.append(base_iri)
        sql += " ORDER BY created_at, id LIMIT %s"
        params.append(limit)
        with self.pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._review_row(r, cols) for r in rows]

    def get_review(self, review_id: str) -> dict | None:
        cols = (
            "id",
            "candidate_uid",
            "entry_type",
            "gloss",
            "nearest",
            "payload",
            "status",
            "decision",
            "llm_verdict",
            # The namespace this item was queued under. Applying a decision mints and re-gates,
            # so the applier has to act in the graph the item came from rather than in whatever
            # this instance's default happens to be — see apply_review_decision.
            "base_iri",
        )
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT id, candidate_uid, entry_type, gloss, nearest, diff_payload, status, "
                "decision, verdict, base_iri FROM review_queue WHERE id = %s",
                (review_id,),
            ).fetchone()
        return self._review_row(row, cols) if row else None

    def set_review_verdict(self, review_id: str, verdict: dict, *, conn=None) -> None:
        """Attach the advisory domain-term verifier verdict to a review item. The human decision
        remains authoritative — this is context only."""
        with self._connection(conn) as c:
            c.execute(
                "UPDATE review_queue SET verdict = %s WHERE id = %s",
                (Json(verdict), review_id),
            )

    def mark_review_resolved(
        self, review_id: str, status: str, decision: dict | None = None, *, conn=None
    ) -> None:
        """Flip a pending item to 'applied' or 'rejected' and record the decision."""
        with self._connection(conn) as c:
            c.execute(
                "UPDATE review_queue SET status = %s, decision = %s, resolved_at = now() "
                "WHERE id = %s",
                (
                    _STATUS_TO_DB.get(status, status.upper()),
                    Json(decision) if decision is not None else None,
                    review_id,
                ),
            )

    def get_review_vector(self, review_id: str) -> np.ndarray | None:
        """The stored (already-centered) candidate vector for a queued item, or None."""
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT candidate_vector FROM review_queue WHERE id = %s", (review_id,)
            ).fetchone()
        return _vec_to_np(row[0]) if row else None

    # --- terms ---

    def upsert_concept(
        self,
        iri: str,
        kind: str,
        pref_labels: dict[str, str],
        canonical_name: str,  # kept for call-site compatibility; not stored
        *,
        embedding_model_id: str,
        embedding_val: np.ndarray,
        snapshot_id: str | None = None,
        conn=None,
    ) -> str:
        """Create/refresh a term and return its IRI (which is its primary key).

        The vector is required rather than attached afterwards: the column is NOT NULL, so a
        two-step insert-then-update would fail on the first step. ``pref_labels`` become PREF
        surface forms — the schema has no label column, one representative per language lives in
        ``surface_form``.
        """
        vec = np.asarray(embedding_val, dtype=np.float32)
        with self._connection(conn) as c:
            c.execute(
                "INSERT INTO tbox_term "
                "(id, term_type, status, embedding_model_id, embedding_val, created_in_snapshot_id) "
                "VALUES (%s, %s, 'ACTIVE', %s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET term_type = EXCLUDED.term_type, "
                "embedding_model_id = EXCLUDED.embedding_model_id, "
                "embedding_val = EXCLUDED.embedding_val",
                (
                    iri,
                    _KIND_TO_DB.get(kind, kind.upper()),
                    embedding_model_id,
                    vec,
                    snapshot_id,
                ),
            )
        if pref_labels:
            self.set_pref_labels(iri, pref_labels, conn=conn)
        return iri

    def set_pref_labels(
        self, iri: str, pref_labels: dict[str, str], *, conn=None
    ) -> None:
        """Set the representative label per language (one PREF per language is a DB invariant).

        The mirror of :meth:`insert_surface_forms`'s "never shadow the term's own PREF" guard, which
        stops an alias equal to the label going in. This is the other direction: a relabel moves the
        label onto a string the term may already hold as an alias (``review.py`` demotes the old
        label to one, so a later relabel back to it is ordinary), and a term listing its own name as
        a synonym of itself is the same nonsense read the other way round. Since
        ``idx_unique_surface_role`` now keys on ``(term_id, lower(surface_text), lang)`` it is also
        the difference between this and a constraint violation surfacing as a 500 out of a review.
        """
        with self._connection(conn) as c:
            for lang, text in pref_labels.items():
                c.execute(
                    "DELETE FROM surface_form WHERE term_id = %s AND lang = %s "
                    "AND form_type <> 'PREF' AND lower(surface_text) = lower(%s)",
                    (iri, lang, text),
                )
                c.execute(
                    "INSERT INTO surface_form (term_id, surface_text, lang, form_type, source) "
                    "VALUES (%s, %s, %s, 'PREF', 'pipeline') "
                    "ON CONFLICT (term_id, lang) WHERE form_type = 'PREF' "
                    "DO UPDATE SET surface_text = EXCLUDED.surface_text",
                    (iri, text, lang),
                )

    def insert_surface_forms(
        self,
        iri: str,
        entries: list[tuple[str, str, str | None]],
        *,
        term_type: str = "alter",
        update_role: bool = False,
        conn=None,
    ) -> None:
        """entries: list of (surface, lang, source). ``term_type`` is the domain-expert role —
        ``'alter'`` (kept synonym, default) or ``'hidden'`` (dropped but retained for ABox
        mapping). PREF is never written here; :meth:`set_pref_labels` owns it.

        Default is a no-op on conflict — the automatic ingest path must never silently overwrite an
        existing role. ``update_role=True`` marks a human decision, which is authoritative and must
        move an existing surface between roles; since role is part of the unique key, that means
        deleting the other role first rather than an ON CONFLICT update.
        """
        if not entries:
            return
        role = _ROLE_TO_DB.get(term_type, term_type.upper())
        rows: list[tuple] = []
        with self._connection(conn) as c:
            for surface, lang, source in entries:
                if update_role:
                    c.execute(
                        # Case-insensitively, like the key it has to move a row under: an exact
                        # match would leave a differently-spelled row in the old role and the
                        # insert would then hit idx_unique_surface_role and do nothing, so the
                        # human decision would be silently dropped.
                        "DELETE FROM surface_form WHERE term_id = %s "
                        "AND lower(surface_text) = lower(%s) "
                        "AND lang = %s AND form_type <> %s AND form_type <> 'PREF'",
                        (iri, surface, lang, role),
                    )
                # Never shadow the term's own PREF. The unique key includes form_type, so a
                # surface equal to the representative label does not conflict — it lands as a
                # SECOND row for the same text, and the term reads as having an alias for its
                # own name. Callers pass a component's every writable surface (its correct_term
                # is usually exactly that label), so the guard belongs here, not in each of them.
                #
                # Compared case-insensitively to match :meth:`resolve_surface`, which is the only
                # other surface lookup and goes through the lower(surface_text) index. An exact
                # comparison would let a case variant through, and that variant is not merely
                # cosmetic: resolve_surface excludes PREF on purpose, so the variant row makes the
                # representative label resolvable again through the back door — the exclusion
                # would hold or not depending on which spellings happened to be inserted before.
                rows.append(
                    (iri, surface, lang, role, source or "pipeline", iri, surface, lang)
                )
            # One round trip for the whole term instead of one per surface. Same statement, same
            # guard — executemany only changes how many times the client waits, which is what a
            # bootstrap writing thousands of surfaces one at a time was paying for. The topology
            # writer has batched this way since it was written.
            with c.cursor() as cur:
                cur.executemany(
                    "INSERT INTO surface_form (term_id, surface_text, lang, form_type, source) "
                    "SELECT %s, %s, %s, %s, %s WHERE NOT EXISTS ("
                    "  SELECT 1 FROM surface_form WHERE term_id = %s "
                    "  AND lower(surface_text) = lower(%s) AND lang = %s AND form_type = 'PREF') "
                    "ON CONFLICT DO NOTHING",
                    rows,
                )

    def set_concept_verdict(
        self, iri: str, verdict: dict, *, model: str, at=None, conn=None
    ) -> None:
        """Record the advisory domain-term verdict on a term. Never gates — questionable seeds are
        found later by querying it. ``model`` is mandatory: verdicts from different verifiers are
        not comparable, and the schema refuses a verdict without provenance."""
        with self._connection(conn) as c:
            c.execute(
                "UPDATE tbox_term SET llm_verdict = %s, llm_verdict_model = %s, "
                "llm_verdict_at = %s WHERE id = %s",
                (Json(verdict), model, at or datetime.now(UTC), iri),
            )

    def deprecate_concept(self, iri: str, *, conn=None) -> None:
        """Retire a term without removing it (deprecate-in-place) so references never dangle."""
        with self._connection(conn) as c:
            c.execute(
                "UPDATE tbox_term SET status = 'DEPRECATED' WHERE id = %s", (iri,)
            )

    def demote_to_draft(self, iris: list[str], *, conn=None) -> int:
        """Move ACTIVE terms back to DRAFT — minted, no longer asserted. Returns rows changed.

        The store has always allowed DRAFT and nothing ever wrote it; this is that state's first
        use. It is the right one here: every consumer reads ``status = 'ACTIVE'`` (the Classes /
        ObjectProperties / DataProperties views, the ABox resolver), so a demoted term stops
        resolving while its row, foreign keys and provenance stay exactly where they were, and an
        approved review can put it back. DEPRECATED means "superseded" — a judgement — and the
        case this exists for is precisely one where no judgement has been made yet.

        Only ACTIVE rows move, so re-running a batch does not walk a DEPRECATED term backwards,
        and the returned count is what actually changed rather than what was asked for. That
        matters: this writes to rows an *earlier* batch persisted, so the caller has to be able to
        report which terms it touched and why.
        """
        if not iris:
            return 0
        with self._connection(conn) as c:
            return c.execute(
                "UPDATE tbox_term SET status = 'DRAFT' "
                "WHERE id = ANY(%s) AND status = 'ACTIVE'",
                (list(iris),),
            ).rowcount

    def restore_draft_to_active(self, iris: list[str], *, conn=None) -> int:
        """Undo :meth:`demote_to_draft`. Returns rows changed.

        The demotion happens before the batch is routed, so that the rest of the run sees the
        store a fresh run would; the gate that decides whether any of it counts runs long after.
        A batch that then fails the gate leaves the earlier terms retired with **nothing queued**
        to say why — the queue row is written on the persist path, which a failed gate never
        reaches. That is a term with no owner: invisible to every consumer, and no record of the
        decision it is waiting on.

        A compensating write rather than one transaction spanning both, because the demotion has
        to be *visible to the routing reads in between* — wrapping them all would mean threading
        the open connection through every store call the middle of the pipeline makes, which is
        the change most likely to reintroduce the argument/signature bugs this pipeline keeps
        hitting. Undoing one small write at the point the decision is known is the smaller move.

        Only DRAFT rows come back, so a term an earlier decision deprecated is not resurrected,
        and re-running is a no-op rather than a walk backwards.
        """
        if not iris:
            return 0
        with self._connection(conn) as c:
            return c.execute(
                "UPDATE tbox_term SET status = 'ACTIVE' "
                "WHERE id = ANY(%s) AND status = 'DRAFT'",
                (list(iris),),
            ).rowcount

    # Dependents this repository must carry by hand. Discovered from the catalog rather than
    # named here on purpose: the question is not "does abox_individual exist" but "does anything
    # depend on this term in a way the database will not resolve for us", and hardcoding today's
    # answer is how a fix stops covering tomorrow's table.
    #
    # The delete rule partitions them exactly. ON DELETE CASCADE ('c') is this repo's own tables —
    # the store cleans them up. SET NULL ('n') is review_queue, whose link is deliberately severable
    # (the note survives as an audit record, keyed by candidate_uid). RESTRICT ('r') is everything
    # else: rows the database refuses to touch, which is precisely the set nobody was handling.
    _RESTRICT_DEPENDENTS_SQL = """
        SELECT cl.relname, att.attname
        FROM pg_constraint con
        JOIN pg_class cl ON cl.oid = con.conrelid
        JOIN unnest(con.conkey) AS k(attnum) ON true
        JOIN pg_attribute att ON att.attrelid = con.conrelid AND att.attnum = k.attnum
        WHERE con.confrelid = 'tbox_term'::regclass
          AND con.contype = 'f' AND con.confdeltype = 'r'
        ORDER BY 1, 2
    """

    @staticmethod
    def _restrict_dependents(c) -> list[tuple[str, str]]:
        """``(table, column)`` for every FK onto ``tbox_term`` the store will not resolve itself."""
        return [
            (r[0], r[1])
            for r in c.execute(Repository._RESTRICT_DEPENDENTS_SQL).fetchall()
        ]

    def count_external_dependents(self, iri: str, *, conn=None) -> int:
        """How many rows outside this repo's own tables still point at ``iri`` (0 if none exist).

        Answers "is this term still referenced?" for facts the ontology graph cannot show. An
        individual attached to a class leaves no triple in the TTL, so a graph-only check reports
        the class as unreferenced and sends it down the hard-delete path — where the RESTRICT FK
        raises, turning a decision the reviewer should have seen into a database error.
        """
        total = 0
        with self._connection(conn) as c:
            for table, column in self._restrict_dependents(c):
                row = c.execute(
                    SQL("SELECT count(*) FROM {} WHERE {} = %s").format(
                        Identifier(table), Identifier(column)
                    ),
                    (iri,),
                ).fetchone()
                total += row[0] if row else 0
        return total

    def delete_concept(self, iri: str, *, conn=None) -> None:
        """Remove a term; surface forms, CQ links, hierarchy and domain/range rows cascade.

        Dependents outside this repo do NOT cascade — their FKs are RESTRICT, so this raises
        rather than orphaning them. Callers decide before getting here; see
        :meth:`count_external_dependents`.
        """
        with self._connection(conn) as c:
            c.execute("DELETE FROM tbox_term WHERE id = %s", (iri,))

    def rename_concept_iri(self, old_iri: str, new_iri: str, *, conn=None) -> str:
        """IRI renewal: mint the new IRI, move the term's dependents onto it, and leave the old one
        as a DEPRECATED tombstone.

        Not an UPDATE of the primary key. The graph keeps the old IRI as an ``owl:deprecated``
        node with a ``seeAlso`` pointer, so the store mirrors that instead of erasing it — and the
        FKs have no ON UPDATE CASCADE, so rewriting the key would be rejected anyway. Consumer
        views filter on ACTIVE, so the tombstone is invisible downstream while remaining auditable.
        """
        with self._connection(conn) as c:
            row = c.execute(
                "SELECT term_type, embedding_model_id, embedding_val, created_in_snapshot_id "
                "FROM tbox_term WHERE id = %s",
                (old_iri,),
            ).fetchone()
            if row is None:
                raise KeyError(f"term {old_iri} not found")
            # No ON CONFLICT: if the target IRI is already taken, moving this term's surfaces onto
            # it would silently merge two distinct concepts. Fail and let the caller report a
            # conflict instead.
            if c.execute(
                "SELECT 1 FROM tbox_term WHERE id = %s", (new_iri,)
            ).fetchone():
                raise ValueError(f"renewal target IRI already exists: {new_iri}")
            c.execute(
                "INSERT INTO tbox_term (id, term_type, status, embedding_model_id, embedding_val, "
                "created_in_snapshot_id) VALUES (%s, %s, 'ACTIVE', %s, %s, %s)",
                (new_iri, row[0], row[1], row[2], row[3]),
            )
            # Dependents follow the concept to its new IRI; the tombstone keeps only its identity.
            c.execute(
                "UPDATE surface_form SET term_id = %s WHERE term_id = %s",
                (new_iri, old_iri),
            )
            c.execute(
                "UPDATE term_cq_map SET term_id = %s WHERE term_id = %s",
                (new_iri, old_iri),
            )
            # Including the ones this repo does not own. Leaving them behind raises no error — the
            # tombstone still satisfies their FK — so the split is silent: individuals keep pointing
            # at the old IRI while the surface forms that resolve their class name move to the new
            # one, and the next load mints a second row for the same entity under the new class.
            # The identity constraint is per class, so both survive and nothing complains.
            for table, column in self._restrict_dependents(c):
                c.execute(
                    SQL("UPDATE {} SET {} = %s WHERE {} = %s").format(
                        Identifier(table), Identifier(column), Identifier(column)
                    ),
                    (new_iri, old_iri),
                )
            c.execute(
                "UPDATE tbox_term SET status = 'DEPRECATED' WHERE id = %s", (old_iri,)
            )
            return old_iri

    # --- relational topology (tbox_hierarchy / tbox_domain_range) ---

    def replace_topology(
        self, topology, *, base_iri, edge_sources=None, conn=None
    ) -> dict[str, int]:
        """Rewrite ``base_iri``'s stored structure to match ``topology`` (see
        :mod:`tboxcraft.emit.topology`).

        The graph is the source of truth for structure, so this is a full replace rather than a
        diff: the namespace's edges are cleared and re-inserted inside the caller's transaction,
        which keeps the stored edges exactly consistent with the snapshot being committed.

        The replace is scoped to one namespace because the snapshot chain is. Once a second
        document can start cold against a populated store — which is exactly what per-namespace
        chains allow — a store-wide clear would delete the first document's entire hierarchy and
        write back only the caller's, since a caller's topology contains only its own namespace.
        Scoped by the subject of the edge (``prop_id`` / ``child_id``): the document that asserts
        an edge owns it, so an edge reaching out to another namespace is still this one's to drop.

        Rows referencing an IRI that is not a persisted term are dropped (the FK would reject them)
        and **counted in the return value** — a silent drop would look like a clean load while the
        hierarchy is quietly incomplete. ``skipped_hierarchy`` is every edge the graph produced
        that the store does not hold, cycle-closing ones included, so it reconciles against the
        graph rather than against whatever survived on the way here.

        ``edge_sources`` carries the provenance to record alongside the edges::

            {"hierarchy":    {(child_iri, parent_iri): [source_id, ...]},
             "domain_range": {(prop_iri, relation, target_iri, xsd): [source_id, ...]}}

        Only edges that actually landed get their map rows, and a source_id the registry does not
        hold is dropped and counted as ``skipped_edge_sources`` rather than failing the batch on a
        foreign key — one malformed reference from an adapter should not cost the whole ingest,
        and the same "report what was dropped" rule as above applies.

        ``edge_sources`` is unioned with the provenance already recorded, not substituted for it.
        Both map tables cascade from the edge tables this method clears, so every earlier
        document's attributions are deleted on the way through and only the caller's batch would
        be written back: one warm run after a bootstrap left the hierarchy fully stored and
        entirely unattributed. A caller only knows what *its* batch asserts, and an edge an
        earlier document asserted is still asserted by it.
        """
        with self._connection(conn) as c:
            known = {r[0] for r in c.execute("SELECT id FROM tbox_term").fetchall()}
            edge_sources = self._merge_recorded_edge_sources(c, edge_sources)

            hier = [
                (ch, pa) for ch, pa in topology.hierarchy if ch in known and pa in known
            ]
            dr = [
                row
                for row in topology.domain_range
                if row[0] in known and (row[2] is None or row[2] in known)
            ]
            # Defended here too, not only where the rows are built: this method is what talks to
            # the trigger, and a caller that assembles edges another way would otherwise get the
            # exception back as a 500 instead of a count.
            hier, cycle_dropped = break_cycles(hier)
            # Against what the GRAPH had, not against what arrived. extract_topology already lifts
            # cycle-closing edges out of ``hierarchy`` into ``cycle_edges_dropped``, so differencing
            # the incoming list reported 0 skipped on a run that dropped two edges — the count said
            # the load was complete while the hierarchy was not.
            offered = len(topology.hierarchy) + len(topology.cycle_edges_dropped)
            skipped_h = offered - len(hier)
            skipped_dr = len(topology.domain_range) - len(dr)

            # starts_with(), not LIKE prefix || '%': a base IRI may legitimately carry % or _
            # (a percent-encoded value does), and those are LIKE wildcards that would widen the
            # delete past this namespace.
            prefix = namespace_prefix(base_iri)
            c.execute(
                "DELETE FROM tbox_domain_range WHERE starts_with(prop_id, %s)",
                (prefix,),
            )
            c.execute(
                "DELETE FROM tbox_hierarchy WHERE starts_with(child_id, %s)", (prefix,)
            )
            if hier:
                c.cursor().executemany(
                    "INSERT INTO tbox_hierarchy (child_id, parent_id) VALUES (%s, %s) "
                    "ON CONFLICT DO NOTHING",
                    hier,
                )
            if dr:
                c.cursor().executemany(
                    "INSERT INTO tbox_domain_range "
                    "(prop_id, relation_type, target_class_id, xsd_datatype) "
                    "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    dr,
                )
            skipped_src, unattributed = self._link_edge_sources(
                c, hier, dr, edge_sources
            )
        if skipped_h or skipped_dr:
            logger.warning(
                "topology: dropped %d hierarchy + %d domain/range rows the store cannot hold "
                "(unknown term or cycle-closing)",
                skipped_h,
                skipped_dr,
            )
        if skipped_src:
            logger.warning(
                "topology: dropped %d edge-source link(s) naming a source the registry does not "
                "hold — the edges are stored but their provenance is not",
                skipped_src,
            )
        return {
            "hierarchy": len(hier),
            "domain_range": len(dr),
            "skipped_hierarchy": skipped_h,
            "skipped_domain_range": skipped_dr,
            "skipped_edge_sources": skipped_src,
            # Edges no source asserted. Expected to be non-zero for merge-only edges; a harness
            # can still pin skipped_edge_sources at 0, which is the one that means a real fault.
            "unattributed_edges": unattributed,
            # Which edges, not just how many: a dropped cycle edge is a modelling error upstream,
            # and a count alone gives nobody anything to fix.
            "cycle_edges_dropped": [
                {"child": c, "parent": p}
                for c, p in sorted(
                    set(topology.cycle_edges_dropped) | set(cycle_dropped)
                )
            ],
        }

    def _merge_recorded_edge_sources(self, c, edge_sources) -> dict[str, dict]:
        """``edge_sources`` unioned with what the maps already hold, read before they are cleared.

        Read by natural key on both sides. ``domain_range_source_map`` points at a SERIAL that the
        replace regenerates, so the id it holds means nothing a moment later — the join back to
        ``tbox_domain_range`` is what makes the carried-over rows re-attachable.

        An edge that does not survive the replace simply never comes up again in
        :meth:`_link_edge_sources`, which is right: its provenance goes with it rather than being
        re-pinned to an edge the graph no longer asserts.
        """
        merged = {
            "hierarchy": {
                k: list(v) for k, v in (edge_sources or {}).get("hierarchy", {}).items()
            },
            "domain_range": {
                k: list(v)
                for k, v in (edge_sources or {}).get("domain_range", {}).items()
            },
        }

        def add(bucket, key, sid):
            seen = bucket.setdefault(key, [])
            if sid not in seen:
                seen.append(sid)

        for child, parent, sid in c.execute(
            "SELECT child_id, parent_id, source_id FROM hierarchy_source_map"
        ).fetchall():
            add(merged["hierarchy"], (child, parent), sid)
        for prop, rel, target, xsd, sid in c.execute(
            "SELECT d.prop_id, d.relation_type, d.target_class_id, d.xsd_datatype, m.source_id "
            "FROM domain_range_source_map m JOIN tbox_domain_range d ON d.id = m.domain_range_id"
        ).fetchall():
            add(merged["domain_range"], (prop, rel, target, xsd), sid)
        return merged

    def _link_edge_sources(self, c, hier, dr, edge_sources) -> tuple[int, int]:
        """Write the edge->source maps for the edges that landed.

        Returns ``(dropped, unattributed)``: links naming a source the registry does not hold, and
        stored edges no source asserted. The second is not a defect to drive to zero — provenance is
        asserted-only for now, so an edge that exists only as a consequence of the merge
        legitimately has no asserting source. It is counted so that "no provenance" stays visible
        instead of being indistinguishable from "provenance was never collected".

        Called from inside :meth:`replace_topology`'s transaction, after the edge tables have been
        re-inserted, so both maps refer to rows that certainly exist. The domain/range side needs a
        read-back because its key is a SERIAL the insert does not hand us per row; the table was
        just fully replaced, so that read is exactly this batch and one round trip.
        """
        if not edge_sources:
            return 0, len(hier) + len(dr)
        known = {r[0] for r in c.execute("SELECT source_id FROM source").fetchall()}
        dropped = 0
        attributed = 0

        hier_src = edge_sources.get("hierarchy") or {}
        rows = []
        for edge in hier:
            before = len(rows)
            for sid in hier_src.get(tuple(edge), ()):
                if sid in known:
                    rows.append((edge[0], edge[1], sid))
                else:
                    dropped += 1
            attributed += len(rows) > before
        if rows:
            c.cursor().executemany(
                "INSERT INTO hierarchy_source_map (child_id, parent_id, source_id) "
                "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                rows,
            )

        dr_src = edge_sources.get("domain_range") or {}
        if dr_src:
            id_by_key = {
                (p, rel, t, x): rid
                for rid, p, rel, t, x in c.execute(
                    "SELECT id, prop_id, relation_type, target_class_id, xsd_datatype "
                    "FROM tbox_domain_range"
                ).fetchall()
            }
            rows = []
            for row in dr:
                rid = id_by_key.get(tuple(row))
                if rid is None:
                    continue
                before = len(rows)
                for sid in dr_src.get(tuple(row), ()):
                    if sid in known:
                        rows.append((rid, sid))
                    else:
                        dropped += 1
                attributed += len(rows) > before
            if rows:
                c.cursor().executemany(
                    "INSERT INTO domain_range_source_map (domain_range_id, source_id) "
                    "VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    rows,
                )
        return dropped, len(hier) + len(dr) - attributed

    # --- source / CQ ---

    def insert_source_document(
        self, uri: str, title: str | None = None, *, conn=None
    ) -> str:
        """No-op: the store has no document table of its own — upstream owns document/paragraph/cq.
        The uri is returned unchanged so existing callers can keep threading it as a label."""
        return uri

    def sync_document_sources(self, *, cited_cq_ids=None, conn=None) -> int:
        """Register a document-kind ``source`` row per CQ of this batch's documents. Rows written.

        ``cited_cq_ids`` are the CQ ids this batch's components cite. The documents they belong to
        are what the batch is *about*, and every CQ of those documents is registered — not only the
        cited ones, so a CQ registered before any term cites it already has its source row when one
        does. Pass None to register the whole table (the old behaviour, kept for callers with no
        citations at all, e.g. an RDB-only batch).

        Scoping matters because the store is shared. Registering every CQ made each document's
        ``count(source)`` the store-wide total rather than the document's own, which makes the
        ``count(cq) == count(source)`` consistency check pass unconditionally and stop being able
        to see a per-document gap. The check is only worth running if it can fail.

        CQs are loaded by the upstream ingestion model, which owns ``cq`` and knows nothing about
        this spine — so the document half of the registry is *derived* here at ingest rather than
        pushed by a second endpoint. The key is the same deterministic natural key an adapter would
        compute (``doc:cq:<cq_id>``), so this and an adapter-supplied ``sources.json`` converge on
        the same rows and a re-ingest is a no-op.

        The table read is upstream's own ``cq(cq_id, question, answer, paragraph_id, document_id)``,
        not the consumer-side name ``cq_repository`` this used to assume. Reading the producer's
        name is what removes the mapping layer: a path that skips the upstream orchestrator used
        to find an empty ``cq_repository``, register zero sources, and return 200 with every term
        left unattributed — a silent failure that only surfaced downstream as an ABox FK violation.

        Call inside the persist transaction and BEFORE terms are linked: ``link_term_sources``
        drops pairs naming a source that does not exist yet, so the wrong order costs the batch its
        provenance silently rather than loudly.

        RDB sources have no equivalent derivation — they arrive via :meth:`upsert_sources`.
        """
        with self._connection(conn) as c:
            if cited_cq_ids is None:
                return c.execute(
                    "INSERT INTO source (source_id, kind, label, cq_id) "
                    "SELECT 'doc:cq:' || q.cq_id, 'document', q.question, q.cq_id "
                    "FROM cq q ON CONFLICT (source_id) DO NOTHING"
                ).rowcount
            if not cited_cq_ids:
                return 0
            return c.execute(
                "INSERT INTO source (source_id, kind, label, cq_id) "
                "SELECT 'doc:cq:' || q.cq_id, 'document', q.question, q.cq_id "
                "FROM cq q WHERE q.document_id IN ("
                "    SELECT document_id FROM cq WHERE cq_id = ANY(%s)"
                ") ON CONFLICT (source_id) DO NOTHING",
                (list(cited_cq_ids),),
            ).rowcount

    def upsert_sources(self, rows: list[dict], *, conn=None) -> int:
        """Register adapter-declared sources, returning the row count seen.

        ``source_id`` is computed by the adapter from a deterministic natural key, so this is an
        upsert and re-running an ingest is a no-op rather than a duplicate-key failure. Callers
        run it inside the ingest transaction and *before* the terms that reference these ids — the
        ordering is what keeps a term from ever pointing at an unwritten source.
        """
        if not rows:
            return 0
        with self._connection(conn) as c, c.cursor() as cur:
            cur.executemany(
                "INSERT INTO source (source_id, kind, label, run_id, cq_id, rdb_ref_id) "
                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (source_id) DO UPDATE SET "
                "label = COALESCE(EXCLUDED.label, source.label), "
                "run_id = COALESCE(EXCLUDED.run_id, source.run_id)",
                [
                    (
                        r["source_id"],
                        r["kind"],
                        r.get("label"),
                        r.get("run_id"),
                        r.get("cq_id"),
                        r.get("rdb_ref_id"),
                    )
                    for r in rows
                ],
            )
        return len(rows)

    def register_rdb_sources(self, payload: dict, *, conn=None) -> int:
        """Register rdb-kind sources with their rdb_database/rdb_table/rdb_ref leaves, in FK order.

        ``payload`` is the adapter's ``sources.annotate()`` shape::

            {"database": {dsn_label, schema_name, catalog_hash?, introspected_at?},
             "sources": [{source_id, kind, table_name, column_name?, pk_value?,
                          temporality?, evidence_text?, label?}, ...]}

        source.rdb_ref_id is NOT NULL FK to rdb_ref (CHECK: kind='rdb' requires it), so the leaves
        must exist before :meth:`upsert_sources` runs — this method chains all four in one pass.

        Leaf ids are DERIVED here, not sent: the payload carries table_name/column_name/pk_value,
        not the surrogate keys. The derivation is deterministic so a re-ingest is idempotent
        (ON CONFLICT DO NOTHING):
            rdb_database_id = rdb:<dsn_label>.<schema_name>
            rdb_table_id    = <rdb_database_id>.<table_name>
            rdb_ref_id      = source_id            (the adapter's natural key, 1:1 with a ref)
        ASSUMPTION to confirm with the adapter: this matches how source_id is composed
        (rdb:<dsn>.<schema>.<table>#...). rdb_ref_id = source_id is safe (the adapter fills
        source.rdb_ref_id with the same value); the two surrogate ids are tboxcraft-internal.
        """
        db = payload.get("database") or {}
        rows = [s for s in (payload.get("sources") or []) if s.get("kind") == "rdb"]
        if not rows:
            return 0
        if not db.get("dsn_label") or not db.get("schema_name"):
            raise ValueError(
                "rdb sources need a database block with dsn_label and schema_name"
            )
        db_id = f"rdb:{db['dsn_label']}.{db['schema_name']}"
        with self._connection(conn) as c:
            c.execute(
                "INSERT INTO rdb_database (rdb_database_id, dsn_label, schema_name, catalog_hash, "
                "introspected_at) VALUES (%s, %s, %s, %s, COALESCE(%s::timestamptz, "
                "CURRENT_TIMESTAMP)) ON CONFLICT (rdb_database_id) DO NOTHING",
                (
                    db_id,
                    db["dsn_label"],
                    db["schema_name"],
                    db.get("catalog_hash"),
                    db.get("introspected_at"),
                ),
            )
            tables = {}
            for s in rows:
                tid = f"{db_id}.{s['table_name']}"
                tables[tid] = (tid, db_id, s["table_name"], s.get("temporality"))
            with c.cursor() as cur:
                cur.executemany(
                    "INSERT INTO rdb_table (rdb_table_id, rdb_database_id, table_name, temporality) "
                    "VALUES (%s, %s, %s, %s) ON CONFLICT (rdb_table_id) DO NOTHING",
                    list(tables.values()),
                )
                cur.executemany(
                    "INSERT INTO rdb_ref (rdb_ref_id, rdb_table_id, column_name, pk_value, "
                    "evidence_text) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (rdb_ref_id) DO NOTHING",
                    [
                        (
                            s["source_id"],
                            f"{db_id}.{s['table_name']}",
                            s.get("column_name"),
                            s.get("pk_value"),
                            s.get("evidence_text"),
                        )
                        for s in rows
                    ],
                )
            self.upsert_sources(
                [
                    {
                        "source_id": s["source_id"],
                        "kind": "rdb",
                        "label": s.get("label"),
                        "rdb_ref_id": s["source_id"],
                    }
                    for s in rows
                ],
                conn=c,
            )
        return len(rows)

    def link_term_sources(self, pairs: list[tuple[str, str]], *, conn=None) -> int:
        """Link many ``(term_iri, source_id)`` provenance rows in one round trip.

        The source-neutral successor to :meth:`link_concept_cqs`. Same batching reason: this is the
        highest-volume write on the ingest path, and a per-row loop here would extend the ontology
        advisory lock the persist transaction holds.

        Pairs naming a source the registry does not hold are dropped and **returned as a count**
        rather than raising. A component may cite a CQ that was never registered (the same loss
        ``n_cq_missing`` already reports), and the foreign key would turn that ordinary input gap
        into a failed batch. Filtering here rather than at each call site keeps the one rule in one
        place — both pipelines feed this method.
        """
        if not pairs:
            return 0
        with self._connection(conn) as c, c.cursor() as cur:
            wanted = list({s for _, s in pairs})
            known = {
                r[0]
                for r in c.execute(
                    "SELECT source_id FROM source WHERE source_id = ANY(%s)", (wanted,)
                ).fetchall()
            }
            rows = [p for p in pairs if p[1] in known]
            if rows:
                cur.executemany(
                    "INSERT INTO term_source_map (term_id, source_id) VALUES (%s, %s) "
                    "ON CONFLICT DO NOTHING",
                    rows,
                )
        return len(pairs) - len(rows)

    def fetch_competency_questions(self, *, conn=None) -> list[tuple[str, str, str]]:
        """All stored CQs as ``(cq_uuid, question, answer)`` — the gloss/source_cq_ids join.

        Reads upstream's ``cq``; ``answer`` is NOT NULL there, so the COALESCE below is now only a
        guard for a store still carrying nullable rows.
        """
        with self._connection(conn) as c:
            rows = c.execute("SELECT cq_id, question, answer FROM cq").fetchall()
        return [(r[0], r[1], r[2] or "") for r in rows]

    def link_concept_cq(self, iri: str, cq_id: str, *, conn=None) -> None:
        self.link_concept_cqs([(iri, cq_id)], conn=conn)

    def link_concept_cqs(self, pairs: list[tuple[str, str]], *, conn=None) -> None:
        """Link many ``(term_iri, cq_id)`` provenance rows in one round trip.

        This is the highest-volume write in a bootstrap — a batch of a few hundred CQs produces
        thousands of links — and it was the last per-row loop left on the path. Same statement as
        the single-row form, which now delegates here so there is one INSERT to keep correct.
        """
        if not pairs:
            return
        with self._connection(conn) as c, c.cursor() as cur:
            cur.executemany(
                "INSERT INTO term_cq_map (term_id, cq_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                pairs,
            )

    # --- queries ---

    def get_concept_id(self, iri: str) -> str | None:
        """The IRI itself when the term exists, else None — the IRI *is* the key, so this is both
        the id lookup and the existence check callers use before attaching to a term."""
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT id FROM tbox_term WHERE id = %s", (iri,)
            ).fetchone()
            return row[0] if row else None

    def get_pref_label(self, iri: str, lang: str = "en") -> str | None:
        """A term's representative label for a language — verifier context."""
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT surface_text FROM surface_form "
                "WHERE term_id = %s AND lang = %s AND form_type = 'PREF' LIMIT 1",
                (iri, lang),
            ).fetchone()
            return row[0] if row else None

    def resolve_surface(self, surface: str, *, kind: str | None = None) -> str | None:
        """Exact non-representative surface -> canonical IRI (None if unknown). Case-insensitive so
        it can use the lower(surface_text) index; PREF forms are excluded (they are the label, not
        an alternative spelling).

        ``kind`` narrows the search to one term type. Names are unique per kind, not globally —
        ``hasRole`` is routinely both an object and a data property, and each mints its own IRI by
        design — so without it the same string has two owners and this returns whichever
        row comes back first. A caller that knows the kind it wants has a question with one
        answer; an individual, which can only anchor to a class, always does.
        """
        sql = (
            "SELECT term_id FROM surface_form sf JOIN tbox_term t ON t.id = sf.term_id "
            "WHERE lower(sf.surface_text) = lower(%s) AND sf.form_type <> 'PREF'"
        )
        params: tuple = (surface,)
        if kind is not None:
            sql += " AND t.term_type = %s"
            params += (_KIND_TO_DB.get(kind, kind.upper()),)
        with self.pool.connection() as conn:
            row = conn.execute(sql + " LIMIT 1", params).fetchone()
            return row[0] if row else None

    def active_surface_owner(
        self,
        surface: str,
        kind: str,
        *,
        base_iri: str,
        exclude_iri: str | None = None,
        conn=None,
    ) -> str | None:
        """An ACTIVE term of ``kind`` (other than ``exclude_iri``) this surface already resolves to.

        The mint-time uniqueness check: ACTIVE-only (the ABox loader resolves ACTIVE terms only, so
        a string owned by a DEPRECATED term is not actually taken) and inside the caller's
        transaction (a term minted earlier in this same batch counts, or two new concepts in one
        ingest could each claim the string).

        PREF rows count. resolve_surface excludes them, but this must match what the ABox loader
        actually resolves, and that query does NOT filter form_type — so a string that is another
        concept's representative label (Country PREF -> class/Country) is just as taken as one held
        as a synonym. Excluding PREF here made surface A miss exactly the PREF-vs-ALTER collisions
        that dominate the residual, and left the cold-path counter at 0. ``exclude_iri`` drops the
        candidate's own IRI so its just-written PREF does not read as a rival.
        """
        sql = (
            "SELECT sf.term_id FROM surface_form sf JOIN tbox_term t ON t.id = sf.term_id "
            "WHERE lower(sf.surface_text) = lower(%s) AND t.term_type = %s "
            "AND t.status = 'ACTIVE' AND starts_with(t.id, %s)"
        )
        params: tuple = (
            surface,
            _KIND_TO_DB.get(kind, kind.upper()),
            namespace_prefix(base_iri),
        )
        if exclude_iri is not None:
            sql += " AND t.id <> %s"
            params += (exclude_iri,)
        with self._connection(conn) as c:
            row = c.execute(sql + " LIMIT 1", params).fetchone()
            return row[0] if row else None

    def surface_claimants(
        self,
        surface: str,
        kind: str,
        *,
        base_iri: str,
        exclude_iri: str | None = None,
        conn=None,
    ) -> list[dict]:
        """Every ACTIVE term of ``kind`` holding this surface, as ``[{iri, form_type}]``.

        :meth:`active_surface_owner` answers "is this string taken", which is all the mint-time
        ALTER check needs. This answers "taken *how*", which is what the PREF side needs: a string
        another term merely lists as a synonym is not the same situation as one that is another
        term's name, and the two get opposite treatment.

        Case-insensitive and ACTIVE-only, for the same reasons as ``active_surface_owner`` — this
        has to agree with what the ABox loader actually resolves, and that query filters neither
        ``form_type`` nor case.
        """
        sql = (
            "SELECT DISTINCT sf.term_id, sf.form_type FROM surface_form sf "
            "JOIN tbox_term t ON t.id = sf.term_id "
            "WHERE lower(sf.surface_text) = lower(%s) AND t.term_type = %s "
            "AND t.status = 'ACTIVE' AND starts_with(t.id, %s)"
        )
        params: tuple = (
            surface,
            _KIND_TO_DB.get(kind, kind.upper()),
            namespace_prefix(base_iri),
        )
        if exclude_iri is not None:
            sql += " AND t.id <> %s"
            params += (exclude_iri,)
        with self._connection(conn) as c:
            rows = c.execute(sql + " ORDER BY 1, 2", params).fetchall()
        return [{"iri": r[0], "form_type": r[1]} for r in rows]

    def drop_surface_form(self, iri: str, surface: str, *, conn=None) -> int:
        """Remove a term's non-representative rows for this surface. Returns how many went.

        Never touches PREF: a term's own name is not a spelling it can give up, and the caller that
        needs this is resolving a clash *with* a PREF elsewhere — dropping one to fix another would
        just move the problem and leave a concept unlabelled.
        """
        with self._connection(conn) as c:
            cur = c.execute(
                "DELETE FROM surface_form WHERE term_id = %s AND lower(surface_text) = lower(%s) "
                "AND form_type <> 'PREF'",
                (iri, surface),
            )
            return cur.rowcount or 0

    def find_ambiguous_active_surfaces(self, *, base_iri: str, conn=None) -> list[dict]:
        """Surfaces that resolve to more than one ACTIVE term of one kind, within ``base_iri``.

        The accumulated-store view of a collision, which the warm path never recomputes: batch-time
        detection sees only the batch, so two concepts that split across documents (each carrying
        the same string) go unseen, and the ABox loader then returns an arbitrary one and drops the
        individuals. Surface uniqueness (A) stops NEW splits; this surfaces the ones already in the
        store so a human can merge, relabel, or reject.

        PREF rows count, for the same reason they count in :meth:`active_surface_owner` — the loader
        resolves them, so two concepts sharing a representative label (Location -> class/Loc and
        class/Location) is a real collision A cannot fix (a PREF is the concept's name, not a
        droppable synonym); it can only be queued. Excluding PREF here left that whole class of
        collision undetected — those surfaces surfaced nowhere.

        Each row carries the representative term's stored vector so the queue can freeze one — the
        collision is not a fresh candidate, but ``enqueue_review`` requires a vector for a gray item
        and re-embedding at decision time could land in a different space.

        Scoped to one namespace, and not as tidiness: unscoped, ``IPB#Process`` and ``ADP#Process``
        both hold the surface "Process" and this reports them as colliding. They do not. One
        document is one KG, the two terms live in different graphs, and no loader is asked to
        choose between them — so the row is a false positive that also enqueues a review item
        asking a human to merge two concepts that were never in contact. The store-wide version
        additionally made this batch's ``surface_collisions`` depend on which *other* documents
        happened to be loaded.
        """
        with self._connection(conn) as c:
            rows = c.execute(
                "SELECT s.surface, s.term_type, s.iris, t.embedding_val FROM ("
                "  SELECT lower(sf.surface_text) AS surface, t.term_type AS term_type, "
                "         array_agg(DISTINCT t.id ORDER BY t.id) AS iris, min(t.id) AS rep_id "
                "  FROM surface_form sf JOIN tbox_term t ON t.id = sf.term_id "
                "  WHERE t.status = 'ACTIVE' AND starts_with(t.id, %s) "
                "  GROUP BY lower(sf.surface_text), t.term_type "
                "  HAVING count(DISTINCT t.id) > 1"
                ") s JOIN tbox_term t ON t.id = s.rep_id ORDER BY s.surface",
                (namespace_prefix(base_iri),),
            ).fetchall()
        return [
            {
                "surface": r[0],
                "kind": _KIND_FROM_DB.get(r[1], r[1]),
                "iris": list(r[2]),
                "vector": _vec_to_np(r[3]),
            }
            for r in rows
        ]

    def resolve_anchor_by_name(
        self, name: str, kind: str, *, base_iri: str, conn=None
    ) -> str | None:
        """A live term of ``kind`` written exactly ``name`` — the attach anchor, or None.

        Not :meth:`resolve_surface` with a kind, for two reasons it cannot give up:

        * PREF rows count here. ``resolve_surface`` excludes them because a representative label is
          the concept's name rather than an alternative spelling to resolve — but naming the
          concept is exactly the match being looked for, and a single-word term has no other row
          (``desplit_camel("Activity") == "Activity"``), so excluding PREF would miss precisely the
          names most likely to recur across documents.
        * ACTIVE only. A tombstone must never become an anchor.

        A PREF match wins over an ALTER one. The name being someone's representative label and
        merely a synonym on another concept are not the same claim: if this string is exactly one
        term's PREF, that term owns it, whatever else lists it as an alias. Treating the two alike
        returned None for ``Enemy`` when a second concept happened to carry it as a synonym, so a
        re-encountered ``Enemy`` fell through to a slug collision against its own existing IRI and
        was sent to review — the name match and the collision check disagreeing about the same term.

        Still None when the name is genuinely ambiguous at the same rank: two concepts holding it
        as PREF, or (no PREF owner) two holding it as ALTER. That is the collision a human settles;
        guessing would silently pick a side, and the caller falls back to similarity.
        """
        db_kind = _KIND_TO_DB.get(kind, kind.upper())
        with self._connection(conn) as c:
            # PREF owners first: at most one is the answer regardless of who lists it as an alias.
            pref = c.execute(
                "SELECT DISTINCT sf.term_id FROM surface_form sf JOIN tbox_term t ON t.id = sf.term_id "
                "WHERE lower(sf.surface_text) = lower(%s) AND t.term_type = %s "
                "AND t.status = 'ACTIVE' AND sf.form_type = 'PREF' "
                "AND starts_with(t.id, %s) LIMIT 2",
                (name, db_kind, namespace_prefix(base_iri)),
            ).fetchall()
            if pref:
                return pref[0][0] if len(pref) == 1 else None
            # No representative owns it — fall back to any live term that carries it at all.
            rows = c.execute(
                "SELECT DISTINCT sf.term_id FROM surface_form sf JOIN tbox_term t ON t.id = sf.term_id "
                "WHERE lower(sf.surface_text) = lower(%s) AND t.term_type = %s "
                "AND t.status = 'ACTIVE' AND starts_with(t.id, %s) LIMIT 2",
                (name, db_kind, namespace_prefix(base_iri)),
            ).fetchall()
        return rows[0][0] if len(rows) == 1 else None

    def stored_terms_by_local_name(
        self, names: list[str], base_iri: str, *, conn=None
    ) -> dict[str, list[dict]]:
        """For each name, the ACTIVE terms already minted under it — ``{name: [{kind, iri, range}]}``.

        A kind conflict can straddle a batch boundary: this document sends ``usesWeapon`` as a data
        property while an earlier document already minted the object property. The in-batch scan
        never sees the stored side, so the contradiction goes unqueued. This is that missing side.

        Looked up by reconstructing the IRI with :func:`~tboxcraft.represent.mint` — the same
        function that minted it — rather than parsing ``tbox_term.id``, so a change to the minting
        rule cannot drift the two apart. That is why this reconstruction had to move when the hash
        namespace form landed: it used to build ``{base}/{kind}/{slug}`` inline, which finds nothing
        in a store whose terms are ``{base}#{Name}``, and finding nothing here reads as "no stored
        side" — a cross-batch kind conflict that silently stops being detected.

        The stored range is assembled from ``tbox_domain_range`` because it is the
        evidence the reviewer decides on (a class range on one side and xsd:string on the other are
        not the same claim); a union range is several RANGE rows, joined.

        Matched case-insensitively. An exact match asks whether the earlier document happened to
        capitalize the name the same way, which is not the question — the store resolves surfaces
        on ``lower(surface_text)`` throughout, so two terms whose names differ only in case contest
        the same string. There is no cross-batch equivalent of ``fold_case_variants`` to normalize
        them first, so an exact key simply never sees this shape. Keyed by the *stored* spelling,
        which is what the caller has to show a reviewer.
        """
        from app.tboxcraft.represent import KIND_SEGMENT, mint

        wanted = {
            mint(base_iri, kind, name).lower()
            for name in names
            for kind in KIND_SEGMENT
        }
        if not wanted:
            return {}
        out: dict[str, list[dict]] = {}
        with self._connection(conn) as c:
            live = c.execute(
                "SELECT id, term_type FROM tbox_term "
                "WHERE lower(id) = ANY(%s) AND status = 'ACTIVE'",
                (list(wanted),),
            ).fetchall()
            for iri, db_kind in live:
                name = local_of(iri)
                ranges = c.execute(
                    "SELECT target_class_id, xsd_datatype FROM tbox_domain_range "
                    "WHERE prop_id = %s AND relation_type = 'RANGE'",
                    (iri,),
                ).fetchall()
                parts = [(local_of(tc) if tc else xsd) for tc, xsd in ranges]
                out.setdefault(name, []).append(
                    {
                        "kind": _KIND_FROM_DB.get(db_kind, db_kind),
                        "iri": iri,
                        "range": ", ".join(p for p in parts if p) or None,
                    }
                )
        return out

    def nearest_concepts(
        self,
        vector: np.ndarray,
        model_id: str,
        k: int = 5,
        *,
        kind: str | None = None,
        base_iri: str | None = None,
        include_deprecated: bool = False,
    ) -> list[tuple[str, float]]:
        """Top-k terms by cosine similarity for a given model.

        ``kind`` restricts the search to one term kind — Update routing is kind-partitioned, so a
        class candidate must never match a property regardless of vector proximity. ``/resolve``
        omits it and searches all kinds. Deprecated (tombstoned) terms are excluded by default:
        attach-only clustering treats existing terms as *live* anchors, so a retired one must not
        collect new attaches or raise UPDATE proposals.
        """
        vec = np.asarray(vector, dtype=np.float32)
        sql = (
            "SELECT id, 1 - (embedding_val <=> %s) AS cosine_sim FROM tbox_term "
            "WHERE embedding_model_id = %s"
        )
        params: list = [vec, model_id]
        if not include_deprecated:
            # Written as <> 'DEPRECATED' to match the partial HNSW index predicate exactly.
            sql += " AND status <> 'DEPRECATED'"
        if kind is not None:
            sql += " AND term_type = %s"
            params.append(_KIND_TO_DB.get(kind, kind.upper()))
        if base_iri is not None:
            sql += " AND starts_with(id, %s)"
            params.append(namespace_prefix(base_iri))
        sql += " ORDER BY embedding_val <=> %s LIMIT %s"
        params += [vec, k]
        with self.pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [(iri, float(sim)) for iri, sim in rows]
