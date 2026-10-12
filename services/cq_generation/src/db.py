"""PostgreSQL loading.

Table definitions: the DDL is sql/schema.sql, the row models are src/schemas.py.

What outside code uses is the "load operations" below (save_*/load_*/replace_*). These
functions open and close their own transactions, so callers never need to see
connect(). connect/insert_models above are the primitives those functions stand on, and
are also used by one-off scripts.

The transaction is held by connect()'s with block: passing through the block normally
commits, an exception rolls back (psycopg's default behavior). To bind several tables in
one transaction, call insert_models several times with one conn (replace_cq binds
delete+insert that way).

Caution: even if a constraint violation is caught *inside* the with block, the
transaction is already in a failed state, so the whole block rolls back. "Skip only the
failed item and continue" works only if that item is moved into its own block.
"""

import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import LiteralString, cast

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from config import settings
from src.mcs_chunker import Chunk
from src.schemas import CQDraftRow, CQRow, DraftCQ, ParagraphRow, Row

# The value itself is arbitrary, but every instance attached to this DB must use the
# same value to be serialized.
_SCHEMA_LOCK_KEY = 0x0C95CE7A


@contextmanager
def connect(advisory_lock_key: int | None = None) -> Generator[psycopg.Connection]:
    """Connects with settings.postgres_dsn.

    Args:
        advisory_lock_key: If given, the transaction's first statement takes an
            exclusive advisory lock on this key. It serializes with other connections
            using the same key and is released when the transaction ends.
    """
    with psycopg.connect(
        settings.postgres_dsn, connect_timeout=settings.postgres_connect_timeout
    ) as conn:
        if advisory_lock_key is not None:
            conn.execute(
                "SELECT pg_advisory_xact_lock(%s)", (advisory_lock_key,)
            )
        yield conn


def insert_models(conn: psycopg.Connection, rows: Sequence[Row]) -> int:
    """INSERTs *Row models into their table and returns the number of rows inserted.

    Takes only one table at a time: executemany reuses the query built once, so mixed
    models would put values into the wrong columns. Hence it is blocked up front.
    """
    if not rows:
        return 0

    model = type(rows[0])
    for i, row in enumerate(rows):
        if type(row) is not model:
            raise ValueError(
                f"Only one table at a time: rows[0]={model.__name__}, "
                f"rows[{i}]={type(row).__name__}"
            )

    columns = list(model.model_fields)
    query = sql.SQL("INSERT INTO {table} ({columns}) VALUES ({placeholders})").format(
        table=sql.Identifier(model.TABLE),
        columns=sql.SQL(", ").join(sql.Identifier(c) for c in columns),
        placeholders=sql.SQL(", ").join(sql.Placeholder(c) for c in columns),
    )

    with conn.cursor() as cur:
        cur.executemany(query, [_params(row) for row in rows])
        return cur.rowcount


def _params(row: Row) -> dict:
    """Converts a *Row into bind parameters, wrapping dict/list in jsonb.

    psycopg3 does not turn a Python dict into jsonb on its own and dies with "cannot
    adapt type 'dict'": the jsonb column (cq_draft.refine_check_flag) must be wrapped in
    Jsonb(). Wrapping on the model side would make *Row no longer pure data, so it is
    done at this boundary where data goes out to SQL.
    """
    return {
        k: Jsonb(v) if isinstance(v, (dict, list)) else v
        for k, v in row.model_dump().items()
    }


def _delete_cq(conn: psycopg.Connection, cq_id: str) -> int:
    """Deletes the row of cq_id and returns the number deleted."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM cq WHERE cq_id = %s", (cq_id,))
        return cur.rowcount


def create_tables() -> None:
    """Runs sql/schema.sql (all IF NOT EXISTS, so it may be called many times).

    Notes:
        IF NOT EXISTS guarantees only *sequential* re-runs. psycopg sends a query
        without parameters over the simple protocol, so the whole schema.sql becomes one
        transaction, and inside it CREATE INDEX (ShareLock) and COMMENT ON
        (ShareUpdateExclusiveLock) reverse the acquisition order per table, so
        concurrent runs deadlock.
    """
    # psycopg accepts str queries only as LiteralString (a type guard against SQL built
    # from strings). The input here is not user input but a DDL file in the repo, so it
    # is passed through with cast.
    ddl = cast(LiteralString, settings.db_schema_path.read_text(encoding="utf-8"))
    with connect(advisory_lock_key=_SCHEMA_LOCK_KEY) as conn:
        conn.execute(ddl)


# ---------------------------------------------------------------------------
# Load operations - pipeline outputs -> rows. The transaction boundary is here.
#
# Callers (the router) do not see connect(). Still, *when* the transaction opens is
# decided by the caller, because these functions are short and do not call the LLM.
# Conversely, pushing this loading into cq_generator/cq_refiner would keep the
# transaction open for as long as the LLM latency.
#
# FK violations are re-raised as MissingReference, so the router can map them to 400
# without importing psycopg. Which status code to answer is still decided by the side
# that knows HTTP.
# ---------------------------------------------------------------------------


class MissingReference(Exception):
    """The referenced parent row (document/paragraph) is missing, so loading failed."""


def save_paragraphs(
    chunks: Sequence[Chunk], document_id: str, run_id: str
) -> list[str]:
    """Loads chunks as paragraph rows and returns paragraph_ids in chunk order."""
    rows = [to_paragraph_row(c, document_id, run_id) for c in chunks]
    try:
        with connect() as conn:
            insert_models(conn, rows)
    except psycopg.errors.ForeignKeyViolation as e:
        raise MissingReference(f"Unregistered document_id: {document_id}") from e
    return [r.paragraph_id for r in rows]


def save_cqs(
    cqs: Sequence[DraftCQ], document_id: str, paragraph_id: str, run_id: str
) -> list[str]:
    """Loads CQs and returns their cq_ids."""
    rows = [to_cq_row(c, document_id, paragraph_id, run_id) for c in cqs]
    try:
        with connect() as conn:
            insert_models(conn, rows)
    except psycopg.errors.ForeignKeyViolation as e:
        raise MissingReference(
            f"Unregistered document_id/paragraph_id: {document_id}/{paragraph_id}"
        ) from e
    return [r.cq_id for r in rows]


def load_paragraph(paragraph_id: str) -> ParagraphRow | None:
    """Reads one paragraph. None if absent."""
    columns = list(ParagraphRow.model_fields)
    query = sql.SQL("SELECT {columns} FROM paragraph WHERE paragraph_id = %s").format(
        columns=sql.SQL(", ").join(sql.Identifier(c) for c in columns)
    )
    with connect() as conn, conn.cursor() as cur:
        cur.execute(query, (paragraph_id,))
        row = cur.fetchone()
    return ParagraphRow(**dict(zip(columns, row))) if row else None


def load_cq(cq_id: str) -> CQRow | None:
    """Reads one CQ. None if absent."""
    columns = list(CQRow.model_fields)
    query = sql.SQL("SELECT {columns} FROM cq WHERE cq_id = %s").format(
        columns=sql.SQL(", ").join(sql.Identifier(c) for c in columns)
    )
    with connect() as conn, conn.cursor() as cur:
        cur.execute(query, (cq_id,))
        row = cur.fetchone()
    return CQRow(**dict(zip(columns, row))) if row else None


def replace_cq(
    old_cq_id: str,
    cqs: Sequence[DraftCQ],
    document_id: str,
    paragraph_id: str,
    run_id: str,
) -> list[str]:
    """Deletes old_cq_id and inserts cqs. Returns the inserted cq_ids.

    When refine splits a CQ, new ids appear, and when it discards one, nothing remains,
    so UPDATE cannot handle it.
    """
    rows = [to_cq_row(c, document_id, paragraph_id, run_id) for c in cqs]
    with connect() as conn:
        _delete_cq(conn, old_cq_id)
        insert_models(conn, rows)
    return [r.cq_id for r in rows]


def load_draft(draft_id: str) -> CQDraftRow | None:
    """Reads one cq_draft. None if absent."""
    columns = list(CQDraftRow.model_fields)
    query = sql.SQL("SELECT {columns} FROM cq_draft WHERE draft_id = %s").format(
        columns=sql.SQL(", ").join(sql.Identifier(c) for c in columns)
    )
    with connect() as conn, conn.cursor() as cur:
        cur.execute(query, (draft_id,))
        row = cur.fetchone()
    return CQDraftRow(**dict(zip(columns, row))) if row else None


def load_drafts_by_run(run_id: str) -> list[CQDraftRow]:
    """Reads every cq_draft belonging to run_id (for the gate review)."""
    columns = list(CQDraftRow.model_fields)
    query = sql.SQL("SELECT {columns} FROM cq_draft WHERE run_id = %s ORDER BY draft_id").format(
        columns=sql.SQL(", ").join(sql.Identifier(c) for c in columns)
    )
    with connect() as conn, conn.cursor() as cur:
        cur.execute(query, (run_id,))
        rows = cur.fetchall()
    return [CQDraftRow(**dict(zip(columns, row))) for row in rows]


def _update_draft(draft_id: str, assignments: str, params: tuple) -> bool:
    """UPDATEs one cq_draft row in place and returns whether the row existed.

    Uses UPDATE only (no DELETE+INSERT): the promotion invariant is built on the row
    count of cq_draft, so if a row disappears, promote rolls back on an
    expected/actual mismatch.
    """
    # assignments only ever receives the literals the functions below put in (not user
    # input). All values are parameter-bound. Passed through with cast for the same
    # reason as create_tables.
    query = cast(LiteralString, f"UPDATE cq_draft SET {assignments} WHERE draft_id = %s")
    with connect() as conn, conn.cursor() as cur:
        cur.execute(query, (*params, draft_id))
        return cur.rowcount > 0


def update_draft_review(
    draft_id: str, action: str | None, feedback: str | None
) -> bool:
    """Writes the review decision (reviewer_action/feedback).

    None resets it to "unreviewed".
    """
    return _update_draft(
        draft_id, "reviewer_action = %s, reviewer_feedback = %s", (action, feedback)
    )


def update_draft_verdict(draft_id: str, verdict: str | None) -> bool:
    """Writes the verdict on a regeneration (regeneration_verdict).

    Only 'reject' is left out of promotion.
    """
    return _update_draft(draft_id, "regeneration_verdict = %s", (verdict,))


def update_draft_regenerated(draft_id: str, question: str, answer: str) -> bool:
    """Writes the regenerated question/answer and stamps regenerated_at.

    The values before overwriting are moved to original_*: the reviewer needs the
    original to see what changed. COALESCE is used because a second regeneration must
    not overwrite the original with the first regeneration's result (the original is
    always the sentence a person first saw).
    """
    return _update_draft(
        draft_id,
        "original_question = COALESCE(original_question, question),"
        " original_answer = COALESCE(original_answer, answer),"
        " question = %s, answer = %s, regenerated_at = NOW()",
        (question, answer),
    )


# Promotability predicate. The SELECT counting the expected number, the INSERT and the
# verification SELECT use the same string: written separately in three places, fixing
# only one would make them diverge silently.
#
# Why IS DISTINCT FROM: in SQL, NULL != 'drop' is NULL, not FALSE, and WHERE keeps only
# TRUE rows, so with != every unanswered (NULL) row drops out. The rule is unanswered =
# accept, so most of them would silently fall out of promotion.
#
# Rows asked to be regenerated but not yet regenerated (regenerated_at IS NULL) are also
# excluded. A sentence the reviewer called "unusable as is" cannot be delivered as the
# original. Only verdict='accept' is the exception: an explicit decision to use the
# original without regeneration.
_PROMOTABLE = (
    "reviewer_action IS DISTINCT FROM 'drop' "
    "AND regeneration_verdict IS DISTINCT FROM 'reject' "
    # IN without COALESCE drops every unreviewed row whose reviewer_action is NULL
    # (NULL IN (...) is NULL, and so is NOT NULL).
    "AND NOT (COALESCE(reviewer_action, '') IN ('rewrite', 'reanswer') "
    "AND regenerated_at IS NULL "
    "AND regeneration_verdict IS DISTINCT FROM 'accept')"
)


def _count(cur: psycopg.Cursor, query: LiteralString, params: tuple) -> int:
    """Reads one COUNT(*).

    An aggregate query returns one row even with no rows, so there is no None.
    """
    cur.execute(query, params)
    row = cur.fetchone()
    assert row is not None
    return row[0]


def promote_drafts(run_id: str) -> int:
    """Promotes cq_draft → cq in a single transaction.

    The invariant looks not at "how many were inserted this time" (rowcount) but at "how
    many exist afterwards": because of ON CONFLICT DO NOTHING, a re-run has rowcount 0,
    so checking rowcount would blow up on every normal re-run. Checking the final state,
    the first run and a re-run give the same value.

    The expected/actual comparison recounts the set the INSERT made with the same
    predicate, so it is closer to confirming the INSERT did not die. What actually
    filters is leaked, which looks outside that set.

    Raises:
        RuntimeError: If the result differs from the expectation; the transaction then
            rolls back.
    """
    with connect() as conn:
        with conn.cursor() as cur:
            total = _count(
                cur, "SELECT COUNT(*) FROM cq_draft WHERE run_id = %s", (run_id,)
            )

            expected = _count(
                cur,
                f"SELECT COUNT(*) FROM cq_draft WHERE run_id = %s AND {_PROMOTABLE}",
                (run_id,),
            )

            cur.execute(
                f"""
                INSERT INTO cq (cq_id, document_id, paragraph_id, run_id, question, answer)
                SELECT draft_id, document_id, paragraph_id, run_id, question, answer
                FROM cq_draft
                WHERE run_id = %s AND {_PROMOTABLE}
                ON CONFLICT (cq_id) DO NOTHING
                """,
                (run_id,),
            )

            actual = _count(
                cur,
                f"""
                SELECT COUNT(*)
                FROM cq_draft d JOIN cq c ON c.cq_id = d.draft_id
                WHERE d.run_id = %s AND {_PROMOTABLE}
                """,
                (run_id,),
            )

            leaked = _count(
                cur,
                f"""
                SELECT COUNT(*)
                FROM cq_draft d JOIN cq c ON c.cq_id = d.draft_id
                WHERE d.run_id = %s AND NOT ({_PROMOTABLE})
                """,
                (run_id,),
            )

        if actual != expected:
            raise RuntimeError(
                f"promote_drafts promotion count mismatch: expected {expected}, "
                f"actual {actual}"
            )
        if leaked:
            raise RuntimeError(
                f"promote_drafts non-promotable rows found in cq: {leaked} "
                f"(run_id={run_id}). These rows turned drop/reject after promotion; "
                f"they remain after rollback, so delete those draft_ids from cq and "
                f"run again."
            )

    if total != expected:
        logging.warning(
            f"[promote] promoted {expected} / excluded {total - expected} "
            f"(run_id={run_id}). Exclusion reason is one of drop, reject or "
            f"not regenerated."
        )
    return actual


def to_paragraph_row(chunk: Chunk, document_id: str, run_id: str) -> ParagraphRow:
    """Maps one chunk to one paragraph row.

    A ParagraphRef inside a chunk cannot become a row: it has no text.
    """
    pages = [p for ref in chunk.paragraphs for p in ref.page]
    return ParagraphRow(
        document_id=document_id,
        run_id=run_id,
        section_path=" > ".join(chunk.section_path),
        # Page range over every source the chunk covers. A chunk with no source (a
        # heading-only section) gets 0: the column is NOT NULL so it cannot be empty,
        # and there is no other value meaning "no page".
        page_start=min(pages, default=0),
        page_end=max(pages, default=0),
        seq=chunk.index,
        text=chunk.text,
    )


def to_cq_row(cq: DraftCQ, document_id: str, paragraph_id: str, run_id: str) -> CQRow:
    """Maps a DraftCQ to a cq row.

    flag (the verification result) has no place in the table, so it is dropped.
    """
    return CQRow(
        cq_id=cq.id,
        document_id=document_id,
        paragraph_id=paragraph_id,
        run_id=run_id,
        question=cq.question,
        answer=cq.answer,
    )


def to_draft_row(cq: DraftCQ, document_id: str, paragraph_id: str, run_id: str) -> CQDraftRow:
    """Maps a DraftCQ to a cq_draft row. flag is stored in refine_check_flag."""
    return CQDraftRow(
        draft_id=cq.id,
        document_id=document_id,
        paragraph_id=paragraph_id,
        run_id=run_id,
        question=cq.question,
        answer=cq.answer,
        refine_check_flag=cq.flag.model_dump() if cq.flag else None,
    )


def save_drafts(
    cqs: Sequence[DraftCQ], document_id: str, paragraph_id: str, run_id: str
) -> list[str]:
    """Loads CQ drafts into cq_draft and returns their draft_ids."""
    rows = [to_draft_row(c, document_id, paragraph_id, run_id) for c in cqs]
    try:
        with connect() as conn:
            insert_models(conn, rows)
    except psycopg.errors.ForeignKeyViolation as e:
        raise MissingReference(
            f"Unregistered document_id/paragraph_id: {document_id}/{paragraph_id}"
        ) from e
    return [r.draft_id for r in rows]


def _delete_draft(conn: psycopg.Connection, draft_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM cq_draft WHERE draft_id = %s", (draft_id,))
        return cur.rowcount


def replace_draft(
    old_draft_id: str,
    cqs: Sequence[DraftCQ],
    document_id: str,
    paragraph_id: str,
    run_id: str,
) -> list[str]:
    """Deletes old_draft_id and inserts cqs into cq_draft (for refine split/discard)."""
    rows = [to_draft_row(c, document_id, paragraph_id, run_id) for c in cqs]
    with connect() as conn:
        _delete_draft(conn, old_draft_id)
        insert_models(conn, rows)
    return [r.draft_id for r in rows]


def load_promoted_ids(run_id: str) -> list[str]:
    """Reads the cq_ids of promoted rows in paragraph order.

    This order is exactly the terminologize input order and the order of the final
    CQ output. It must be fixed so that two runs (e.g. with and without the review
    gate) can be diffed; without ORDER BY there is no ground to rely on the order
    postgres returns.

    paragraph_id cannot be sorted on: it is a UUID unrelated to the order within the
    document, so paragraph.seq is JOINed in.

    "Promoted" means rows in cq that satisfy _PROMOTABLE. Leaking an id absent from cq
    makes terminologize get a 404 from load_cq, and cq has no delete path, so looking at
    the JOIN alone would also pull in rows dropped/rejected after promotion.
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT d.draft_id
            FROM cq_draft d
            JOIN cq c        ON c.cq_id = d.draft_id
            JOIN paragraph p ON p.paragraph_id = d.paragraph_id
            WHERE d.run_id = %s AND {_PROMOTABLE}
            ORDER BY p.seq, d.draft_id
            """,
            (run_id,),
        )
        return [row[0] for row in cur.fetchall()]
