"""Direct store access: register a document, export one cell's rows.

The CQ generation service has no document registration endpoint; production
inserts the row itself, and so does this module, with the same statement. The
export copies one cell out of Postgres and FalkorDB so that scoring reads files
only.

The export is scoped differently from production's on purpose:

* CQ-side tables by ``run_id``. The document row is shared by every run of the
  same PDF (it is keyed by content hash), so a ``document_id`` scope mixes the
  repetitions of one manual.
* Every paragraph of the run, not only those that produced a CQ: paragraph
  coverage needs the denominator.
* TBox and ABox tables by the cell's base IRI, as production does; each cell
  has its own namespace.

Vector columns are left out, as in production: they are larger than the rest
and the terms they embed are already in the term rows.
"""

import hashlib
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import psycopg
from falkordb import FalkorDB
from psycopg import sql
from psycopg.rows import dict_row

_RUN = "run_id = %(run_id)s"
_IRI_TERM = "starts_with({col}, %(iri)s)"
_CQS = "SELECT cq_id FROM cq WHERE run_id = %(run_id)s"
_IND_WHERE = (
    f"(source_id IN (SELECT source_id FROM source WHERE cq_id IN ({_CQS}))"
    " OR starts_with(class_id, %(iri)s))"
)
_DR_WHERE = (
    "(starts_with(prop_id, %(iri)s)"
    " OR (target_class_id IS NOT NULL AND starts_with(target_class_id, %(iri)s)))"
)
_USED_SOURCES = (
    f"SELECT source_id FROM abox_individual WHERE {_IND_WHERE}"
    " UNION SELECT source_id FROM abox_data_assertion WHERE subject_id IN"
    f" (SELECT id FROM abox_individual WHERE {_IND_WHERE})"
    " UNION SELECT source_id FROM term_source_map WHERE starts_with(term_id, %(iri)s)"
    " UNION SELECT source_id FROM hierarchy_source_map"
    " WHERE starts_with(child_id, %(iri)s) OR starts_with(parent_id, %(iri)s)"
    " UNION SELECT source_id FROM domain_range_source_map WHERE domain_range_id IN"
    f" (SELECT id FROM tbox_domain_range WHERE {_DR_WHERE})"
    f" UNION SELECT source_id FROM source WHERE cq_id IN ({_CQS})"
)

# Table, row filter, order. The order makes two exports of the same cell
# byte-identical.
SCOPES: tuple[tuple[str, str, str], ...] = (
    (
        "document",
        "document_id IN (SELECT document_id FROM paragraph WHERE run_id = %(run_id)s)",
        "document_id",
    ),
    ("paragraph", _RUN, "seq"),
    ("cq_draft", _RUN, "draft_id"),
    ("cq", _RUN, "cq_id"),
    ("ontology_snapshot", "base_iri = %(iri)s", "created_at"),
    (
        "embedding_model",
        "id IN (SELECT embedding_model_id FROM tbox_term"
        " WHERE starts_with(id, %(iri)s))",
        "id",
    ),
    ("tbox_term", "starts_with(id, %(iri)s)", "id"),
    ("surface_form", "starts_with(term_id, %(iri)s)", "term_id, surface_text"),
    ("source", f"source_id IN ({_USED_SOURCES})", "source_id"),
    ("term_cq_map", "starts_with(term_id, %(iri)s)", "term_id, cq_id"),
    ("term_source_map", "starts_with(term_id, %(iri)s)", "term_id, source_id"),
    (
        "tbox_hierarchy",
        "starts_with(child_id, %(iri)s) OR starts_with(parent_id, %(iri)s)",
        "child_id, parent_id",
    ),
    (
        "hierarchy_source_map",
        "starts_with(child_id, %(iri)s) OR starts_with(parent_id, %(iri)s)",
        "child_id, parent_id, source_id",
    ),
    ("tbox_domain_range", _DR_WHERE, "id"),
    (
        "domain_range_source_map",
        f"domain_range_id IN (SELECT id FROM tbox_domain_range WHERE {_DR_WHERE})",
        "domain_range_id, source_id",
    ),
    ("abox_individual", _IND_WHERE, "id"),
    (
        "abox_data_assertion",
        f"subject_id IN (SELECT id FROM abox_individual WHERE {_IND_WHERE})",
        "id",
    ),
    ("review_queue", "base_iri = %(iri)s", "id"),
)

# Scalars only: node and edge objects serialize differently across client
# versions, and a stored export must not depend on the library.
GRAPH_QUERIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("class", "MATCH (c:Class) RETURN c.iri, c.name ORDER BY c.iri", ("iri", "name")),
    (
        "subclass_of",
        "MATCH (a:Class)-[:SUBCLASS_OF]->(b:Class) RETURN a.iri, b.iri"
        " ORDER BY a.iri, b.iri",
        ("child", "parent"),
    ),
    (
        "object_prop",
        "MATCH (a:Class)-[r:OBJECT_PROP]->(b:Class) RETURN a.iri, r.iri, r.name, b.iri"
        " ORDER BY a.iri, r.iri, b.iri",
        ("domain", "iri", "name", "range"),
    ),
)


class StoreError(RuntimeError):
    """The store does not hold what the cell should have written."""


def stack_dsn(env: dict[str, str]) -> str:
    """The host-side DSN of the stack's Postgres, from the ``.env`` values."""
    port = env.get("POSTGRES_PORT", "55432")
    return f"postgresql://ontoops:{env['POSTGRES_PASSWORD']}@127.0.0.1:{port}/ontoops"


def register_document(dsn: str, pdf: bytes, name: str) -> str:
    """Upserts the document row exactly as production does.

    Args:
        dsn: Postgres DSN of the stack.
        pdf: The prepared PDF the services will see.
        name: Human-readable document name.

    Returns:
        The document id: new, or the existing one for the same bytes.
    """
    content_hash = hashlib.sha256(pdf).hexdigest()
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            "INSERT INTO document"
            " (document_id, document_name, location, content_hash, processed_at)"
            " VALUES (%s, %s, %s, %s, now())"
            " ON CONFLICT (content_hash) DO UPDATE"
            " SET document_name = EXCLUDED.document_name,"
            " location = EXCLUDED.location, processed_at = EXCLUDED.processed_at"
            " RETURNING document_id",
            (str(uuid.uuid4()), name, name, content_hash),
        ).fetchone()
    if row is None:
        raise StoreError("document upsert returned no row")
    return str(row[0])


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, uuid.UUID | bytes | memoryview):
        return str(value)
    return value


def _columns(conn: psycopg.Connection[Any], table: str) -> list[str]:
    # Read from the catalog, never listed here: the upstream schemas grow by
    # ADD COLUMN, and a hard-coded list would drop a new column silently.
    rows = conn.execute(
        "SELECT column_name, udt_name FROM information_schema.columns"
        " WHERE table_schema = 'public' AND table_name = %s"
        " ORDER BY ordinal_position",
        (table,),
    ).fetchall()
    return [name for name, udt in rows if udt != "vector"]


def export_rows(
    dsn: str, *, run_id: str, base_iri: str
) -> dict[str, list[dict[str, Any]]]:
    """Copies one cell's rows out of Postgres.

    Args:
        dsn: Postgres DSN of the stack.
        run_id: The cell's CQ generation run.
        base_iri: The cell's namespace.

    Returns:
        ``{table: rows}`` for every table in ``SCOPES``, rows as plain dicts
        of JSON-safe values, in a stable order.
    """
    params = {"run_id": run_id, "iri": base_iri}
    out: dict[str, list[dict[str, Any]]] = {}
    with psycopg.connect(dsn) as conn:
        for table, where, order in SCOPES:
            columns = _columns(conn, table)
            query = sql.SQL(
                "SELECT {cols} FROM {table} WHERE {where} ORDER BY {order}"
            ).format(
                cols=sql.SQL(", ").join(sql.Identifier(c) for c in columns),
                table=sql.Identifier(table),
                where=sql.SQL(where),
                order=sql.SQL(order),
            )
            with conn.cursor(row_factory=dict_row) as cur:
                rows = cur.execute(query, params).fetchall()
            out[table] = [{k: _jsonable(v) for k, v in row.items()} for row in rows]
    return out


def export_graph(host: str, port: int, graph: str) -> dict[str, list[dict[str, Any]]]:
    """Copies one cell's graph out of FalkorDB.

    Raises:
        StoreError: If the graph key does not exist. A MATCH on a missing
            graph creates an empty one, which would turn "wrong instance"
            into "empty graph", so existence is checked first.
    """
    db = FalkorDB(host=host, port=port)
    if not db.connection.exists(graph):
        raise StoreError(f"FalkorDB has no graph {graph!r}")
    handle = db.select_graph(graph)
    out: dict[str, list[dict[str, Any]]] = {}
    for name, query, fields in GRAPH_QUERIES:
        result = handle.ro_query(query).result_set
        out[name] = [dict(zip(fields, row, strict=True)) for row in result]
    return out


class StackStore:
    """The running stack's Postgres and FalkorDB, as a cell sees them.

    Attributes:
        dsn: Host-side Postgres DSN.
        falkordb: ``(host, port)`` of FalkorDB.
    """

    def __init__(self, dsn: str, falkordb: tuple[str, int]) -> None:
        self.dsn = dsn
        self.falkordb = falkordb

    def register(self, pdf: bytes, name: str) -> str:
        """Upserts the document row; returns its id."""
        return register_document(self.dsn, pdf, name)

    def export(
        self, run_id: str, base_iri: str, graph: str
    ) -> dict[str, list[dict[str, Any]]]:
        """One cell's rows and graph; graph tables are prefixed ``graph_``."""
        rows = export_rows(self.dsn, run_id=run_id, base_iri=base_iri)
        for name, items in export_graph(*self.falkordb, graph).items():
            rows[f"graph_{name}"] = items
        return rows
