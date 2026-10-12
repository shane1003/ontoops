"""Application of the checked-in DDL at service start.

The DDL files under ``db/ddl`` remain the single definition of ontorouter's
schema; this module only runs them. Every statement in them is written
``IF NOT EXISTS``, so applying them on every start is idempotent and a service
that finds its schema already in place does nothing.

Running them here rather than leaving it to an operator removes a deployment
step that is easy to forget and whose omission is invisible until the first
semantic lookup. The cost is that the service's database role needs CREATE on
the database. Where it does not have that — a least-privilege deployment — the
application fails, is logged, and the service still starts: the exact and
lexical tiers do not touch the ontorouter schema, so they keep working, and the
semantic tier degrades through the usual channel-drop path.
"""

import logging
import pathlib
from contextlib import AbstractContextManager
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class SchemaTarget(Protocol):
    """The database surface applying DDL needs: one connection, uncommitted."""

    def session(self) -> AbstractContextManager[Any]: ...


DDL_DIR = pathlib.Path(__file__).resolve().parents[2] / "db" / "ddl"


def apply_ddl(db: SchemaTarget, ddl_dir: pathlib.Path = DDL_DIR) -> list[str]:
    """Apply every checked-in DDL file, in filename order.

    Args:
        db: The connection pool wrapper.
        ddl_dir: Directory holding the ``*.sql`` files.

    Returns:
        The names of the files applied, in the order they ran.

    Raises:
        OSError: If ``ddl_dir`` cannot be read.
        Exception: Whatever the driver raises when a statement fails — most
            often insufficient privilege. Callers decide whether that is fatal.
    """
    applied: list[str] = []
    # Sorted so a later file may depend on an earlier one; the numeric prefix
    # in the filenames is what carries that order.
    for path in sorted(ddl_dir.glob("*.sql")):
        with db.session() as conn:
            conn.execute(path.read_text())
            conn.commit()
        applied.append(path.name)
    return applied
