"""Database access layer (Postgres + pgvector)."""

from app.tboxcraft.db.pool import build_pool, shared_pool
from app.tboxcraft.db.repository import Repository

__all__ = ["Repository", "build_pool", "shared_pool"]
