"""Application settings sourced exclusively from environment variables."""

from functools import lru_cache

from psycopg.conninfo import make_conninfo
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the lexical routing service.

    All values come from the environment. No host, port or credential is ever
    hardcoded.

    FalkorDB coordinates are deliberately absent: no code path here opens a
    graph connection, and a consumer that needs one reads its own environment.
    Responses name the graph in ``graph_id``; locating it is the caller's.
    """

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_dbname: str = "postgres"
    postgres_user: str = "postgres"
    postgres_password: str = ""

    embed_url: str | None = None
    embedding_model_id: str | None = None
    embed_timeout_s: float = 5.0
    embed_backfill_interval_s: int = 300

    pg_timeout_s: float = 2.0
    lexical_min_score: float = 0.75
    semantic_min_score: float = 0.5
    exact_variant_max_words: int = 6
    lexical_candidate_keys: int = 20
    key_index_refresh_interval_s: int = 300

    @property
    def dsn(self) -> str:
        """Build a psycopg connection string from the Postgres settings.

        Returns:
            A libpq conninfo string, with each value quoted/escaped by
            ``psycopg.conninfo.make_conninfo`` so passwords or other fields
            containing spaces, backslashes or quotes cannot corrupt the DSN.
        """
        return make_conninfo(
            host=self.postgres_host,
            port=self.postgres_port,
            dbname=self.postgres_dbname,
            user=self.postgres_user,
            password=self.postgres_password,
        )

    @property
    def semantic_enabled(self) -> bool:
        """Whether the semantic tier is configured.

        Both the TEI endpoint and the model id are required: without the model id
        the service cannot tell which stored vectors belong to the current
        embedding space.
        """
        return bool(self.embed_url) and bool(self.embedding_model_id)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (FastAPI dependency)."""
    return Settings()
