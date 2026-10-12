"""Embedding provider factory.

Selects a provider from config. Local only: there is no hosted provider — gloss text
stays on-host, so there is nothing here that sends it off-box.
"""

from __future__ import annotations

from app.tboxcraft.config import Config
from app.tboxcraft.embed.base import EmbeddingProvider, l2_normalize


def make_provider(config: Config) -> EmbeddingProvider:
    provider = config.embedding_provider.strip().lower()
    if provider == "e5":
        from app.tboxcraft.embed.e5 import E5EmbeddingProvider

        return E5EmbeddingProvider(
            config.e5_model,
            revision=config.embedding_revision,
            cache_folder=config.hf_cache_dir,
        )
    if provider == "remote_e5":
        from app.tboxcraft.embed.remote_e5 import RemoteE5Provider

        return RemoteE5Provider(
            config.embedding_url,
            revision=config.embedding_revision,
            model_name=config.e5_model,
        )
    raise ValueError(f"Unknown embedding provider: {config.embedding_provider!r}")


__all__ = ["EmbeddingProvider", "l2_normalize", "make_provider"]
