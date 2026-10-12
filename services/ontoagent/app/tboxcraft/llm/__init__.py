"""LLM provider factory for the domain-term verifier.

Selects a provider from config. Local only: there is no hosted provider, so there is
nothing here that sends text off-box.
"""

from __future__ import annotations

from app.tboxcraft.config import Config
from app.tboxcraft.llm.base import LLMProvider, Verdict


def make_llm(config: Config) -> LLMProvider:
    provider = config.llm_provider.strip().lower()
    if provider == "local":
        from app.tboxcraft.llm.local import LocalLLMProvider

        return LocalLLMProvider(config.llm_url, config.llm_model)
    raise ValueError(f"Unknown LLM provider: {config.llm_provider!r}")


__all__ = ["LLMProvider", "Verdict", "make_llm"]
