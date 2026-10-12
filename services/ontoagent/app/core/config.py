"""Settings loader: reads config/settings.yaml + .env into a single Settings object.

All paths are made absolute against the repository root (= the parent of the directory
holding settings.yaml).

conceptualizer transition notes:
 - HermiT/OOPS gating and TTL generation moved to the external ontology system, so
   OntologySettings/ValidationSettings were removed entirely (leftovers nothing read).
 - ExtractionSettings (chunk_size) is wired into load_settings. Settings reads only the
   declared dataclass fields, so without the field the YAML extraction.* keys would be
   silently ignored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class LLMSettings:
    """Settings for the local LLM (OpenAI-compatible server) only.

    The server address is given by OPENAI_BASE_URL.
    """

    # Main model (reasoning-heavy coder/correction) kept separate from the fast model
    # (light work such as review and subject derivation).
    openai_model: str = "gpt-4o"
    openai_fast_model: str = "gpt-4o-mini"
    temperature: float = 0.0

    def model_for(self, fast: bool) -> str:
        """Returns the model ID for the tier (the light model when fast=True)."""
        return self.openai_fast_model if fast else self.openai_model


@dataclass
class PathSettings:
    intermediate_dir: Path
    prompts_dir: Path


@dataclass
class ExtractionSettings:
    """Stage 1 chunking settings.

    A large CQList is split into units of chunk_size; expert reads fan out/fan in and are
    merged back before the single external hand-off (under one batch ID).

    Keep chunk_size small: the experts JSON response for all components of one chunk must
    fit in the LLM max_tokens (8192); beyond that the response is truncated, leading to
    parse failure -> degrade (verdicts lost -> 0 individuals). CQs carry many components,
    so it is set conservatively.
    """

    chunk_size: int = 10


@dataclass
class AgentSettings:
    """Domain setting for the experiment stage."""

    domain: str = "military"


@dataclass
class Settings:
    root: Path
    llm: LLMSettings = field(default_factory=LLMSettings)
    paths: PathSettings | None = None
    extraction: ExtractionSettings = field(default_factory=ExtractionSettings)
    agents: AgentSettings = field(default_factory=AgentSettings)

    def __post_init__(self) -> None:
        # When paths is not given, fill in defaults relative to root
        # (so .paths access is safe on bare Settings()/Settings(root=...) too).
        if self.paths is None:
            self.paths = PathSettings(
                intermediate_dir=self.root / "data" / "intermediate",
                prompts_dir=self.root / "config" / "prompts",
            )

    # --- API key (.env) ---
    @property
    def openai_api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY") or None

    def llm_available(self) -> bool:
        """Tells whether real LLM calls are possible (a key is set).

        A local server needs a dummy key too.
        """
        return bool(self.openai_api_key)


def load_settings(config_path: str | Path = "config/settings.yaml") -> Settings:
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent  # config/settings.yaml -> repo root
    load_dotenv(root / ".env")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    llm_raw = raw.get("llm", {})
    paths_raw = raw.get("paths", {})
    extraction_raw = raw.get("extraction", {})
    agents_raw = raw.get("agents", {})

    def abspath(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (root / p)

    def env_or(env_name: str, yaml_value):
        """Uses the environment variable if set (empty string excluded), else the YAML value.

        Lets containers/compose swap models without baking settings.yaml.
        Precedence: env > settings.yaml > default.
        """
        return os.getenv(env_name) or yaml_value

    return Settings(
        root=root,
        llm=LLMSettings(
            openai_model=env_or(
                "CONCEPTUALIZER_OPENAI_MODEL", llm_raw.get("openai_model", "gpt-4o")
            ),
            openai_fast_model=env_or(
                "CONCEPTUALIZER_OPENAI_FAST_MODEL",
                llm_raw.get("openai_fast_model", "gpt-4o-mini"),
            ),
            temperature=float(llm_raw.get("temperature", 0.0)),
        ),
        paths=PathSettings(
            intermediate_dir=abspath(paths_raw.get("intermediate_dir", "data/intermediate")),
            prompts_dir=abspath(paths_raw.get("prompts_dir", "config/prompts")),
        ),
        extraction=ExtractionSettings(
            chunk_size=int(extraction_raw.get("chunk_size", 10)),
        ),
        agents=AgentSettings(
            # The domain is a property of the corpus (deployment unit), not the document,
            # so the env layer fits. Of the modules, only this file was yaml-only, a hard
            # block that forced an image rebuild to switch to a non-military corpus; this
            # lifts it. A request
            # parameter waits until one deployment really serves two domains at once.
            domain=env_or("CONCEPTUALIZER_DOMAIN", agents_raw.get("domain", "military")),
        ),
    )
