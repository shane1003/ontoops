"""Runtime configuration for tboxcraft, loaded from environment / ``.env``.

Config is a plain dataclass so it is trivial to construct in tests without touching
the environment. :meth:`Config.from_env` is the production entry point.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from app.core.db import ENV_VAR as DATABASE_URL_ENV_VAR


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass
class Config:
    """Resolved runtime settings.

    Defaults keep the pipeline local-first: every provider runs on-host or against
    an on-stack service, so real data cannot be routed off-box by accident.
    """

    # Embedding provider selection.
    embedding_provider: str = "e5"  # "e5" (in-process) | "remote_e5" (TEI, production)
    e5_model: str = "intfloat/multilingual-e5-large-instruct"
    # Where the in-process e5 weights (safetensors) are cached. A repo-local, gitignored dir so
    # the model downloads ONCE and is reused, never committed/pushed. (Docker uses the TEI
    # `modelcache` volume instead of this.)
    hf_cache_dir: str = ".hf_cache"

    # Domain-term verifier LLM. Local only.
    # llm_url points at the local vLLM OpenAI-compatible chat server.
    # A wrong address degrades to "verifier unavailable" instead of failing visibly.
    llm_provider: str = "local"
    llm_url: str = "http://localhost:8000"
    llm_model: str = "local-model"
    # The domain the verifier judges "official term" against, injected into its prompt. "Official"
    # is domain-relative — the model needs the domain named to interpret it and to reach for the
    # right prior knowledge — but which domain is not the pipeline's business. Neutral by default
    # so tboxcraft is general-purpose; set TBOXCRAFT_DOMAIN_DESCRIPTION to specialize a deployment.
    domain_description: str = "specialized-domain"

    # Ontology identity.
    base_iri: str = "http://example.org/military"
    version_iri: str = "http://example.org/military/0.1.0"
    license_iri: str = "https://creativecommons.org/licenses/by/4.0/"

    # Threshold seeds; provider-specific, overridden by calibration when possible.
    tau_high_seed: float = 0.85
    tau_low_seed: float = 0.55
    # Major-Update threshold: IRI renewal is proposed only at similarity >= tau_major,
    # set >= tau_high so renewals fire only on near-certain same-concept matches.
    tau_major: float = 0.90
    # How much cosine this pipeline treats as a meaningful difference, used only on the gray/new
    # boundary (the tau_high merge gate never consults it).
    #
    # tau_low cannot decide that boundary. It is the 95th percentile of similarity between two
    # RANDOM candidates, and it is compared against the nearest of every stored term — a maximum
    # over a set that grows with every document. A per-pair quantile against an order statistic
    # drifts apart as the store grows, which is why the gray share climbed steeply over a few
    # documents while the threshold sat still. No value of tau_low fixes that; the statistic it is
    # compared against moves.
    #
    # So the boundary asks a scale-free question instead: top-1 must lead top-2 by this much to
    # count as a real neighbour rather than background. In a large store top-1 and top-2 rise
    # together, so their difference does not.
    #
    # There is deliberately no absolute companion to this. One was tried — attach anything within
    # this distance of tau_high, whatever the neighbourhood looked like — and over successive
    # documents its attaches grew steeply while this rule stayed flat: an absolute threshold
    # compared against a maximum over a growing store drifts, which is the whole defect. Any future
    # exception has to be scale-free too, or it will reintroduce it.
    attach_margin: float = 0.05
    # Confidence floor for auto-hiding a confidently-attached word as a HiddenTerm: the verifier
    # must return is_standard_form=False AND confidence >= this floor to bury the term; an
    # uncertain "non-standard" verdict keeps it a safe AlterTerm. The verdict is advisory and
    # can hallucinate on niche jargon, so a low-confidence drop is not trusted.
    hidden_confidence_floor: float = 0.7
    # How many verifier calls may be in flight at once. Verdicts are independent per
    # concept, so the batch was bounded by round-trip latency × concept count, not by the
    # server: a sequential bootstrap held the LLM at a concurrency of 1 for its whole run.
    # Higher concurrency was many times faster against one GPU with the outputs unchanged, so
    # this is a conservative floor, not a ceiling — raise it to fill the server. 1 is strictly
    # sequential.
    verify_concurrency: int = 8
    # How many candidates share one verifier call. Unlike verify_concurrency this is not a
    # pure speedup: the model sees a term's neighbours while judging it, so verdicts can differ
    # from a one-at-a-time run. Each candidate still gets its own verdict — the batch reply is
    # keyed per candidate — so nothing about what a verdict *means* changes. 1 disables batching
    # and reproduces the un-batched prompt exactly, which is how the two are compared.
    verify_cluster_size: int = 8

    # Reasoner. None → owlready2 picks java from PATH.
    java_exe: str | None = None
    # Whether HermiT runs when a request does not say. Consistency is a hard gate only if it is
    # actually attempted: with this off, `consistent` stays None and gate() has nothing to refuse,
    # so a TBox commits with its consistency never proven. It exists because tests and local runs
    # need a path that does not require a JVM — not as an operational choice.
    reasoner_enabled: bool = True

    # OOPS! scan. Advisory, and a network call that posts the **whole ontology inline** — the
    # TBox is not published at a resolvable URI, so `OntologyURI` is empty and the RDF/XML travels
    # in the request body, Korean `rdfs:comment` rationale included.
    #
    # No default endpoint, for the same reason `database_url` two lines down has no default: a
    # default recipient for a whole-ontology POST is one nobody has to choose, and this used to
    # carry the public host while the comment above it said to point it somewhere internal
    # *before* enabling the scan. The comment stated a precondition the code then broke, and
    # "enabled by default" made deploy order the only thing standing between an ingest and a
    # third party. Unset now means the scan is skipped and says so (`oops_error` in the
    # validation block) rather than sent somewhere nobody picked. The repository-root stack disables
    # the scan (TBOXCRAFT_OOPS_ENABLED=false).
    oops_endpoint: str | None = None
    oops_enabled: bool = True

    # Database. No committed static-password default — fail closed when unset.
    database_url: str | None = None

    # Remote embedding server (TEI) — used by the remote_e5 provider.
    embedding_url: str = "http://localhost:8601"
    embedding_revision: str = "main"

    # Update-mode gray zone; top-k adjudicator candidates.
    gray_zone_k: int = 5

    # /resolve acceptance threshold: the minimum cosine similarity (in the stored centered
    # space) at which a k-NN hit is returned as a confident IRI match. Server-owned so callers
    # never re-judge it (that would diverge from the space the TBox was built in). Below it,
    # /resolve reports "no confident match" and returns the candidates for inspection only.
    # NOTE: name-vs-gloss embeddings sit in a different distribution than the gloss-vs-gloss
    # merge band, so this needs tuning against real Stage-2 traffic — treat the default as a
    # conservative starting point, not a calibrated value.
    resolve_threshold: float = 0.80

    # Langfuse tracing (optional). Enabled only when all three are set AND base_url is a
    # loopback/private host — a public host is refused so gloss/CQ text never leaves the box.
    langfuse_base_url: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None

    # I/O.
    output_dir: str = "data/output"

    # CQ sampling bound for gloss assembly.
    max_cq_sample: int = 5

    _dotenv_loaded: bool = field(default=False, repr=False)

    @classmethod
    def from_env(cls, dotenv_path: str | None = None) -> Config:
        """Build a :class:`Config` from environment variables (loading ``.env`` first)."""
        load_dotenv(dotenv_path=dotenv_path, override=False)
        return cls(
            embedding_provider=os.environ.get(
                "TBOXCRAFT_EMBEDDING_PROVIDER", cls.embedding_provider
            )
            .strip()
            .lower(),
            e5_model=os.environ.get("TBOXCRAFT_E5_MODEL", cls.e5_model),
            hf_cache_dir=os.environ.get("TBOXCRAFT_HF_CACHE_DIR", cls.hf_cache_dir),
            llm_provider=os.environ.get("TBOXCRAFT_LLM_PROVIDER", cls.llm_provider)
            .strip()
            .lower(),
            llm_url=os.environ.get("TBOXCRAFT_LLM_URL", cls.llm_url),
            llm_model=os.environ.get("TBOXCRAFT_LLM_MODEL", cls.llm_model),
            domain_description=os.environ.get(
                "TBOXCRAFT_DOMAIN_DESCRIPTION", cls.domain_description
            ),
            base_iri=os.environ.get("TBOXCRAFT_BASE_IRI", cls.base_iri),
            version_iri=os.environ.get("TBOXCRAFT_VERSION_IRI", cls.version_iri),
            license_iri=os.environ.get("TBOXCRAFT_LICENSE_IRI", cls.license_iri),
            tau_high_seed=_get_float("TBOXCRAFT_TAU_HIGH_SEED", cls.tau_high_seed),
            tau_low_seed=_get_float("TBOXCRAFT_TAU_LOW_SEED", cls.tau_low_seed),
            tau_major=_get_float("TBOXCRAFT_TAU_MAJOR", cls.tau_major),
            attach_margin=_get_float("TBOXCRAFT_ATTACH_MARGIN", cls.attach_margin),
            hidden_confidence_floor=_get_float(
                "TBOXCRAFT_HIDDEN_CONFIDENCE_FLOOR", cls.hidden_confidence_floor
            ),
            verify_concurrency=int(
                os.environ.get("TBOXCRAFT_VERIFY_CONCURRENCY", cls.verify_concurrency)
            ),
            verify_cluster_size=int(
                os.environ.get("TBOXCRAFT_VERIFY_CLUSTER_SIZE", cls.verify_cluster_size)
            ),
            database_url=os.environ.get(DATABASE_URL_ENV_VAR) or None,
            embedding_url=os.environ.get("TBOXCRAFT_EMBEDDING_URL", cls.embedding_url),
            embedding_revision=os.environ.get(
                "TBOXCRAFT_EMBEDDING_REVISION", cls.embedding_revision
            ),
            java_exe=os.environ.get("TBOXCRAFT_JAVA_EXE") or None,
            # `or cls...` so an empty/whitespace value is unset rather than a blank URL: a
            # half-filled deploy template must fail closed like a missing one.
            oops_endpoint=(os.environ.get("TBOXCRAFT_OOPS_ENDPOINT", "") or "").strip()
            or cls.oops_endpoint,
            reasoner_enabled=_get_bool(
                "TBOXCRAFT_REASONER_ENABLED", cls.reasoner_enabled
            ),
            oops_enabled=_get_bool("TBOXCRAFT_OOPS_ENABLED", cls.oops_enabled),
            gray_zone_k=int(os.environ.get("TBOXCRAFT_GRAY_ZONE_K", cls.gray_zone_k)),
            resolve_threshold=_get_float(
                "TBOXCRAFT_RESOLVE_THRESHOLD", cls.resolve_threshold
            ),
            langfuse_base_url=os.environ.get("LANGFUSE_BASE_URL") or None,
            langfuse_public_key=os.environ.get("LANGFUSE_PUBLIC_KEY") or None,
            langfuse_secret_key=os.environ.get("LANGFUSE_SECRET_KEY") or None,
            output_dir=os.environ.get("TBOXCRAFT_OUTPUT_DIR", cls.output_dir),
            max_cq_sample=int(
                os.environ.get("TBOXCRAFT_MAX_CQ_SAMPLE", cls.max_cq_sample)
            ),
            _dotenv_loaded=True,
        )


# The validation rule lives in one repository-wide place (app.core.iri): it must be the
# same rule as aboxmapper's, so a namespace that passed one stage is never rejected by
# the next. The existing import path (config.validate_base_iri) is kept as a re-export.
from app.core.iri import validate_base_iri as validate_base_iri  # noqa: E402
