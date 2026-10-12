"""conceptualizer router: exposes the classification stage as a synchronous endpoint.

- ``POST /extract``: CQ document → annotated TBox components + ABox individuals.
  No TTL.

Import-safe contract: importing this module alone needs no LLM key or network.
Settings are loaded once at module load (load_settings); LLM availability is
checked at request time.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.core.config import Settings, load_settings
from app.conceptualizer.schemas import CQDocument, ExtractionOutput
from app.conceptualizer.service.runner import build_extraction

log = logging.getLogger("conceptualizer.api")

# Message for a missing LLM (contains no secrets).
_NO_LLM_MSG = (
    "No LLM key is configured. This endpoint needs an LLM for the expert/identity "
    "merge. Set OPENAI_API_KEY (a dummy value works for a local server)."
)


def _load_settings_safe() -> Settings:
    """Loads the settings, degrading to a bare Settings on failure (stays import-safe).

    Import and startup must not break even when config/settings.yaml is missing or
    unparsable. LLM availability is checked separately at request time.
    """
    try:
        return load_settings()
    except Exception as exc:  # noqa: BLE001 — degrade safely on any load failure.
        log.warning("Settings load failed, degrading to default Settings: %s", exc)
        return Settings(root=Path("."))


router = APIRouter()

# Settings load once at startup (not reloaded per request). The LLM key is
# checked per request.
settings = _load_settings_safe()


def _error(status: int, message: str) -> JSONResponse:
    """Builds a standard error response that carries no secrets."""
    return JSONResponse(status_code=status, content={"detail": message})


@router.get("/target")
def target() -> dict:
    """Reports the domain this instance judges by, as a runtime fact.

    The request document's ontology.domain wins as the classifier's actual domain,
    so the value here is the **default** used when the document domain is empty
    (the same naming philosophy as default_base_iri in tboxcraft /target).
    It reads the loaded settings, not an env echo: a wrong domain is a silent defect
    that only lowers judging quality without any exception or warning, so the
    harness needs a place to check it against the runtime fact.
    """
    return {"default_domain": settings.agents.domain}


@router.post(
    "/extract",
    response_model=None,
    responses={200: {"model": ExtractionOutput}},
)
def extract(
    # Request body = CQ document (same shape that load_cq_document parses).
    body: CQDocument,
    chunk_size: int | None = Query(
        default=None, description="settings.extraction.chunk_size override"
    ),
    language: str | None = Query(
        default=None,
        description=(
            "Fallback language of the domain-layer text (correct_term, notes) "
            "(BCP-47, e.g. ko). Used only for values the classifier gave no "
            "language for; without this too, they go out as und."
        ),
    ),
) -> ExtractionOutput:
    """Turns a CQ document into annotated TBox components + ABox individuals.

    No TTL is generated. DomainSpecificClassifier uses an LLM, so a key is required
    (503 without one).

    `language` used to be declared and then discarded: callers believed
    `?language=ko` was applied while the value silently vanished. It is now
    actually used as the fallback for the domain-layer language (whether it arrived
    is visible in the response's `correct_term_lang`/`notes_lang`).
    """
    if not settings.llm_available():
        return _error(503, _NO_LLM_MSG)

    output = build_extraction(body, settings, chunk_size=chunk_size, language=language)
    # ExtractionOutput = {components, individuals}. No TTL field.
    return output
