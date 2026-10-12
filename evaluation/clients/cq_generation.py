"""Client of the CQ generation service.

One method per endpoint the production flow calls, in flow order: parse the
PDF with opendataloader, find the skeleton, chunk, generate per paragraph,
refine per draft, promote, terminologize per CQ. The retry and skip rules are
the production flow's, not choices made here.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from evaluation.clients.common import ServiceError, send

SERVICE = "cq_generation"
# generate: three attempts, a missing paragraph (404) is skipped, not retried.
GENERATE_ATTEMPTS = 3
# refine: only 503 means "could not converge, original kept"; three tries.
REFINE_ATTEMPTS = 3


@dataclass(frozen=True)
class Structure:
    """Result of the skeleton step.

    Attributes:
        content: The structure file to upload with the chunking call, or None
            when the service could not build one.
        skipped_reason: Why there is no structure; production then chunks by
            coordinates and numbering alone.
    """

    content: bytes | None
    skipped_reason: str | None


@dataclass(frozen=True)
class Refined:
    """Result of refining one draft.

    Attributes:
        cqs: The drafts that replaced it (``cq_id`` and ``flag``); empty when
            the draft was dropped, several when it was split.
        kept_original: True when the service answered 503 three times and the
            original draft stays as it was, unflagged.
    """

    cqs: list[dict[str, Any]]
    kept_original: bool


class CQGenerationClient:
    """Calls the CQ generation service.

    Attributes:
        base_url: Service root, e.g. ``http://127.0.0.1:8081``.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 900.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self.base_url, timeout=timeout_s, transport=transport
        )
        self._sleep = sleep

    def close(self) -> None:
        """Releases the connection pool."""
        self._http.close()

    def parser_odl(self, pdf: bytes, filename: str) -> bytes:
        """Converts a PDF with opendataloader; returns the ODL JSON bytes."""
        response = send(
            lambda: self._http.post(
                "/api/v1/parser-odl",
                files={"file": (filename, pdf, "application/pdf")},
            ),
            service=SERVICE,
            endpoint="/api/v1/parser-odl",
        )
        return response.content

    def structure_odl(self, odl: bytes) -> Structure:
        """Builds the document skeleton; any failure means "chunk without one"."""
        try:
            response = send(
                lambda: self._http.post(
                    "/api/v1/structure-odl",
                    files={"file": ("parsed.odl.json", odl, "application/json")},
                ),
                service=SERVICE,
                endpoint="/api/v1/structure-odl",
            )
        except ServiceError as exc:
            # Production's flow treats any failure here (no headings: 422; an
            # LLM error, including a window too long for the context: 500) as
            # "no skeleton" and chunks by coordinates and numbering; the
            # skeleton is a best-effort refinement.
            return Structure(None, str(exc))
        return Structure(response.content, None)

    def chunk_odl(
        self,
        odl: bytes,
        structure: bytes | None,
        *,
        document_id: str,
        run_id: str,
        budget: int,
    ) -> list[str]:
        """Chunks into paragraph rows; returns their ids in reading order."""
        files: dict[str, tuple[str, bytes, str]] = {
            "file": ("parsed.odl.json", odl, "application/json")
        }
        if structure is not None:
            files["structure"] = ("structure.json", structure, "application/json")
        response = send(
            lambda: self._http.post(
                "/api/v1/chunk-odl",
                params={"budget": budget, "document_id": document_id, "run_id": run_id},
                files=files,
            ),
            service=SERVICE,
            endpoint="/api/v1/chunk-odl",
        )
        ids: list[str] = response.json()["paragraph_ids"]
        return ids

    def generate(
        self, *, purpose: str, scope: str, paragraph_id: str, run_id: str
    ) -> list[str]:
        """Generates draft CQs from one paragraph; returns their ids.

        Raises:
            ServiceError: After three failed attempts, or at once on a 404.
        """
        response = send(
            lambda: self._http.post(
                "/api/v1/generate",
                json={
                    "purpose": purpose,
                    "scope": scope,
                    "paragraph_id": paragraph_id,
                    "run_id": run_id,
                },
            ),
            service=SERVICE,
            endpoint="/api/v1/generate",
            retry_statuses=tuple(s for s in range(400, 600) if s != 404),
            retry_transport=True,
            attempts=GENERATE_ATTEMPTS,
            wait_s=lambda attempt: float(attempt),
            sleep=self._sleep,
        )
        ids: list[str] = response.json()["cq_ids"]
        return ids

    def drafts(self, run_id: str) -> list[dict[str, Any]]:
        """Lists every draft of a run."""
        response = send(
            lambda: self._http.get("/api/v1/drafts", params={"run_id": run_id}),
            service=SERVICE,
            endpoint="/api/v1/drafts",
        )
        rows: list[dict[str, Any]] = response.json()
        return rows

    def refine(self, cq_id: str, run_id: str) -> Refined:
        """Refines one draft until it converges."""
        try:
            response = send(
                lambda: self._http.post(
                    "/api/v1/refine", params={"cq_id": cq_id, "run_id": run_id}
                ),
                service=SERVICE,
                endpoint="/api/v1/refine",
                retry_statuses=(503,),
                attempts=REFINE_ATTEMPTS,
                sleep=self._sleep,
            )
        except ServiceError as exc:
            if exc.status == 503:
                return Refined([{"cq_id": cq_id, "flag": None}], kept_original=True)
            raise
        cqs: list[dict[str, Any]] = response.json()["cqs"]
        return Refined(cqs, kept_original=False)

    def promote(self, run_id: str) -> list[str]:
        """Promotes the run's drafts to CQs; returns the CQ ids in order."""
        response = send(
            lambda: self._http.post(
                "/api/v1/promote-drafts", params={"run_id": run_id}
            ),
            service=SERVICE,
            endpoint="/api/v1/promote-drafts",
        )
        ids: list[str] = response.json()
        return ids

    def terminologize(self, cq_id: str) -> dict[str, Any]:
        """Annotates one CQ with classes and properties."""
        response = send(
            lambda: self._http.post("/api/v1/terminologize", params={"cq_id": cq_id}),
            service=SERVICE,
            endpoint="/api/v1/terminologize",
        )
        cq: dict[str, Any] = response.json()
        return cq


def cq_document(
    cqs: list[dict[str, Any]],
    *,
    base_iri: str,
    prefix: str,
    purpose: str,
    scope: str,
    domain: str,
) -> dict[str, Any]:
    """Wraps terminologized CQs the way the production merge step does.

    ``domain`` is added to the ontology block so the extraction judges terms
    against the run's domain explicitly rather than a container default.
    """
    return {
        "ontology": {
            "iri": base_iri,
            "prefix": prefix,
            "purpose": purpose,
            "scope": scope,
            "domain": domain,
        },
        "source": prefix,
        "competency_questions": cqs,
    }
