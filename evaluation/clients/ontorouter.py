"""Client of the OntoRouter lookup service."""

from typing import Any, Literal

import httpx

from evaluation.clients.common import send

SERVICE = "ontorouter"
BATCH_LIMIT = 100  # the service rejects larger batches with 422

Tiers = Literal["exact", "lexical", "semantic"]


class OntorouterClient:
    """Calls the lookup service.

    Attributes:
        base_url: Service root, e.g. ``http://127.0.0.1:8083``.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self.base_url, timeout=timeout_s, transport=transport
        )

    def close(self) -> None:
        """Releases the connection pool."""
        self._http.close()

    def lookup(
        self, terms: list[str], *, k: int = 5, tiers: Tiers = "lexical"
    ) -> list[dict[str, Any]]:
        """Looks terms up, in batches the service accepts.

        Args:
            terms: Query terms, in order.
            k: Suggestions per channel.
            tiers: How deep the service goes.

        Returns:
            One answer per term, in input order.
        """
        answers: list[dict[str, Any]] = []
        for start in range(0, len(terms), BATCH_LIMIT):
            batch = terms[start : start + BATCH_LIMIT]

            def call(batch: list[str] = batch) -> httpx.Response:
                return self._http.post(
                    "/lookup", params={"k": k, "tiers": tiers}, json={"terms": batch}
                )

            response = send(
                call,
                service=SERVICE,
                endpoint="/lookup",
            )
            items: list[dict[str, Any]] = response.json()["items"]
            answers.extend(items)
        return answers
