"""Client of the ontoagent service: extraction, TBox, ABox and graph.

Every write names its namespace by the IRI pieces and the assembled base IRI
together, so the server cross-checks them and refuses a drifted rule before it
stores anything. The response checks after each call are the production
flow's: a namespace that did not arrive, CQs the TBox
could not link, a failed gate or an empty graph stop the cell.
"""

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from evaluation.clients.common import ServiceError, send

SERVICE = "ontoagent"
_WHITESPACE_RUN = re.compile(r"\s+")
# Production: preflight 3 x 10 s, TBox and ABox 3 x 30 s on 502/503 (never on
# a 504: the update may still be running).
PREFLIGHT_ATTEMPTS, PREFLIGHT_WAIT_S = 3, 10.0
WRITE_ATTEMPTS, WRITE_WAIT_S = 3, 30.0


@dataclass(frozen=True)
class Namespace:
    """The IRI pieces one cell writes under.

    The assembly repeats the server's rule (``app.core.iri``) only so that
    both forms can be sent; the server compares them and answers 422 when
    they differ, which is how a drift in either copy is caught.

    Attributes:
        root: IRI root, e.g. ``http://example.org``.
        domain: One path segment, e.g. ``doctrine``.
        document: The cell's document name, e.g. ``ipb-r0``.
    """

    root: str
    domain: str
    document: str

    def _segment(self) -> str:
        return _WHITESPACE_RUN.sub("_", self.document.strip())

    @property
    def base_iri(self) -> str:
        """The namespace every term of the cell is minted under."""
        encoded = quote(self._segment(), safe="-_.!~*'()")
        return f"{self.root.rstrip('/')}/{self.domain}/{encoded}#"

    @property
    def graph(self) -> str:
        """The FalkorDB graph the cell's TBox is loaded into."""
        return f"{self.domain}/{self._segment()}"

    def params(self) -> dict[str, str]:
        """Query parameters naming the namespace both ways."""
        return {
            "base_iri": self.base_iri,
            "iri_root": self.root,
            "iri_domain": self.domain,
            "document": self.document,
        }


class ContractError(ServiceError):
    """A 2xx answer that the production checks reject."""


class OntoagentClient:
    """Calls the ontoagent service.

    Attributes:
        base_url: Service root, e.g. ``http://127.0.0.1:8082``.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 10800.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self.base_url, timeout=timeout_s, transport=transport
        )
        self._sleep = sleep

    def close(self) -> None:
        """Releases the connection pool."""
        self._http.close()

    def _send(self, endpoint: str, call: Any, **retry: Any) -> httpx.Response:
        if self._sleep is not None:
            retry["sleep"] = self._sleep
        return send(call, service=SERVICE, endpoint=endpoint, **retry)

    def preflight(self) -> dict[str, Any]:
        """Reads the service's targets and checks that both stores agree.

        Returns:
            The raw answers of ``/health``, ``/target``, ``/tbox/target``,
            ``/abox/ready`` and ``/abox/target``, for the cell's record.

        Raises:
            ContractError: If the TBox and ABox sides use different databases.
        """
        answers: dict[str, Any] = {}
        for path in (
            "/health",
            "/target",
            "/tbox/target",
            "/abox/ready",
            "/abox/target",
        ):
            answers[path] = self._send(
                path,
                lambda p=path: self._http.get(p),
                retry_statuses=(500, 502, 503, 504),
                retry_transport=True,
                attempts=PREFLIGHT_ATTEMPTS,
                wait_s=lambda attempt: PREFLIGHT_WAIT_S,
            ).json()
        tbox = answers["/tbox/target"].get("postgres") or {}
        abox = answers["/abox/target"].get("postgres") or {}
        if tbox.get("dbname") != abox.get("dbname") or not tbox.get("dbname"):
            raise ContractError(
                SERVICE, "/abox/target", f"TBox store {tbox} != ABox store {abox}"
            )
        return answers

    def extract(self, document: dict[str, Any], *, chunk_size: int) -> dict[str, Any]:
        """Turns a CQ document into annotated components and individuals.

        Raises:
            ContractError: If no component came back.
        """
        body: dict[str, Any] = self._send(
            "/extract",
            lambda: self._http.post(
                "/extract", json=document, params={"chunk_size": chunk_size}
            ),
        ).json()
        if not body.get("components"):
            raise ContractError(SERVICE, "/extract", "no component was extracted")
        return body

    def tbox_update(
        self, components: list[dict[str, Any]], namespace: Namespace, *, domain: str
    ) -> dict[str, Any]:
        """Builds and stores the TBox; returns the full response.

        Raises:
            ContractError: If the namespace or domain did not arrive, some
                cited CQ is missing from the store, no CQ was linked, the gate
                failed, or no Turtle came back.
        """
        endpoint = "/tbox/update"
        body = {
            "components": components,
            "run_reasoner": True,
            # OOPS! has no service in the stack; it only advises.
            "run_oops": False,
            "verify": True,
        }
        tbox: dict[str, Any] = self._send(
            endpoint,
            lambda: self._http.post(
                endpoint, params={**namespace.params(), "domain": domain}, json=body
            ),
            retry_statuses=(502, 503),
            attempts=WRITE_ATTEMPTS,
            wait_s=lambda attempt: WRITE_WAIT_S,
        ).json()
        _check_namespace(endpoint, tbox, namespace)
        if tbox.get("domain_source") != "request":
            raise ContractError(
                SERVICE, endpoint, f"domain not applied: {tbox.get('domain_source')}"
            )
        if tbox.get("n_cq_missing"):
            raise ContractError(
                SERVICE,
                endpoint,
                f"{tbox['n_cq_missing']} cited CQ(s) not found in the store",
            )
        if tbox.get("n_cq_linked") == 0:
            raise ContractError(SERVICE, endpoint, "no term is linked to any CQ")
        if tbox.get("ok") is not True:
            raise ContractError(SERVICE, endpoint, f"gate: {tbox.get('gate_reasons')}")
        if not tbox.get("ttl"):
            raise ContractError(SERVICE, endpoint, "no Turtle in the response")
        return tbox

    def abox_load(
        self, individuals: list[dict[str, Any]], namespace: Namespace
    ) -> dict[str, Any]:
        """Stores the individuals and their data values.

        Raises:
            ContractError: If the namespace did not arrive or the service
                received a different number of individuals than were sent.
        """
        endpoint = "/abox/load"
        abox: dict[str, Any] = self._send(
            endpoint,
            lambda: self._http.post(
                endpoint,
                params={**namespace.params(), "dry_run": "false"},
                json={"individuals": individuals},
            ),
            retry_statuses=(502, 503),
            attempts=WRITE_ATTEMPTS,
            wait_s=lambda attempt: WRITE_WAIT_S,
        ).json()
        _check_namespace(endpoint, abox, namespace)
        if abox.get("individuals") != len(individuals):
            raise ContractError(
                SERVICE,
                endpoint,
                f"received {abox.get('individuals')} individuals, "
                f"sent {len(individuals)}",
            )
        return abox

    def graph_upload(self, ttl: str, namespace: Namespace) -> dict[str, Any]:
        """Loads the TBox Turtle into the cell's FalkorDB graph.

        Raises:
            ContractError: If the graph is empty or is not the cell's graph.
        """
        endpoint = "/abox/graph/tbox/upload"
        report: dict[str, Any] = self._send(
            endpoint,
            lambda: self._http.post(
                endpoint,
                params={
                    "graph": namespace.graph,
                    "iri_domain": namespace.domain,
                    "document": namespace.document,
                },
                files={"file": ("tbox.ttl", ttl.encode("utf-8"), "text/turtle")},
            ),
        ).json()
        if not report.get("classes"):
            raise ContractError(SERVICE, endpoint, f"graph has no class: {report}")
        if report.get("graph") != namespace.graph:
            raise ContractError(
                SERVICE,
                endpoint,
                f"wrote graph {report.get('graph')!r}, expected {namespace.graph!r}",
            )
        return report


def _check_namespace(endpoint: str, body: dict[str, Any], namespace: Namespace) -> None:
    if body.get("base_iri") != namespace.base_iri:
        raise ContractError(
            SERVICE,
            endpoint,
            f"namespace {body.get('base_iri')!r} != sent {namespace.base_iri!r}",
        )
    if body.get("base_iri_source") != "request":
        raise ContractError(
            SERVICE, endpoint, f"base_iri_source is {body.get('base_iri_source')!r}"
        )
    echoed = body.get("base_iri_pieces") or {}
    sent = {
        "root": namespace.root,
        "domain": namespace.domain,
        "document": namespace.document,
    }
    drift = {k: (v, echoed.get(k)) for k, v in sent.items() if echoed.get(k) != v}
    if drift:
        raise ContractError(SERVICE, endpoint, f"IRI pieces drifted: {drift}")
