"""What every service client shares: one error type, retries, a worker pool.

The retry rules are not generic policy: each caller passes the statuses and
attempt counts the production flow uses for that endpoint, so a run behaves
the way the deployed pipeline does when a service hiccups.
"""

import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx


class ServiceError(RuntimeError):
    """A service call failed or answered something the contract forbids.

    Attributes:
        service: Which service: ``cq_generation``, ``ontoagent`` or ``ontorouter``.
        endpoint: The path that failed.
        status: HTTP status, or None when no response arrived.
        timed_out: True when the call hit its deadline.
    """

    def __init__(
        self,
        service: str,
        endpoint: str,
        message: str,
        *,
        status: int | None = None,
        timed_out: bool = False,
    ) -> None:
        super().__init__(f"{service} {endpoint}: {message}")
        self.service = service
        self.endpoint = endpoint
        self.status = status
        self.timed_out = timed_out


def send(
    call: Callable[[], httpx.Response],
    *,
    service: str,
    endpoint: str,
    retry_statuses: Sequence[int] = (),
    retry_transport: bool = False,
    attempts: int = 1,
    wait_s: Callable[[int], float] = lambda attempt: 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """Sends a request, retrying only where production retries.

    Args:
        call: Issues the request once.
        service: Service name for errors.
        endpoint: Path for errors.
        retry_statuses: Statuses worth another attempt.
        retry_transport: Whether a connection error is worth another attempt.
        attempts: Total attempts, including the first.
        wait_s: Seconds to wait after the given (1-based) failed attempt.
        sleep: Injected for tests.

    Returns:
        The first 2xx response.

    Raises:
        ServiceError: On a non-retried status, on the last failed attempt,
            or on a timeout (timeouts are never retried: the work may still
            be running on the server).
    """
    for attempt in range(1, attempts + 1):
        try:
            response = call()
        except httpx.TimeoutException as exc:
            raise ServiceError(service, endpoint, str(exc), timed_out=True) from exc
        except httpx.TransportError as exc:
            if not retry_transport or attempt == attempts:
                raise ServiceError(service, endpoint, str(exc)) from exc
            sleep(wait_s(attempt))
            continue
        if response.is_success:
            return response
        if response.status_code not in retry_statuses or attempt == attempts:
            raise ServiceError(
                service,
                endpoint,
                response.text[-500:],
                status=response.status_code,
            )
        sleep(wait_s(attempt))
    raise AssertionError("unreachable: the loop returns or raises")


@dataclass(frozen=True)
class Outcome[T, R]:
    """One item of a pooled batch: its result, or the error it raised."""

    item: T
    result: R | None
    error: ServiceError | None


def map_concurrently[T, R](
    fn: Callable[[T], R], items: Iterable[T], limit: int
) -> list[Outcome[T, R]]:
    """Runs ``fn`` over ``items`` with at most ``limit`` in flight.

    A ``ServiceError`` is recorded against its item instead of aborting the
    batch, because production tolerates per-item failures and reports them.
    Any other exception is a bug and propagates.

    Returns:
        Outcomes in input order.
    """

    def guarded(item: T) -> Outcome[T, R]:
        try:
            return Outcome(item, fn(item), None)
        except ServiceError as exc:
            return Outcome(item, None, exc)

    with ThreadPoolExecutor(max_workers=max(1, limit)) as pool:
        return list(pool.map(guarded, items))
