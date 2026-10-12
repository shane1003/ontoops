"""Optional Langfuse tracing (brainstorm: optional; keys in .env).

Traces stay on-host. The client is built ONLY when all Langfuse settings are present AND
``LANGFUSE_BASE_URL`` points at a loopback or private (RFC1918) host — so the gloss/CQ text
carried in traces never leaves the machine. A public host, missing keys, or a missing
library disables tracing (no-op). Tracing must never break the pipeline: every client call is
wrapped so an observability failure degrades to nothing.

Instrument a boundary with the context manager::

    tracer = get_tracer(config)
    with tracer.trace("update", input={"components": path}) as tr:
        ...  # do work
        tr.update(metadata={"n_new": n_new}, output={"ok": True})
"""

from __future__ import annotations

import functools
import ipaddress
import logging
import time
from contextlib import contextmanager
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_TRACER: Tracer | None = None
_TRACER_KEY: tuple | None = None


def is_local_host(url: str | None) -> bool:
    """True when the URL host is loopback or a private/internal address (never public)."""
    if not url:
        return False
    host = urlparse(url).hostname
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_private or ip.is_link_local
    except ValueError:
        # Not an IP: allow only clearly-internal names, reject public domains.
        return (
            host == "localhost"
            or host.endswith((".local", ".internal"))
            or "." not in host
        )


class _Handle:
    """A live trace record the caller enriches; flushed to Langfuse on context exit."""

    def __init__(self, metadata: dict):
        self.metadata = metadata
        self.output = None

    def update(self, *, metadata: dict | None = None, output=None) -> None:
        if metadata:
            self.metadata.update(metadata)
        if output is not None:
            self.output = output


class Tracer:
    def __init__(self, client, status: str):
        self._client = client
        self.status = status  # human-readable: "enabled" or the reason it is off

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def trace(self, name: str, *, input=None, metadata: dict | None = None):
        handle = _Handle(dict(metadata or {}))
        start = time.perf_counter()
        try:
            yield handle
        finally:
            if self._client is not None:
                handle.metadata["duration_ms"] = round(
                    (time.perf_counter() - start) * 1000, 1
                )
                try:
                    self._client.trace(
                        name=name,
                        input=input,
                        output=handle.output,
                        metadata=handle.metadata,
                    )
                except Exception as exc:  # noqa: BLE001 - never let tracing break the pipeline
                    logger.debug("langfuse trace %r failed: %s", name, exc)

    def flush(self) -> None:
        if self._client is not None:
            try:
                self._client.flush()
            except Exception as exc:  # noqa: BLE001
                logger.debug("langfuse flush failed: %s", exc)


def langfuse_client_and_status(
    url: str | None, public_key: str | None, secret_key: str | None
):
    """Builds the Langfuse client only when the guards pass. Never raises.

    Both modules (tboxcraft's Tracer, conceptualizer's generation logging) read the same
    keys, so the setup and data-locality guards are owned here, in one place.

    Returns:
        A ``(client | None, status)`` tuple.
    """
    if not (url and public_key and secret_key):
        return None, "disabled: LANGFUSE_* not fully set"
    if not is_local_host(url):
        logger.warning(
            "Langfuse tracing DISABLED: %s is not a local/private host (data locality)",
            url,
        )
        return None, "disabled: non-local host"
    try:
        from langfuse import Langfuse

        client = Langfuse(public_key=public_key, secret_key=secret_key, host=url)
    except Exception as exc:  # noqa: BLE001 - missing lib or bad config -> no-op
        logger.warning("Langfuse tracing disabled: %s", exc)
        return None, f"disabled: {exc}"
    logger.info("Langfuse tracing enabled -> %s", url)
    return client, "enabled"


def build_tracer(config) -> Tracer:
    """Build a Tracer from config, honoring the data-locality guard. Never raises."""
    return Tracer(
        *langfuse_client_and_status(
            config.langfuse_base_url,
            config.langfuse_public_key,
            config.langfuse_secret_key,
        )
    )


def get_tracer(config) -> Tracer:
    """Process-memoized tracer keyed on the Langfuse settings (first config wins per process)."""
    global _TRACER, _TRACER_KEY
    key = (
        config.langfuse_base_url,
        config.langfuse_public_key,
        config.langfuse_secret_key,
    )
    if _TRACER is None or _TRACER_KEY != key:
        _TRACER = build_tracer(config)
        _TRACER_KEY = key
    return _TRACER


def flush_tracer() -> None:
    """Flush the already-built tracer, if any. Never constructs one (so it stays a true no-op
    when nothing has traced yet — e.g. a CLI run whose pipeline call was mocked in tests)."""
    if _TRACER is not None:
        _TRACER.flush()


_RESULT_KEYS = (
    "ok",
    "persisted",
    "n_components",
    "n_concepts",
    "n_candidates",
    "n_new",
    "n_merged",
    "n_relabel",
    "n_gray",
    "version_id",
)


def _result_meta(result) -> dict:
    return {k: getattr(result, k) for k in _RESULT_KEYS if hasattr(result, k)}


def traced_pipeline(name: str):
    """Trace a pipeline entry point sharing the ``(components_path, config, ...)`` signature
    (generate / update). No-op when tracing is disabled."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(components_path, config, *args, **kwargs):
            tracer = get_tracer(config)
            with tracer.trace(name, input={"components": str(components_path)}) as tr:
                result = fn(components_path, config, *args, **kwargs)
                tr.update(
                    metadata=_result_meta(result),
                    output={"ok": getattr(result, "ok", None)},
                )
                return result

        return wrapper

    return decorator
