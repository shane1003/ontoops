"""Langfuse integration (optional). Traces LLM calls to show stages, latency and tokens.

Active only when the environment variables are set; without them, or when the SDK is not
installed or fails, it is a **silent no-op** (every call is wrapped in try/except so a
tracing failure never blocks the pipeline).

Client setup and the **data-locality guard** are owned by core.tracing (unified in 3c):
tracing is disabled when LANGFUSE_HOST/BASE_URL is not a local/private host, so the
CQ/domain text carried in traces never leaves the host; same policy as tboxcraft.

Required environment variables:
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST (or LANGFUSE_BASE_URL)

Target SDK: langfuse v2 (`Langfuse().trace()/.generation()`).
"""

from __future__ import annotations

import contextvars
import functools
import os
from contextlib import contextmanager
from datetime import datetime

from app.core.tracing import langfuse_client_and_status

# Langfuse trace of the current run (one extract). Its generations are grouped under it.
_current_trace: contextvars.ContextVar = contextvars.ContextVar("lf_trace", default=None)
_client = None  # lazy singleton


def enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def _host() -> str | None:
    """Self-hosted address.

    The SDK standard is LANGFUSE_HOST, but LANGFUSE_BASE_URL is accepted too.
    """
    return os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or None


def _client_or_none():
    global _client
    if not enabled():
        return None
    if _client is None:
        _client, _status = langfuse_client_and_status(
            _host(), os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY")
        )
    return _client


@contextmanager
def trace_run(name: str, **metadata):
    """Groups one stage run into one trace (context manager).

    When disabled or **already inside a parent trace**, passes through without creating a
    new trace (prevents nesting).
    """
    lf = _client_or_none()
    if lf is None or _current_trace.get() is not None:
        yield
        return
    try:
        trace = lf.trace(name=name, metadata=metadata or None)
    except Exception:  # noqa: BLE001
        yield
        return
    token = _current_trace.set(trace)
    try:
        yield
    finally:
        _current_trace.reset(token)
        try:
            lf.flush()
        except Exception:  # noqa: BLE001
            pass


def traced(name: str):
    """Decorator grouping a whole function run into one Langfuse trace.

    The LLM calls inside it nest under that trace.
    """

    def deco(fn):
        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            with trace_run(name):
                return fn(*args, **kwargs)

        return wrap

    return deco


def log_generation(
    name: str,
    model: str,
    system: str,
    user: str,
    output: str,
    usage: dict,
    start: datetime,
    end: datetime,
) -> None:
    """Records one LLM call as a Langfuse generation. Ignored when disabled or on error."""
    lf = _client_or_none()
    if lf is None:
        return
    try:
        kwargs = dict(
            name=name,
            model=model,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            output=output,
            usage=usage,
            start_time=start,
            end_time=end,
        )
        parent = _current_trace.get()
        (parent.generation if parent is not None else lf.generation)(**kwargs)
    except Exception:  # noqa: BLE001 - a tracing failure does not affect the pipeline
        pass
