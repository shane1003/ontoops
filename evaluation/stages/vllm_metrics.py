"""Snapshots of a vLLM server's counters around a cell.

The LLM servers are shared with other work on the machine, so a cell cannot
own them. What it can do is record the counters before and after, and the
queue at each step boundary, so a result can be read with the traffic that
ran beside it. ``finished_reason="length"`` counts completions cut off by
their token cap, which is worth knowing on its own.
"""

import re
import urllib.request
from typing import Any

_SAMPLE = re.compile(r"^(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$")
COUNTERS = (
    "vllm:request_success_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
)
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")


def parse(text: str) -> dict[str, float]:
    """Sums each metric over its labels, keeping ``finished_reason`` apart.

    Returns:
        ``{metric: value}``; ``request_success_total`` also appears as
        ``request_success_total:<reason>`` for each finish reason.
    """
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = _SAMPLE.match(line.strip())
        if not match:
            continue
        name, labels, raw = match.group(1), match.group(2) or "", float(match.group(3))
        if name not in COUNTERS and name not in GAUGES:
            continue
        values[name] = values.get(name, 0.0) + raw
        reason = re.search(r'finished_reason="([a-z_]+)"', labels)
        if reason:
            key = f"{name}:{reason.group(1)}"
            values[key] = values.get(key, 0.0) + raw
    return values


def snapshot(root_url: str) -> dict[str, float] | None:
    """Reads ``<root>/metrics``; None when the server does not answer."""
    try:
        with urllib.request.urlopen(root_url.rstrip("/") + "/metrics", timeout=5) as r:  # noqa: S310
            return parse(r.read().decode("utf-8", errors="replace"))
    except Exception:
        # A missing snapshot is recorded as missing, never as zero traffic.
        return None


def delta(
    before: dict[str, float] | None, after: dict[str, float] | None
) -> dict[str, Any]:
    """Counter differences over a cell; gauges are not differenced."""
    if before is None or after is None:
        return {"available": False}
    out: dict[str, Any] = {"available": True}
    for key, value in after.items():
        metric = ":".join(key.split(":")[:2])  # drop a ":<finish reason>" suffix
        if metric in GAUGES:
            continue
        out[key] = value - before.get(key, 0.0)
    return out
