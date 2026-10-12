"""Local LLM client (OpenAI-compatible servers only; the server address is OPENAI_BASE_URL).

When the key is missing or an SDK call fails, the exception is raised as is so the caller
can fall back. Transient errors (429/5xx) get a bounded retry with exponential backoff.

Every call leaves a progress log (start/finish, model, tokens, elapsed time) and, when
Langfuse is configured, is traced as a generation (tracing.py). complete(fast=...) picks
the model tier.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.core.config import Settings
from app.core.llm import tracing

# Large JSON outputs (experts/merge) need more than 60s, or timeouts waste retries, so
# this is generous.
_TIMEOUT = 180.0
_MAX_ATTEMPTS = 3

log = logging.getLogger("conceptualizer.llm")


def _status_of(exc: BaseException) -> int | None:
    """Extracts the HTTP status code from an exception (any provider SDK).

    Returns:
        The status code, or None if there is none.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    return status if isinstance(status, int) else None


def _is_transient(exc: BaseException) -> bool:
    """Tells whether this is a transient error such as rate-limit (429)/5xx.

    Decided by status_code, independent of the provider SDK.
    """
    status = _status_of(exc)
    if status is not None:
        return status == 429 or 500 <= status < 600
    # Also decided by the explicit rate-limit exception type name
    return exc.__class__.__name__ in {"RateLimitError", "APIConnectionError"}


def _is_client_error(exc: BaseException) -> bool:
    """Tells whether this is a 4xx (client error), to detect a rejected request parameter.

    Same rule as tboxcraft (`400 <= status < 500`). **5xx is False here**: a server failure
    must not be mistaken for a parameter problem and silently worked around. Exceptions
    with no extractable status code (connection errors etc.) are also False, so they
    propagate as is.

    (429 only gets here after tenacity has already retried and absorbed it via
    `_is_transient` above.)
    """
    status = _status_of(exc)
    return status is not None and 400 <= status < 500


# Retry of transient errors (429/5xx): bounded retry with exponential backoff.
_retry_transient = retry(
    retry=retry_if_exception(_is_transient),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    reraise=True,
)


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Create and cache the SDK client once, rather than on every complete() call.
        self._client = self._build_client()

    def _build_client(self):
        from openai import OpenAI

        key = self.settings.openai_api_key
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return OpenAI(api_key=key, timeout=_TIMEOUT)

    def complete(self, system: str, user: str, *, fast: bool = False, label: str = "llm") -> str:
        """Returns one completion (text) for the system/user prompts.

        Args:
            system: The system prompt.
            user: The user prompt.
            fast: When True, uses the light model (review, subject derivation etc.).
            label: Identifier for logs/traces.

        Returns:
            The completion text.
        """
        model = self.settings.llm.model_for(fast)
        log.info("▶ [%s] %s call…", label, model)
        start = datetime.now()
        t0 = time.perf_counter()
        text, usage = self._openai(system, user, model)
        dt = time.perf_counter() - t0
        log.info("✔ [%s] %s · out %s tok · %.1fs", label, model, usage.get("output", "?"), dt)
        tracing.log_generation(label, model, system, user, text, usage, start, datetime.now())
        return text

    def _openai(self, system: str, user: str, model: str) -> tuple[str, dict]:
        """OpenAI-compatible path. Forces JSON mode; falls back once on servers without it.

        Every caller on this path (classifier, semantic merge) wants "one JSON object". A
        local LLM (OpenAI-compatible server) keeps the format less reliably than Claude,
        and a broken response leads to extract parse failure -> verdicts lost -> 0
        individuals. So JSON is forced with response_format. If an older gateway rejects
        that parameter, it retries once without JSON mode (otherwise the 400 silently
        flows into an empty-annotation degrade).

        The fallback is decided by **status code 4xx** (same rule as tboxcraft). Matching
        the exception message string is wrong both ways: a gateway that rejects the
        parameter without putting "response_format" in its wording misses the fallback
        and flows into the very degrade path it should prevent, and conversely a 5xx that
        happens to contain the word drops JSON mode and silently succeeds, hiding the
        server failure. **5xx never falls back** (it is a server failure, not a parameter
        problem).
        """

        def _create(json_mode: bool):
            kwargs: dict = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            # The gpt-5 family allows only the default (1) and rejects temperature=0.0
            # with a 400.
            if not str(model).startswith("gpt-5"):
                kwargs["temperature"] = self.settings.llm.temperature
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            @_retry_transient
            def _call():
                return self._client.chat.completions.create(**kwargs)

            return _call()

        try:
            resp = _create(json_mode=True)
        except Exception as exc:  # noqa: BLE001 — only 4xx falls back; rest (5xx, conn) propagates
            if not _is_client_error(exc):
                raise
            log.warning(
                "[%s] JSON mode request rejected with 4xx → retrying without response_format: %s",
                model,
                exc,
            )
            resp = _create(json_mode=False)
        u = getattr(resp, "usage", None)
        usage = {
            "input": getattr(u, "prompt_tokens", None),
            "output": getattr(u, "completion_tokens", None),
            "unit": "TOKENS",
        }
        return (resp.choices[0].message.content or ""), usage
