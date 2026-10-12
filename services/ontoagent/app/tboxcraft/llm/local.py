"""Local LLM provider — calls an on-host OpenAI-compatible chat server.

Posts to ``{url}/v1/chat/completions`` with JSON-object response formatting and parses the
message content as the Verdict object. Runs against a local server (e.g. vLLM / llama.cpp /
Ollama's OpenAI shim), so no egress guard is needed — the prompt never leaves
the box.
"""

from __future__ import annotations

import json
import logging
import threading

from app.tboxcraft.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class LocalLLMProvider(LLMProvider):
    name = "local"

    def __init__(self, url: str, model: str, *, session=None, timeout: int = 120):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._injected = session
        self._local = threading.local()

    def _get_session(self):
        """One session per calling thread.

        The verifier fans its calls out over a thread pool, and ``requests.Session`` is not
        documented as thread-safe — its connection pool is, but the adapter/cookie state around it
        is not, so one shared session under load can hand a response to the wrong caller. Threads
        are pooled and reused, so this creates a handful of sessions, not one per call. An
        injected session belongs to the caller and is used exactly as given.
        """
        if self._injected is not None:
            return self._injected
        session = getattr(self._local, "session", None)
        if session is None:
            import requests

            session = self._local.session = requests.Session()
        return session

    def _post(self, system: str, user: str, *, json_mode: bool) -> object:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return self._get_session().post(
            f"{self.url}/v1/chat/completions", json=body, timeout=self.timeout
        )

    def judge(self, system: str, user: str) -> dict:
        """Ask the local server and parse the reply as the Verdict object.

        ``response_format=json_object`` is sent first because it makes the server enforce the
        shape, but not every OpenAI-compatible backend accepts it (vLLM needs a guided-decoding
        backend; llama.cpp/Ollama shims vary). A server that rejects the parameter answers 4xx,
        and since :func:`~tboxcraft.verify_term.verify_term` deliberately swallows transport errors
        to keep a down LLM from blocking human review, that 4xx would surface as "verifier
        unavailable" for every candidate — verification silently off, with a valid-looking run.
        So a 4xx is retried once without the parameter; the prompt already demands JSON, and the
        parse below still rejects anything else.
        """
        resp = self._post(system, user, json_mode=True)
        if 400 <= resp.status_code < 500:
            # WARNING, not INFO: only the CLI calls basicConfig(INFO); under uvicorn the root
            # logger stays at WARNING, so an INFO line here would leave the fallback with no
            # trace at all in the API path — exactly where it is hardest to notice.
            logger.warning(
                "local LLM at %s rejected response_format (HTTP %s); retrying without JSON mode "
                "— the server enforces no schema now, so a malformed reply becomes 'unavailable'",
                self.url,
                resp.status_code,
            )
            resp = self._post(system, user, json_mode=False)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"LLM returned non-JSON content: {content!r}") from exc
