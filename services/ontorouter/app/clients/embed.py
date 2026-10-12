"""TEI e5 embedding client.

Batches by 32 (TEI's default --max-client-batch-size), applies the e5 instruct
prefix client-side, POSTs to ``{endpoint}/embed``, L2-normalizes, and returns
plain float lists.

Crucially it does **not** whiten. The upstream ``tbox_term.embedding_val``
column stores whitened vectors whose original norms are unrecoverable, so
ontorouter never reuses them — it re-embeds surface text into its own
``ontorouter`` schema instead.
"""

from __future__ import annotations

import math
from functools import lru_cache

E5_INSTRUCTION = "Identify the domain concept that this term describes"
E5_DIMENSION = 1024
MAX_BATCH = 32


def format_e5_input(text: str) -> str:
    return f"Instruct: {E5_INSTRUCTION}\nQuery: {text}"


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


class EmbedClient:
    def __init__(self, endpoint: str, timeout: float = 30.0) -> None:
        import requests

        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. e5 prefix applied here; TEI does not add it."""
        out: list[list[float]] = []
        for start in range(0, len(texts), MAX_BATCH):
            chunk = texts[start : start + MAX_BATCH]
            formatted = [format_e5_input(t) for t in chunk]
            resp = self._session.post(
                f"{self.endpoint}/embed",
                json={"inputs": formatted},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            vectors = resp.json()
            # Guard against a partial/misaligned TEI response: one vector per
            # input, in order. Without this a short response silently drops
            # the tail (zip) and misaligns every entry after the gap — a
            # silent-corruption class bug.
            if not isinstance(vectors, list) or len(vectors) != len(chunk):
                got = (
                    len(vectors)
                    if isinstance(vectors, list)
                    else type(vectors).__name__
                )
                raise ValueError(
                    f"TEI returned {got} embeddings for a batch of {len(chunk)} inputs "
                    "— partial or misaligned response"
                )
            for vec in vectors:
                if len(vec) != E5_DIMENSION:
                    raise ValueError(
                        f"embedding dim {len(vec)} != expected {E5_DIMENSION}"
                    )
                out.append(l2_normalize([float(x) for x in vec]))
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


@lru_cache(maxsize=4)
def embed_client(endpoint: str, timeout: float) -> EmbedClient:
    """Return the process-wide TEI client for one endpoint and timeout.

    ``EmbedClient`` opens a ``requests.Session``, whose value is the pooled,
    keep-alive connections underneath it; building one per request or per
    backfill pass throws that away and pays a fresh TCP (and TLS) handshake
    every time. The client is only read from after construction — no header,
    cookie or auth mutation — so sharing it across the threadpool is safe.

    Args:
        endpoint: The TEI base URL.
        timeout: Per-call timeout in seconds.

    Returns:
        The cached client for that pair.
    """
    return EmbedClient(endpoint, timeout=timeout)
