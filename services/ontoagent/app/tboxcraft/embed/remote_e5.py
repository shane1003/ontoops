"""Remote e5 embedding provider — calls a TEI server.

TEI embeds raw text and does NOT apply e5's ``Instruct:/Query:`` prefix, so this provider
applies it client-side (reusing ``format_e5_input``) before POSTing to TEI's ``/embed``,
keeping vectors parity-identical with the in-process e5 path. Runs against a local TEI, so
no OpenAI-style egress guard is needed.
"""

from __future__ import annotations

import numpy as np

from app.tboxcraft.embed.base import EmbeddingProvider, l2_normalize
from app.tboxcraft.embed.e5 import E5_DIMENSION, format_e5_input

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"


class RemoteE5Provider(EmbeddingProvider):
    name = "remote_e5"

    def __init__(
        self,
        endpoint: str,
        *,
        revision: str = "main",
        model_name: str = DEFAULT_MODEL_NAME,
        session=None,
        timeout: int = 120,
        max_batch: int = 32,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.revision = revision
        self.model_name = (
            model_name  # HF model id — provenance keys on this, not `name`
        )
        self.timeout = timeout
        # TEI's default --max-client-batch-size is 32; chunk to stay under it.
        self.max_batch = max_batch
        self._session = session

    def _get_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    @property
    def dimension(self) -> int:
        return E5_DIMENSION

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, E5_DIMENSION), dtype=np.float32)
        session = self._get_session()
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), self.max_batch):
            batch = texts[start : start + self.max_batch]
            formatted = [
                format_e5_input(t) for t in batch
            ]  # instruct prefix applied here
            response = session.post(
                f"{self.endpoint}/embed",
                json={"inputs": formatted},
                timeout=self.timeout,
            )
            response.raise_for_status()
            chunks.append(np.asarray(response.json(), dtype=np.float32))
        vectors = np.vstack(chunks)
        if vectors.shape[1] != E5_DIMENSION:
            raise ValueError(
                f"TEI returned dim {vectors.shape[1]}, expected {E5_DIMENSION} — model/config mismatch"
            )
        # Callers key vectors back to inputs by position (generate/update both do
        # `{c.uid: matrix[i] for i, c in enumerate(ordered)}`), so a row count that does not match
        # the input count silently shifts every vector after the gap onto the wrong component.
        # These vectors are persisted, and the whitening mean and per-kind tau are derived from
        # them, so the misalignment outlives the run that caused it. Positional identity is not
        # recoverable here — fail the batch instead of storing a plausible-looking wrong answer.
        if vectors.shape[0] != len(texts):
            raise ValueError(
                f"TEI returned {vectors.shape[0]} embeddings for {len(texts)} inputs — "
                "cannot align vectors to components"
            )
        return l2_normalize(vectors)
