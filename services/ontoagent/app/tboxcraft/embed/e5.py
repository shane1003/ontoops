"""Local e5 embedding provider (production target).

``intfloat/multilingual-e5-large-instruct`` run locally via sentence-transformers (an
optional dependency — install the ``local`` extra). For symmetric term-to-term clustering
there is no natural query/passage split, so every gloss is embedded as an instructed query
with a fixed clustering instruction — a symmetric, deterministic choice.
"""

from __future__ import annotations

import numpy as np

from app.tboxcraft.embed.base import EmbeddingProvider, l2_normalize

E5_DIMENSION = 1024

# Symmetric clustering instruction applied to every gloss (both sides of a comparison).
CLUSTERING_INSTRUCTION = "Identify the domain concept that this term describes"


def format_e5_input(text: str, instruction: str = CLUSTERING_INSTRUCTION) -> str:
    """e5-instruct query format: ``Instruct: {task}\\nQuery: {text}``."""
    return f"Instruct: {instruction}\nQuery: {text}"


class E5EmbeddingProvider(EmbeddingProvider):
    name = "e5"

    def __init__(
        self,
        model: str = "intfloat/multilingual-e5-large-instruct",
        *,
        revision: str = "main",
        cache_folder: str | None = None,
        encoder=None,
    ):
        self.model = model
        # Provenance keys on the HF model id + revision, so in-process e5 and
        # remote_e5 vectors share the same embedding_model row and stay mutually searchable.
        self.model_name = model
        self.revision = revision
        # A persistent local dir for the weights so they download once and are reused
        # (see Config.hf_cache_dir). None -> HuggingFace's default cache.
        self.cache_folder = cache_folder
        self._encoder = encoder  # injected in tests; lazily loaded otherwise

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise RuntimeError(
                    "The e5 provider needs the 'local' extra: pip install -e '.[local]'"
                ) from exc
            self._encoder = SentenceTransformer(
                self.model, cache_folder=self.cache_folder
            )
        return self._encoder

    @property
    def dimension(self) -> int:
        return E5_DIMENSION

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        formatted = [format_e5_input(t) for t in texts]
        vectors = np.asarray(
            self._get_encoder().encode(formatted, normalize_embeddings=True),
            dtype=np.float32,
        )
        return l2_normalize(vectors)
