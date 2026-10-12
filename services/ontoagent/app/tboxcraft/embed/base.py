"""Embedding provider interface.

Providers return L2-normalized vectors so cosine similarity reduces to a dot product.
The provider owns any model-specific input formatting (e5's ``Instruct:/Query:``
vs OpenAI's raw text) so callers stay provider-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero rows are left as zeros (no divide-by-zero)."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


class EmbeddingProvider(ABC):
    """Maps texts to normalized embedding vectors."""

    #: Provider identifier, used to tag calibrated thresholds.
    name: str = "base"

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimensionality (e.g., 3072 for text-embedding-3-large, 1024 for e5)."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), dimension)`` array of L2-normalized vectors."""
