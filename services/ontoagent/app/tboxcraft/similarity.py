"""Cosine similarity over normalized embedding vectors."""

from __future__ import annotations

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors (assumed L2-normalized, but robust if not)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine of row vectors. For normalized rows this is ``V @ V.T``."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = vectors / norms
    return unit @ unit.T


def mean_center(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-center a batch of embeddings and re-normalize, returning ``(centered, mean)``.

    NOT statistical whitening — it subtracts the batch mean but does not decorrelate the
    dimensions (no covariance term), so it is mean-centering, not the Σ^-1/2 transform "whiten"
    would name. Centering is the part that matters here: e5 cosine scores are compressed into a
    narrow high band (all pairs ~0.87–0.99), which collapses distinct concepts under a single
    threshold. The band is a shared component sitting on every vector; subtracting the batch mean
    removes it and spreads the distribution around zero so cosine becomes discriminative again.
    The mean is returned so query vectors can be centered into the same space (see
    :func:`center_query`).
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[0] < 2:
        return matrix, np.zeros(
            matrix.shape[1] if matrix.ndim == 2 else 0, dtype=np.float32
        )
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (centered / norms).astype(np.float32), mean.astype(np.float32)


def center_query(vector: np.ndarray, mean: np.ndarray | None) -> np.ndarray:
    """Center a single query vector by the stored batch ``mean`` and re-normalize.

    A ``None`` mean (no whitening applied) returns the vector unchanged.
    """
    vector = np.asarray(vector, dtype=np.float32)
    if mean is None:
        return vector
    centered = vector - np.asarray(mean, dtype=np.float32)
    norm = np.linalg.norm(centered)
    return (centered / norm).astype(np.float32) if norm else centered
