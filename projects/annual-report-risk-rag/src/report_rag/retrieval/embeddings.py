"""Embedding back-ends behind a tiny protocol.

:class:`HashingEmbedder` is a deterministic, dependency-free bag-of-words
embedding (feature hashing with unigrams + bigrams, L2-normalised). It is
weaker than a transformer but makes the dense path fully testable offline.
:class:`SentenceTransformerEmbedder` is the production option (extra ``dense``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from report_rag.retrieval.tokenize import tokenize

FloatArray = NDArray[np.float64]


@runtime_checkable
class Embedder(Protocol):
    """Anything that maps texts to L2-normalised vectors."""

    @property
    def dimension(self) -> int:
        """Vector length."""
        ...

    def embed(self, texts: Sequence[str]) -> FloatArray:
        """Return an ``(n, dimension)`` array of unit-norm vectors."""
        ...


def _stable_hash(token: str, dimension: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    sign = 1.0 if value & 1 else -1.0
    return (value >> 1) % dimension, sign


class HashingEmbedder:
    """Feature-hashed unigram+bigram embedding with sublinear term frequency."""

    def __init__(self, dimension: int = 4096) -> None:
        if dimension < 64:
            msg = "dimension must be >= 64"
            raise ValueError(msg)
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Vector length."""
        return self._dimension

    def embed(self, texts: Sequence[str]) -> FloatArray:
        """Embed a batch of texts."""
        out = np.zeros((len(texts), self._dimension), dtype=np.float64)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            grams = tokens + [f"{a}_{b}" for a, b in pairwise(tokens)]
            for gram in grams:
                idx, sign = _stable_hash(gram, self._dimension)
                out[row, idx] += sign
            np.log1p(np.abs(out[row]), out=out[row], where=out[row] != 0)
            norm = float(np.linalg.norm(out[row]))
            if norm > 0:
                out[row] /= norm
        return out


class SentenceTransformerEmbedder:
    """Wrapper over ``sentence-transformers`` models (install extra ``dense``)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            msg = "Dense embeddings need the 'dense' extra: uv sync --extra dense"
            raise ImportError(msg) from exc
        self._model = SentenceTransformer(model_name)
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        """Vector length."""
        return self._dimension

    def embed(self, texts: Sequence[str]) -> FloatArray:  # pragma: no cover - needs model weights
        """Embed a batch of texts with the transformer."""
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float64)
