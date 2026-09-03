"""In-memory cosine-similarity index."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from report_rag.retrieval.embeddings import Embedder
from report_rag.schemas import Chunk


class VectorIndex:
    """Brute-force cosine search; adequate for a handful of reports.

    Swap for FAISS / pgvector behind the same interface for large corpora.
    """

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        if not chunks:
            msg = "VectorIndex needs at least one chunk"
            raise ValueError(msg)
        self.chunks = list(chunks)
        self.embedder = embedder
        self._matrix = embedder.embed([f"{c.section}. {c.text}" for c in self.chunks])

    def score(self, query: str) -> list[float]:
        """Cosine similarity of the query against every chunk."""
        q = self.embedder.embed([query])[0]
        return [float(s) for s in self._matrix @ q]

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Top-``k`` chunks by cosine similarity."""
        scores = np.asarray(self.score(query))
        order = np.argsort(-scores)[:top_k]
        return [(self.chunks[int(i)], float(scores[i])) for i in order if scores[i] > 0]
