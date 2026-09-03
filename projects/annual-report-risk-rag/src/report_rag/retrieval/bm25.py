"""A small, dependency-free BM25 (Okapi) implementation."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from report_rag.retrieval.tokenize import tokenize
from report_rag.schemas import Chunk


class BM25Index:
    """BM25 Okapi over chunk text with the section title prepended.

    Args:
        chunks: The corpus.
        k1: Term-frequency saturation parameter.
        b: Length-normalisation strength.
    """

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            msg = "BM25Index needs at least one chunk"
            raise ValueError(msg)
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._doc_tokens = [tokenize(f"{c.section} {c.text}") for c in self.chunks]
        self._doc_len = [len(toks) for toks in self._doc_tokens]
        self._avg_len = sum(self._doc_len) / len(self._doc_len)
        self._tf = [Counter(toks) for toks in self._doc_tokens]
        df: Counter[str] = Counter()
        for toks in self._doc_tokens:
            df.update(set(toks))
        n = len(self.chunks)
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def score(self, query: str) -> list[float]:
        """Return a BM25 score per chunk (same order as ``self.chunks``)."""
        q_tokens = tokenize(query)
        scores = [0.0] * len(self.chunks)
        for term in q_tokens:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self._tf):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self._doc_len[i] / self._avg_len)
                scores[i] += idf * f * (self.k1 + 1) / (f + norm)
        return scores

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Top-``k`` chunks with a positive score, best first."""
        scores = self.score(query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.chunks[i], scores[i]) for i in ranked[:top_k] if scores[i] > 0]
