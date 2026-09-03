"""Hybrid retrieval with reciprocal rank fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from report_rag.retrieval.bm25 import BM25Index
from report_rag.retrieval.embeddings import Embedder
from report_rag.retrieval.vector import VectorIndex
from report_rag.schemas import Chunk


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk with its fused score and the raw component ranks (1-based, 0 = absent)."""

    chunk: Chunk
    score: float
    bm25_rank: int
    dense_rank: int


class Retriever(Protocol):
    """Minimal retriever contract used by the pipeline and the agent."""

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        """Return the best ``top_k`` chunks for ``query``."""
        ...


class HybridRetriever:
    """Fuse BM25 and dense rankings with weighted reciprocal rank fusion.

    RRF is used instead of score interpolation because BM25 and cosine scores
    live on incomparable scales; rank fusion is robust without calibration.

    Args:
        chunks: Corpus.
        embedder: Dense embedder; ``None`` disables the dense leg (pure BM25).
        bm25_weight: Weight of the sparse leg in ``[0, 1]``.
        rrf_k: RRF smoothing constant (60 is the standard default).
        candidate_multiplier: Each leg returns ``top_k * multiplier`` candidates before fusion.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder | None = None,
        bm25_weight: float = 0.6,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
    ) -> None:
        if not 0.0 <= bm25_weight <= 1.0:
            msg = "bm25_weight must be within [0, 1]"
            raise ValueError(msg)
        self.chunks = list(chunks)
        self.bm25 = BM25Index(self.chunks)
        self.vector = VectorIndex(self.chunks, embedder) if embedder is not None else None
        self.bm25_weight = bm25_weight if self.vector is not None else 1.0
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        """Return fused top-``k`` results, best first."""
        n_cand = top_k * self.candidate_multiplier
        sparse = self.bm25.search(query, n_cand)
        dense = self.vector.search(query, n_cand) if self.vector is not None else []

        fused: dict[str, float] = {}
        bm25_rank: dict[str, int] = {}
        dense_rank: dict[str, int] = {}
        by_id: dict[str, Chunk] = {}

        for rank, (chunk, _) in enumerate(sparse, start=1):
            by_id[chunk.chunk_id] = chunk
            bm25_rank[chunk.chunk_id] = rank
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + self.bm25_weight / (
                self.rrf_k + rank
            )
        for rank, (chunk, _) in enumerate(dense, start=1):
            by_id[chunk.chunk_id] = chunk
            dense_rank[chunk.chunk_id] = rank
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + (1 - self.bm25_weight) / (
                self.rrf_k + rank
            )

        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [
            ScoredChunk(
                chunk=by_id[cid],
                score=score,
                bm25_rank=bm25_rank.get(cid, 0),
                dense_rank=dense_rank.get(cid, 0),
            )
            for cid, score in ordered
        ]
