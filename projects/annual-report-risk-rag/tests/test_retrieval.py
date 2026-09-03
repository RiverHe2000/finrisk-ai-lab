from __future__ import annotations

import numpy as np
import pytest

from report_rag.retrieval import BM25Index, HashingEmbedder, HybridRetriever, VectorIndex
from report_rag.retrieval.tokenize import tokenize
from report_rag.schemas import Chunk


def _chunks(texts: list[str]) -> list[Chunk]:
    return [
        Chunk(chunk_id=f"d:{i:04d}", doc_id="d", section="s", text=t, position=i)
        for i, t in enumerate(texts)
    ]


def test_tokenize_keeps_numbers_and_drops_stopwords() -> None:
    assert tokenize("The CET1 ratio was 12.4% of RWA") == ["cet1", "ratio", "12.4", "rwa"]


def test_bm25_ranks_matching_chunk_first() -> None:
    idx = BM25Index(_chunks(["liquidity coverage ratio 134%", "net profit rose", "dividend"]))
    hits = idx.search("liquidity coverage ratio")
    assert hits[0][0].chunk_id == "d:0000"
    assert len(hits) == 1  # only positive scores are returned


def test_bm25_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError, match="at least one chunk"):
        BM25Index([])


def test_hashing_embedder_is_unit_norm_and_deterministic() -> None:
    emb = HashingEmbedder(dimension=256)
    a = emb.embed(["capital ratio strong"])
    b = emb.embed(["capital ratio strong"])
    assert emb.dimension == 256
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)
    assert np.linalg.norm(emb.embed([""])[0]) == 0.0


def test_hashing_embedder_dimension_guard() -> None:
    with pytest.raises(ValueError, match=">= 64"):
        HashingEmbedder(dimension=8)


def test_vector_index_prefers_lexically_similar_chunk() -> None:
    chunks = _chunks(["CET1 capital ratio 12.4 percent", "cricket scores weekend", "fleet cars"])
    idx = VectorIndex(chunks, HashingEmbedder(512))
    hits = idx.search("CET1 capital ratio")
    assert hits[0][0].chunk_id == "d:0000"


def test_vector_index_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError, match="at least one chunk"):
        VectorIndex([], HashingEmbedder())


def test_hybrid_fuses_and_reports_component_ranks() -> None:
    chunks = _chunks(
        [
            "Liquidity Coverage Ratio LCR was 134 percent",
            "The LCR requirement is 100 percent under APS 210",
            "Unrelated remuneration report",
        ]
    )
    retriever = HybridRetriever(chunks, HashingEmbedder(512), bm25_weight=0.5)
    hits = retriever.retrieve("Liquidity Coverage Ratio LCR", top_k=2)
    assert [h.chunk.chunk_id for h in hits] == ["d:0000", "d:0001"]
    assert hits[0].bm25_rank == 1
    assert hits[0].dense_rank >= 1
    assert hits[0].score > hits[1].score


def test_hybrid_without_embedder_is_pure_bm25() -> None:
    chunks = _chunks(["alpha beta", "gamma delta"])
    retriever = HybridRetriever(chunks, embedder=None)
    assert retriever.bm25_weight == 1.0
    hits = retriever.retrieve("gamma")
    assert [h.chunk.chunk_id for h in hits] == ["d:0001"]
    assert hits[0].dense_rank == 0


def test_hybrid_weight_validation() -> None:
    with pytest.raises(ValueError, match="within"):
        HybridRetriever(_chunks(["x"]), None, bm25_weight=1.5)
