"""End-to-end orchestration: ingest -> index -> retrieve -> extract -> ground."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

from report_rag.catalogue import RISK_METRICS
from report_rag.config import Settings
from report_rag.extraction.base import Extractor
from report_rag.extraction.grounding import GroundingValidator
from report_rag.ingest import chunk_document
from report_rag.retrieval.embeddings import Embedder, HashingEmbedder
from report_rag.retrieval.hybrid import HybridRetriever, Retriever
from report_rag.schemas import Chunk, Document, ExtractionReport, MetricSpec, ValidatedMetric

log = logging.getLogger(__name__)


class RiskExtractionPipeline:
    """Extract a catalogue of risk metrics from one annual report.

    Args:
        extractor: Any :class:`Extractor` (rule-based, LLM, ...).
        settings: Chunking / retrieval settings.
        embedder: Dense embedder for the hybrid retriever; defaults to the
            offline :class:`HashingEmbedder`. Pass ``None`` for pure BM25.
        metrics: Subset of the catalogue to extract (defaults to all).
    """

    def __init__(
        self,
        extractor: Extractor,
        settings: Settings | None = None,
        embedder: Embedder | Literal["default"] | None = "default",
        metrics: Sequence[MetricSpec] = RISK_METRICS,
    ) -> None:
        self.extractor = extractor
        self.settings = settings or Settings()
        self.embedder: Embedder | None = HashingEmbedder() if embedder == "default" else embedder
        self.metrics = tuple(metrics)
        self.validator = GroundingValidator()

    def build_retriever(self, doc: Document) -> tuple[list[Chunk], Retriever]:
        """Chunk a document and index it."""
        chunks = chunk_document(doc, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            msg = f"Document {doc.doc_id} produced no chunks"
            raise ValueError(msg)
        retriever = HybridRetriever(chunks, self.embedder, bm25_weight=self.settings.bm25_weight)
        return chunks, retriever

    def run(self, doc: Document) -> ExtractionReport:
        """Extract every configured metric from ``doc``."""
        _, retriever = self.build_retriever(doc)
        results = [self.extract_one(spec, retriever) for spec in self.metrics]
        return ExtractionReport(
            doc_id=doc.doc_id,
            extractor=_label(self.extractor.name),
            results=results,
        )

    def extract_one(self, spec: MetricSpec, retriever: Retriever) -> ValidatedMetric:
        """Retrieve context for a single metric, extract it and ground-check it."""
        hits = retriever.retrieve(spec.query, top_k=self.settings.top_k)
        chunks = [h.chunk for h in hits]
        log.debug("%s: retrieved %s", spec.metric_id, [c.chunk_id for c in chunks])
        extracted = self.extractor.extract(spec, chunks)
        return self.validator.validate(spec, extracted, {c.chunk_id: c for c in chunks})


def _label(name: str) -> Literal["rules", "llm", "agent"]:
    """Map an extractor name onto the report's closed label set (unknown names count as LLM)."""
    if name == "rules":
        return "rules"
    if name == "agent":
        return "agent"
    return "llm"
